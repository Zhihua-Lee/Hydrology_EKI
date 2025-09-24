#!/usr/bin/env python
"""
模型参数敏感性扫描工具 (诊断方向一)
---------------------------------
本脚本用于系统性地评估HLM模型输出（特别是径流）对降雨校正因子 alpha (即 $Cr$) 的敏感性。

工作流程:
1.  定义一个 alpha 值的扫描范围 (例如从 0.8 到 1.8)。
2.  为范围内的每一个 alpha 值：
    a. 使用新DA框架中的 `create_prm_from_division_params` 函数生成一个独立的 .prm 文件。
    b. 使用新DA框架中的 `_create_single_gbl` 函数生成一个 .gbl 文件，并修改它以输出 .csv 径流文件。
3.  将所有模拟任务作为一个SGE作业数组提交到HPC集群。
4.  等待所有模拟完成。
5.  后处理：
    a. 读取每个 alpha 值对应的输出 .csv 文件。
    b. 提取指定观测点的径流过程线。
    c. 绘制 "alpha vs. 洪峰流量" 关系图。
    d. 生成一个动画GIF，直观展示径流过程线随 alpha 变化的动态过程。

用法
~~~~~
    python sensitivity_scan.py DA/config.j2

依赖
~~~~~~~~~~~~
 * numpy, pandas, matplotlib
 * imageio (pip install imageio)
"""

import os
import sys
import time
import shutil
from pathlib import Path
import re
import io

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imageio.v3 as iio

# 从当前DA项目中导入必要的工具函数
from utils import process_yaml
from io_ifc import (
    get_ids,
    get_subwatershed,
    create_prm_from_division_params,
    _create_single_gbl,
    write_rec_file
)

plt.rcParams.update({
    "figure.figsize": (8, 5),
    "font.size": 12,
})

# ────────────────────────────────────────────────────────────────────────────────
#  文件读取与绘图函数 (与旧脚本基本一致)
# ────────────────────────────────────────────────────────────────────────────────

def read_q_series(csv_path: str) -> np.ndarray:
    """从HLM输出的CSV文件中读取径流数据，返回Numpy数组 [时间步 x 观测点数]"""
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        return np.empty((0, 0))
    with open(csv_path) as f:
        raw = [ln for ln in f.read().splitlines() if ln.strip()]
    # 跳过文件头
    num = re.compile(r"^-?\d")
    while raw and not num.match(raw[0].split(",")[0].strip()):
        raw.pop(0)
    if not raw:
        return np.empty((0, 0))
    df = pd.read_csv(io.StringIO("\n".join(raw)), header=None).dropna(axis=1, how="all")
    # 如果第一列是时间戳，则丢弃
    if pd.to_numeric(df.iloc[:, 0], errors='coerce').isna().any():
        df = df.iloc[:, 1:]
    # 如果最后一列是全0或空的（HLM常见问题），则丢弃
    if (df.iloc[:, -1] == 0).all() or df.iloc[:, -1].isnull().all():
        df = df.iloc[:, :-1]
    return df.to_numpy(float, copy=True)


def make_hydrograph_gif(csv_paths: list[str], alpha_values: list[float], out_gif: str,
                        gauge_idx: int, gauge_name: str) -> None:
    """为指定观测点生成径流过程线的GIF动画"""
    frames = []
    y_max = 0.0
    series_list = []
    for csv in csv_paths:
        q_arr = read_q_series(csv)
        if q_arr.size == 0 or q_arr.shape[1] <= gauge_idx:
            series_list.append(None)
            continue
        q_series = q_arr[:, gauge_idx]
        series_list.append(q_series)
        y_max = max(y_max, np.nanmax(q_series))

    if y_max == 0:
        print("⚠️  未找到有效的径流数据 – 跳过GIF生成。")
        return

    cmap = plt.get_cmap("viridis")
    for alpha, q_series in zip(alpha_values, series_list):
        fig, ax = plt.subplots()
        if q_series is not None:
            color_val = (alpha - min(alpha_values)) / (max(alpha_values) - min(alpha_values))
            ax.plot(q_series, color=cmap(color_val), lw=2)
        ax.set_xlim(0, len(q_series) if q_series is not None else 1)
        ax.set_ylim(0, y_max * 1.05)
        ax.set_xlabel("Simulation Timestep (hour)")
        ax.set_ylabel("Discharge (m³/s)")
        ax.set_title(f"Hydrograph at Gauge {gauge_name}\nAlpha = {alpha:.3f}")
        ax.grid(True, alpha=0.4, linestyle='--')
        fig.tight_layout()

        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        frame = buf.reshape((h, w, 4))[..., :3]
        frames.append(frame)
        plt.close(fig)

    iio.imwrite(out_gif, frames, duration=150, loop=0) # duration in ms
    print("🎞️  径流GIF动画已保存 ->", out_gif)


def create_initial_rec_from_uini(cfg: dict, out_rec_path: str, sorted_link_ids: list):
    """从 .uini 文件创建一个 .rec 初始状态文件"""
    initial_uini_path = cfg['hlm_model']['initial_uini']
    with open(initial_uini_path, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
    
    state_values = np.array([float(v) for v in lines[2].split()])
    n_links = len(sorted_link_ids)
    initial_state_matrix = np.tile(state_values, (n_links, 1))
    
    write_rec_file(out_rec_path, cfg['hlm_model']['model_num'], sorted_link_ids, initial_state_matrix)
    print(f"创建了所有模拟共用的初始状态文件: {out_rec_path}")

# ────────────────────────────────────────────────────────────────────────────────
#  主驱动函数
# ────────────────────────────────────────────────────────────────────────────────

def main(yaml_name: str):
    cfg = process_yaml(yaml_name)

    # 1) 定义 Alpha 扫描范围
    # 包含初始均值(1.0)和真值(1.5)附近的值
    alpha_list = np.linspace(0.8, 2.8, 11).tolist()
    n_run = len(alpha_list)
    print(f"扫描 {n_run} 个 Alpha 值: {[f'{v:.2f}' for v in alpha_list]}")

    # 2) 设置临时和输出目录
    tmp_dir = os.path.join(cfg["paths"]["tmp_dir"] + "_Sensitivity_Scan")
    out_dir = os.path.join(cfg["paths"]["out_dir"] + "_Sensitivity_Scan")
    
    # 3) 清理并创建目录
    print("正在清理和创建工作目录...")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    # 4) 准备模型结构和公共文件
    print("正在准备模型结构和公共文件...")
    sorted_link_ids = get_ids(cfg['hlm_model'])
    division_to_link_map, link_to_division_map = get_subwatershed(cfg['hlm_model'], sorted_link_ids)
    n_divisions = division_to_link_map.shape[0]
    cr_param_index = cfg['parameters']['prm_names'].index('$Cr$')

    # 复制 meas.sav 并找到目标观测点的索引
    meas_sav_path = os.path.join(tmp_dir, "meas.sav")
    shutil.copyfile(cfg["observations"]["link_sav"], meas_sav_path)
    
    target_gauge_lid = '5583000' # 下游出口观测站
    with open(meas_sav_path, 'r') as f:
        sav_lids = [line.strip() for line in f if line.strip()]
    try:
        # 在 .sav 文件中找到目标观测点对应的link ID，再确定其列索引
        from io_ifc import load_usgs_mapping
        usgs_map, _, _ = load_usgs_mapping(cfg['observations'])
        target_link_id_str = str(usgs_map[target_gauge_lid])
        gauge_idx = sav_lids.index(target_link_id_str)
        print(f"目标观测点 '{target_gauge_lid}' (Link ID: {target_link_id_str}) 在输出文件的第 {gauge_idx} 列。")
    except (KeyError, ValueError):
        print(f"错误：无法在映射或.sav文件中找到目标观测点 {target_gauge_lid}。将使用默认索引 0。")
        gauge_idx = 0
        target_gauge_lid = f"Index_{gauge_idx}"


    # 创建一个所有模拟共用的初始状态 .rec 文件
    init_rec_path = os.path.join(tmp_dir, "init.rec")
    create_initial_rec_from_uini(cfg, init_rec_path, sorted_link_ids)


    # 5) 批量为每个 Alpha 值生成 prm 和 gbl 文件
    print("正在为每个 Alpha 值生成 .prm 和 .gbl 文件...")
    gbl_paths, expected_csv = [], []
    for i, alpha in enumerate(alpha_list):
        run_prefix = f"alpha_{alpha:.3f}"
        prm_i = os.path.join(tmp_dir, f"{run_prefix}.prm")
        gbl_i = os.path.join(tmp_dir, f"{run_prefix}.gbl")
        csv_i = os.path.join(tmp_dir, f"{run_prefix}.csv")

        # a. 使用新框架的函数创建 .prm 文件
        physical_params = np.full((1, n_divisions), alpha)
        create_prm_from_division_params(
            cfg['hlm_model'],
            link_to_division_map,
            physical_params,
            [cr_param_index],
            prm_i
        )

        # b. 使用新框架的函数创建 .gbl 文件
        gbl_config = cfg['hlm_model'].copy()
        gbl_config.update({
            "time_start": cfg['da_settings']['assimilation_window']['start'],
            "time_end": cfg['da_settings']['assimilation_window']['end'],
            "model_num": cfg['hlm_model']['model_num'],
            'login_node_root': cfg.get('login_node_root'),
            'compute_node_root': cfg.get('compute_node_root')
        })
        _create_single_gbl(
            test_dict=gbl_config,
            output_gbl_path=gbl_i,
            prm_file_path=prm_i,
            input_rec_path=init_rec_path,
            output_rec_path=os.path.join(tmp_dir, f"{run_prefix}.rec"), # 临时rec输出，不会被使用
            sav_file_path=meas_sav_path,
            scratch_dir_path=os.path.join(cfg['hlm_model']['scratch_dir'], f"scan_{i}"),
            target_env='login'
        )
        
        # c. **关键修改**: 修改GBL文件，使其输出CSV而不是REC (已修正格式问题)
        with open(gbl_i, 'r') as f:
            lines = f.readlines()
        
        # 使用更鲁棒的逻辑来定位和修改行
        # 这种方法通过寻找节标题，然后修改其后固定偏移量的行，避免了注释行数变化带来的错误
        new_lines = list(lines)
        for k, line in enumerate(lines):
            # 定位水文图输出节
            if line.strip() == "%Where to put write hydrographs":
                # 该节的数据行在标题后第2行 (标题 -> 注释 -> 数据行)
                if k + 2 < len(new_lines):
                    new_lines[k+2] = f"2 60 {os.path.abspath(csv_i)}\n"

            # 定位状态快照节
            if line.strip().startswith("%Snapshot information"):
                # 该节的数据行就在标题后第1行
                if k + 1 < len(new_lines):
                    new_lines[k+1] = "0\n" # "0" 表示不输出

        with open(gbl_i, 'w') as f:
            f.writelines(new_lines)

        # --- 关键修复：将生成的文件路径添加到列表中 ---
        gbl_paths.append(gbl_i)
        expected_csv.append(csv_i)

    # 6) 检查结果是否已存在，否则提交HPC作业
    # all_done = all(Path(csv).exists() and Path(csv).stat().st_size > 100 for csv in expected_csv)
    # 为了调试，我们暂时禁用检查，强制重新运行所有模拟
    all_done = False
    print("📢  调试模式：强制重新运行所有HPC模拟...")

    if not all_done:
        # 编写并提交 SGE 作业数组脚本
        array_job_path = os.path.join(tmp_dir, "submit_sensitivity_scan.job")
        executable_path = os.path.join(cfg['login_node_root'], 'exec/asynch/bin/asynch')
        with open(array_job_path, "w") as f:
            f.write("#!/bin/bash\n"
                    "#$ -N Alpha_Sensitivity_Scan\n"
                    "#$ -q IFC\n"
                    f"#$ -pe {cfg['hlm_model']['parallel_argument']} {cfg['hlm_model']['num_parallel_slots']}\n"
                    "#$ -cwd\n"
                    "#$ -j y\n"
                    f"#$ -o {tmp_dir}/$JOB_ID.$SGE_TASK_ID.out\n"
                    "#$ -e {tmp_dir}/$JOB_ID.$SGE_TASK_ID.err\n\n"
                    "module reset\n"
                    "module load openmpi\n\n"
                    f"#$ -t 1-{n_run}\n\n"
                    "ID=$(($SGE_TASK_ID-1))\n"
                    "declare -a gbls=(" + " ".join(gbl_paths) + ")\n"
                    f"mpirun -np {cfg['hlm_model']['num_parallel_slots']} {executable_path} ${{gbls[$ID]}}\n")

        print("正在提交HPC作业数组...")
        os.system(f"qsub {array_job_path}")

        # 等待所有输出CSV文件生成
        tic = time.time()
        while True:
            done_count = sum(1 for csv in expected_csv if Path(csv).exists() and Path(csv).stat().st_size > 100)
            if done_count == n_run:
                break
            sys.stdout.write(f"\r⏳  等待模拟完成... ({done_count}/{n_run}) - {int(time.time()-tic)} s")
            sys.stdout.flush()
            time.sleep(30)
        print("\n✅  所有模拟已完成。")
    else:
        print("✅  所有结果CSV文件均已存在 – 跳过模拟。")

    # 7) 后处理：提取洪峰并写入CSV
    print("正在后处理结果...")
    peaks = []
    for alpha, csv_file in zip(alpha_list, expected_csv):
        q_arr = read_q_series(csv_file)
        if q_arr.size == 0 or q_arr.shape[1] <= gauge_idx:
            print(f"⚠️  在 {os.path.basename(csv_file)} 中无有效数据或找不到观测点列。")
            peaks.append([alpha, np.nan])
            continue
        peak_val = np.nanmax(q_arr[:, gauge_idx])
        peaks.append([alpha, peak_val])

    peak_df = pd.DataFrame(peaks, columns=["Alpha", "Q_peak"])
    peak_df.to_csv(os.path.join(out_dir, "alpha_peak_curve.csv"), index=False)

    # 8) 绘图：Alpha vs. 洪峰流量
    plt.figure()
    plt.plot(peak_df["Alpha"], peak_df["Q_peak"], "o-", lw=2, markersize=8)
    plt.xlabel("Rainfall Correction Factor Alpha ($Cr$)")
    plt.ylabel(f"Peak Discharge (m³/s) @ Gauge {target_gauge_lid}")
    plt.title("Sensitivity of Peak Discharge to Alpha Parameter")
    plt.grid(True, alpha=0.5, linestyle='--')
    plt.tight_layout()
    fig_png = os.path.join(out_dir, "alpha_peak_curve.png")
    plt.savefig(fig_png, dpi=200)
    print("📊  Sensitivity curve plot saved ->", fig_png)
    plt.close()
    
    # 9) 制作径流过程线GIF
    gif_path = os.path.join(out_dir, "alpha_hydrograph_animation.gif")
    make_hydrograph_gif(expected_csv, alpha_list, gif_path, gauge_idx=gauge_idx, gauge_name=target_gauge_lid)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"用法: python {os.path.basename(__file__)} <config.j2>")
        sys.exit(1)
    main(sys.argv[1])