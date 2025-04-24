#!/usr/bin/env python
"""
Enhanced Cr scanning utility
----------------------------
 * Keeps the original Cr‑peak curve creation logic.
 * **NEW**  ‑ Generates an animated GIF that shows the hydrograph
   at the selected gauge (`gauge_idx = 12`) for each Cr value in
   ascending order.  The line colour gradually changes with Cr to
   help visual interpretation.
 * **NEW**  ‑ Before any expensive cluster work we now verify that **all**
   expected CSV result files already exist **and** are non‑empty.  If so,
   we skip the job‑submission + waiting stage entirely and jump straight
   to post‑processing/visualisation.

The remainder of the workflow (directory preparation, job‑array script
writing, etc.) is left intact.

Usage
~~~~~
    python cr_scan.py config_file.j2

Dependencies
~~~~~~~~~~~~
 * numpy, pandas, matplotlib  – as before
 * imageio‑v3 (``pip install imageio``) – for GIF encoding
"""

import os
import sys
import time
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imageio.v3 as iio

from utils import process_yaml
from io_ifc import create_gbl, update_prm_add_or_overwrite_cr

plt.rcParams.update({
    "figure.figsize": (6, 4),
    "font.size": 10,
})

# ────────────────────────────────────────────────────────────────────────────────
# Helper – read hydrograph (identical to original)
# ────────────────────────────────────────────────────────────────────────────────

def read_q_series(csv_path: str) -> np.ndarray:
    """Return **numpy array** of discharge [T × n_gauge] from an IFC CSV."""
    import re, io

    with open(csv_path) as f:
        raw = [ln for ln in f.read().splitlines() if ln.strip()]

    num = re.compile(r"^-?\d")
    while raw and not num.match(raw[0].split(",")[0].strip()):
        raw.pop(0)
    if not raw:
        return np.empty((0, 0))

    df = pd.read_csv(io.StringIO("\n".join(raw)), header=None).dropna(axis=1, how="all")
    
    # Only drop first column if clearly timestamp or non-numeric
    first_col_numeric = pd.to_numeric(df.iloc[:, 0], errors='coerce').notna().all()
    
    if not first_col_numeric:
        df = df.iloc[:, 1:]
    
    return df.to_numpy(float, copy=True)

# ────────────────────────────────────────────────────────────────────────────────
# New – hydrograph GIF generation
# ────────────────────────────────────────────────────────────────────────────────

def make_hydrograph_gif(csv_paths: list[str], cr_values: list[float], out_gif: str,
                        gauge_idx: int = 12) -> None:
    """Create an animated GIF where frames show hydrograph for each Cr.

    Parameters
    ----------
    csv_paths : list[str]
        Paths to CSV files (must align with *cr_values* order).
    cr_values : list[float]
        Corresponding Cr values.
    out_gif   : str
        Output GIF file path.
    gauge_idx : int, default 12
        0‑based column index of gauge to plot.
    """
    frames = []
    y_max = 0.0

    # Pre‑read to normalise axes & compute global ymax
    series_list = []
    for csv in csv_paths:
        q_arr = read_q_series(csv)
        if q_arr.size == 0:
            series_list.append(None)
            continue
        q_series = q_arr[:, gauge_idx]
        series_list.append(q_series)
        y_max = max(y_max, np.nanmax(q_series))

    if y_max == 0:
        print("⚠️  No valid discharge data found – skipping GIF generation.")
        return

    # Build frames
    cmap = plt.get_cmap("viridis")
    for cr, q_series in zip(cr_values, series_list):
        fig, ax = plt.subplots()
        if q_series is not None:
            ax.plot(q_series, color=cmap(cr/ max(cr_values)), lw=1.5)
        ax.set_xlim(0, len(q_series) if q_series is not None else 1)
        ax.set_ylim(0, y_max * 1.05)
        ax.set_xlabel("Timestep (Δt = model output)")
        ax.set_ylabel("Discharge (m³/s)")
        ax.set_title(f"Hydrograph at gauge {gauge_idx} – Cr = {cr:.3f}")
        ax.grid(True, alpha=0.3)

        # Render figure to numpy array and collect as frame
        fig.canvas.draw()                         # 先把图形绘制到缓冲区
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)  # RGBA 字节流
        frame = buf.reshape((h, w, 4))[..., :3]   # 去掉透明度通道，只留 RGB
        frames.append(frame)
        plt.close(fig)

    iio.imwrite(out_gif, frames, duration=0.8, loop=0)
    print("🎞️  Hydrograph GIF saved →", out_gif)

# ────────────────────────────────────────────────────────────────────────────────
# Main driver – mostly unchanged except for *job‑skipping* and *GIF* parts
# ────────────────────────────────────────────────────────────────────────────────

def main(yaml_name: str):
    cfg = process_yaml(yaml_name)

    # 1) 生成 Cr 序列
    cr_list = np.linspace(0, 2.0, 100).tolist()
    n_run = len(cr_list)
    print("Scanning Cr values:", cr_list)

    # 2) 更新目录，随后同步到局部变量
    cfg["tmp_dir"] = cfg["tmp_dir"].rstrip("/") + "_Cr_Scan/"
    cfg["out_dir"] = cfg["out_dir"].rstrip("/") + "_Cr_Scan/"
    tmp_dir: str = cfg["tmp_dir"]
    out_dir: str = cfg["out_dir"]

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # 3) 清理并创建目录
    shutil.rmtree(tmp_dir, ignore_errors=True)
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    # 4) 复制 uini
    shutil.copyfile(cfg["initial_uini"], tmp_dir + "init.uini")
    shutil.copyfile(cfg["link_sav"], tmp_dir + "meas.sav")

    # 5) 批量生成 prm / gbl – same as before, but now track CSV paths earlier
    prm_paths, gbl_paths, expected_csv = [], [], []
    for i, cr in enumerate(cr_list):
        prm_i = f"{tmp_dir}cr_{cr:.3f}.prm"
        gbl_i = f"{tmp_dir}cr_{cr:.3f}.gbl"
        csv_i = f"{tmp_dir}cr_{cr:.3f}.csv"

        shutil.copy(cfg["prm"], prm_i)
        update_prm_add_or_overwrite_cr(prm_i, cr)

        # 写 proto gbl
        create_gbl(cfg, ens=1)
        proto_gbl = f"{tmp_dir}0.gbl"
        with open(proto_gbl) as f:
            lines = f.readlines()
        for k, line in enumerate(lines):
            if line.startswith("0 ") and ".prm" in line:
                lines[k] = "0 " + prm_i + "\n"
            elif line.startswith("2 60 "):
                lines[k] = "2 60 " + csv_i + "\n"
            elif line.strip() == tmp_dir.rstrip("/"):
                lines[k] = tmp_dir + f"_{i}\n"
        with open(gbl_i, "w") as f:
            f.writelines(lines)
        os.remove(proto_gbl)

        prm_paths.append(prm_i)
        gbl_paths.append(gbl_i)
        expected_csv.append(csv_i)

    # ────────────────────────────────────────────────────────────────────
    # 6) 判断 CSV 是否均已存在且非空
    # ────────────────────────────────────────────────────────────────────
    all_done = all(Path(csv).exists() and Path(csv).stat().st_size > 0 for csv in expected_csv)

    if not all_done:
        # === 写 SGE 数组作业脚本并提交 =====================================
        array_job = f"{tmp_dir}submit_cr_scan.job"
        with open(array_job, "w") as f:
            f.write("#!/bin/bash\n"
                    "#$ -N CR_scan\n"
                    "#$ -q IFC\n"
                    "#$ -pe smp 8\n"
                    "#$ -cwd\n"
                    "#$ -o /dev/null\n"
                    "#$ -e /dev/null\n\n"
                    "module reset\n"
                    "module load openmpi\n\n"
                    f"#$ -t 1-{n_run}\n\n"
                    "ID=$(($SGE_TASK_ID-1))\n"
                    "declare -a gbls=(" + " ".join(gbl_paths) + ")\n"
                    "mpirun -np 8 /Users/zli333/DA/2025_EKI/exec/asynch/bin/asynch ${gbls[$ID]}\n")

        print("Submitting job array…")
        os.system(f"qsub {array_job}")

        # === 等待所有输出 CSV ===========================================
        tic = time.time()
        while True:
            done = [Path(csv).exists() and Path(csv).stat().st_size > 0 for csv in expected_csv]
            if all(done):
                break
            sys.stdout.write(f"\r⏳  Waiting… {int(time.time()-tic)} s")
            sys.stdout.flush()
            time.sleep(30)
        print("\n✅  All simulations finished.")
    else:
        print("✅  All CSV already present – skipping simulation.")

    # === 7) 后处理：提峰值 & 写 Cr‑Peak.csv ============================
    gauge_idx = 12  # 0‑based index of the gauge column to use for *all* plots
    peaks = []
    for cr, csv_file in zip(cr_list, expected_csv):
        q_arr = read_q_series(csv_file)
        if q_arr.size == 0:
            print(f"⚠️ {csv_file}   无有效数据")
            continue
        peak_val = np.nanmax(q_arr[:, gauge_idx])
        peaks.append([cr, peak_val])

    peak_df = pd.DataFrame(peaks, columns=["Cr", "Q_peak"])
    peak_df.to_csv(f"{out_dir}cr_peak_curve.csv", index=False)

    # === 8) 画 Cr‑Peak 曲线 ============================================
    plt.figure()
    plt.plot(peak_df["Cr"], peak_df["Q_peak"], "o-", lw=1.5)
    plt.xlabel("Runoff Coefficient Cr")
    plt.ylabel("Peak Discharge (m³/s)")
    plt.title("Cr – Peak flow relationship")
    plt.grid(True, alpha=0.3)
    fig_png = f"{out_dir}cr_peak_curve.png"
    plt.savefig(fig_png, dpi=150)
    print("Figure saved →", fig_png)
    plt.close()

    # === 9) 制作 Hydrograph GIF =======================================
    gif_path = f"{out_dir}cr_hydrograph.gif"
    make_hydrograph_gif(expected_csv, cr_list, gif_path, gauge_idx=gauge_idx)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python cr_scan.py <config.j2>")
        sys.exit(1)
    main(sys.argv[1])
