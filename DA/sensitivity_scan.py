#!/usr/bin/env python
"""
Parameter Sensitivity Scanning Tool (Diagnosis Direction 1)
-----------------------------------------------------------
This script systematically evaluates the sensitivity of the HLM model output
(specifically streamflow) to the rainfall correction factor alpha (i.e., $Cr$).

IMPORTANT ASSUMPTION:
This script assumes the alpha parameter is a GLOBAL value applied uniformly
across all sub-watershed divisions. It is designed to test the overall
model response to a single parameter change, not division-specific variations.

Workflow:
1.  Define a scan range for the alpha parameter (e.g., from 0.8 to 2.8).
2.  For each alpha value in the range:
    a. Generate a unique .prm file using the `create_prm_from_division_params`
       function from the main DA framework.
    b. Generate a corresponding .gbl file and modify it to output .csv hydrographs.
3.  Submit all simulation tasks as an SGE job array to the HPC cluster.
4.  Wait for all simulations to complete.
5.  Post-process the results for each target gauge specified in the config:
    a. Read the output .csv file for each alpha value.
    b. Extract the hydrograph for the target gauge.
    c. Plot the "alpha vs. Peak Flow" relationship.
    d. Generate an animated GIF showing the hydrograph's evolution as alpha changes.

Usage
~~~~~
    python sensitivity_scan.py DA/config.j2

Dependencies
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

# Import necessary utility functions from the current DA project
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
#  File I/O and Plotting Functions
# ────────────────────────────────────────────────────────────────────────────────

def read_q_series(csv_path: str) -> np.ndarray:
    """Read streamflow data from an HLM output CSV file, return as Numpy array [timesteps x n_gauges]"""
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        return np.empty((0, 0))
    with open(csv_path) as f:
        raw = [ln for ln in f.read().splitlines() if ln.strip()]
    # Skip header lines
    num = re.compile(r"^-?\d")
    while raw and not num.match(raw[0].split(",")[0].strip()):
        raw.pop(0)
    if not raw:
        return np.empty((0, 0))
    df = pd.read_csv(io.StringIO("\n".join(raw)), header=None).dropna(axis=1, how="all")
    # Drop the first column if it's a timestamp
    if pd.to_numeric(df.iloc[:, 0], errors='coerce').isna().any():
        df = df.iloc[:, 1:]
    # Drop the last column if it's all zeros or empty (a common HLM artifact)
    if (df.iloc[:, -1] == 0).all() or df.iloc[:, -1].isnull().all():
        df = df.iloc[:, :-1]
    return df.to_numpy(float, copy=True)


def make_hydrograph_gif(csv_paths: list[str], alpha_values: list[float], out_gif: str,
                        gauge_idx: int, gauge_name: str) -> None:
    """Generate an animated GIF of hydrographs for a specific gauge."""
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
        print("⚠️  No valid discharge data found – skipping GIF generation.")
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
    print(f"🎞️  Hydrograph GIF animation saved -> {os.path.basename(out_gif)}")


def create_initial_rec_from_uini(cfg: dict, out_rec_path: str, sorted_link_ids: list):
    """Create a .rec initial state file from a .uini file."""
    initial_uini_path = cfg['hlm_model']['initial_uini']
    with open(initial_uini_path, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
    
    state_values = np.array([float(v) for v in lines[2].split()])
    n_links = len(sorted_link_ids)
    initial_state_matrix = np.tile(state_values, (n_links, 1))
    
    write_rec_file(out_rec_path, cfg['hlm_model']['model_num'], sorted_link_ids, initial_state_matrix)
    print(f"Created common initial state file for all simulations: {out_rec_path}")

# ────────────────────────────────────────────────────────────────────────────────
#  Main Driver Function
# ────────────────────────────────────────────────────────────────────────────────

def main(yaml_name: str):
    cfg = process_yaml(yaml_name)

    # 1) Define Alpha scan range
    alpha_list = np.linspace(0.8, 2.8, 11).tolist()
    n_run = len(alpha_list)
    print(f"Scanning {n_run} Alpha values: {[f'{v:.2f}' for v in alpha_list]}")

    # 2) Set up temporary and output directories
    start_str = cfg['da_settings']['assimilation_window']['start'].split(' ')[0].replace('-', '')
    end_str = cfg['da_settings']['assimilation_window']['end'].split(' ')[0].replace('-', '')
    time_span_str = f"{start_str}-{end_str}"

    # +++ FIX: Subfolder name will now only contain the time span +++
    subfolder_name = f"{time_span_str}"

    base_tmp_dir = cfg["paths"]["tmp_dir"] + "_Sensitivity_Scan"
    base_out_dir = cfg["paths"]["out_dir"] + "_Sensitivity_Scan"

    # The temporary directory for HPC jobs remains at the top level to keep paths simple
    tmp_dir = os.path.join(base_tmp_dir)
    out_dir = os.path.join(base_out_dir, subfolder_name)
    
    # 3) Clean and create directories
    print("Cleaning and creating working directories...")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    # 4) Prepare model structure and common files
    print("Preparing model structure and common files...")
    sorted_link_ids = get_ids(cfg['hlm_model'])
    division_to_link_map, link_to_division_map = get_subwatershed(cfg['hlm_model'], sorted_link_ids)
    n_divisions = division_to_link_map.shape[0]
    cr_param_index = cfg['parameters']['prm_names'].index('$Cr$')

    # Copy the meas.sav file
    meas_sav_path = os.path.join(tmp_dir, "meas.sav")
    shutil.copyfile(cfg["observations"]["link_sav"], meas_sav_path)


    # Create a common initial state .rec file for all simulations
    init_rec_path = os.path.join(tmp_dir, "init.rec")
    create_initial_rec_from_uini(cfg, init_rec_path, sorted_link_ids)


    # 5) Batch generate .prm and .gbl files for each Alpha value
    print("Generating .prm and .gbl files for each Alpha value...")
    gbl_paths, expected_csv = [], []
    for i, alpha in enumerate(alpha_list):
        run_prefix = f"alpha_{alpha:.3f}"
        prm_i = os.path.join(tmp_dir, f"{run_prefix}.prm")
        gbl_i = os.path.join(tmp_dir, f"{run_prefix}.gbl")
        csv_i = os.path.join(tmp_dir, f"{run_prefix}.csv")

        # a. Create .prm file using the function from the main framework
        physical_params = np.full((1, n_divisions), alpha)
        create_prm_from_division_params(
            cfg['hlm_model'],
            link_to_division_map,
            physical_params,
            [cr_param_index],
            prm_i
        )

        # b. Create .gbl file using the function from the main framework
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
            output_rec_path=os.path.join(tmp_dir, f"{run_prefix}.rec"), # Temporary .rec output, will not be used
            sav_file_path=meas_sav_path,
            scratch_dir_path=os.path.join(cfg['hlm_model']['scratch_dir'], f"scan_{i}"),
            target_env='login'
        )
        
        # c. CRITICAL MODIFICATION: Modify the GBL to output CSV instead of REC
        with open(gbl_i, 'r') as f:
            lines = f.readlines()
        
        # Use a robust method to locate and modify lines by finding section headers
        new_lines = list(lines)
        for k, line in enumerate(lines):
            # Locate hydrograph output section
            if line.strip() == "%Where to put write hydrographs":
                # The data line is 2 lines below the header
                if k + 2 < len(new_lines):
                    new_lines[k+2] = f"2 60 {os.path.abspath(csv_i)}\n"

            # Locate snapshot output section
            if line.strip().startswith("%Snapshot information"):
                # The data line is 1 line below the header
                if k + 1 < len(new_lines):
                    new_lines[k+1] = "0\n" # "0" means no snapshot output

        with open(gbl_i, 'w') as f:
            f.writelines(new_lines)

        gbl_paths.append(gbl_i)
        expected_csv.append(csv_i)

    # 6) Check if results already exist, otherwise submit HPC job
    all_done = False
    print("📢  Debug mode: Forcing re-run of all HPC simulations...")

    if not all_done:
        # Write and submit the SGE job array script
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

        print("Submitting HPC job array...")
        os.system(f"qsub {array_job_path}")

        # Wait for all output CSV files to be generated
        tic = time.time()
        while True:
            done_count = sum(1 for csv in expected_csv if Path(csv).exists() and Path(csv).stat().st_size > 100)
            if done_count == n_run:
                break
            sys.stdout.write(f"\r⏳  Waiting for simulations to complete... ({done_count}/{n_run}) - {int(time.time()-tic)} s")
            sys.stdout.flush()
            time.sleep(30)
        print("\n✅  All simulations have completed.")
    else:
        print("✅  All result CSV files already exist – skipping simulation.")

    print("\n--- Starting Post-Processing for Target Gauges ---")
    target_gauges = cfg.get('visualization', {}).get('sensitivity_gauges', [])
    if not target_gauges:
        print("⚠️  'sensitivity_gauges' list not found in config.j2 under visualization section. Skipping post-processing.")
        return

    # Load mappings once before the loop
    from io_ifc import load_usgs_mapping
    usgs_map, _, _ = load_usgs_mapping(cfg['observations'])
    with open(meas_sav_path, 'r') as f:
        sav_lids = [line.strip() for line in f if line.strip()]

    for target_gauge_id in target_gauges:
        print(f"\n--- Processing Gauge: {target_gauge_id} ---")
        try:
            target_link_id_str = str(usgs_map[target_gauge_id])
            gauge_idx = sav_lids.index(target_link_id_str)
            print(f"  > Target gauge '{target_gauge_id}' (Link ID: {target_link_id_str}) corresponds to column {gauge_idx} in output files.")
        except (KeyError, ValueError):
            print(f"  > ⚠️ ERROR: Could not find target gauge {target_gauge_id} in mappings or .sav file. Skipping this gauge.")
            continue

        # 7) Post-process: Extract peak flows and write to CSV
        peaks = []
        for alpha, csv_file in zip(alpha_list, expected_csv):
            q_arr = read_q_series(csv_file)
            if q_arr.size == 0 or q_arr.shape[1] <= gauge_idx:
                print(f"  > ⚠️ No valid data or gauge column in {os.path.basename(csv_file)}.")
                peaks.append([alpha, np.nan])
                continue
            peak_val = np.nanmax(q_arr[:, gauge_idx])
            peaks.append([alpha, peak_val])

        peak_df = pd.DataFrame(peaks, columns=["Alpha", "Q_peak"])
        csv_path = os.path.join(out_dir, f"alpha_peak_curve_gauge_{target_gauge_id}.csv")
        peak_df.to_csv(csv_path, index=False)
        print(f"  > Peak data saved -> {os.path.basename(csv_path)}")

        # 8) Plot: Alpha vs. Peak Flow
        plt.figure()
        plt.plot(peak_df["Alpha"], peak_df["Q_peak"], "o-", lw=2, markersize=8)
        plt.xlabel("Rainfall Correction Factor Alpha ($Cr$)")
        plt.ylabel(f"Peak Discharge (m³/s)")
        plt.title(f"Sensitivity of Peak Discharge to Alpha @ Gauge {target_gauge_id}")
        plt.grid(True, alpha=0.5, linestyle='--')
        plt.tight_layout()
        fig_png = os.path.join(out_dir, f"alpha_peak_curve_gauge_{target_gauge_id}.png")
        plt.savefig(fig_png, dpi=200)
        print(f"  > 📊 Sensitivity curve plot saved -> {os.path.basename(fig_png)}")
        plt.close()
        
        # 9) Generate hydrograph GIF
        gif_path = os.path.join(out_dir, f"alpha_hydrograph_animation_gauge_{target_gauge_id}.gif")
        make_hydrograph_gif(expected_csv, alpha_list, gif_path, gauge_idx=gauge_idx, gauge_name=target_gauge_id)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {os.path.basename(__file__)} <config.j2>")
        sys.exit(1)
    main(sys.argv[1])