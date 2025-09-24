### hpc_da_tasks.py
import os
import sys
import pickle
import time
import subprocess
import numpy as np
from typing import List
import shutil
from string import Template
from textwrap import dedent

from io_ifc import parse_rec_file, create_prm_from_division_params, write_rec_file
from utils import time_to_epoch

# ==============================================================================
# === Presimulation Tasks (for Synthetic Data Generation)
# ==============================================================================

def _create_presim_gbl_file(config: dict, presim_prm_path: str, output_csv_path: str, presim_gbl_path: str):
    """Creates a GBL file for the presimulation run, using relative paths."""
    hlm_config = config['hlm_model']
    da_config = config['da_settings']
    
    gbl_template_str = dedent("""
        %Model UID
        $MODEL_NUM

        %Begin and end date time
        $START_TIME 
        $END_TIME

        0\t%Parameters to filenames

        %Components to print
        1
        State0

        %Peakflow function
        Classic

        %Global parameters
        %9 v_0   lambda_1 lambda_2 Hu(mm)   infil(mm/hr) perc(mm/hr)  res_surf[minutes]  res_subsurf[days]  res_gw[days]
        $GLOBAL_PARAMS

        %No. steps stored at each link and
        %Max no. steps transfered between procs
        %Discontinuity buffer size
        30 10 30

        %Topology (0 = .rvr, 1 = database)
        0 $RVR_FILE

        %DEM Parameters (0 = .prm, 1 = database)
        0 $PRM_FILE

        %Initial state (0 = .ini, 1 = .uini, 2 = .rec, 3 = .dbc, 3 = .h5)
        2 $INPUT_REC_FILE

        %Forcings (0 = none, 1 = .str, 2 = binary, 3 = database, 4 = .ustr, 5 = forecasting, 6 = .gz binary, 7 = recurring)
        3

        %Rain
        5 $RAIN_DIR
        10 60 $EPOCH_START $EPOCH_END

        %Evaporation
        7 $EVAPO_FILE
        $EPOCH_START $EPOCH_END

        %Temperature 
        7 $TEMP_FILE
        $EPOCH_START $EPOCH_END

        %Dam (0 = no dam, 1 = .dam, 2 = .qvs)
        0

        %Reservoir ids (0 = no reservoirs, 1 = .rsv, 2 = .dbc file)
        0

        %Where to put write hydrographs
        %(0 = no output, 1 = .dat file, 2 = .csv file, 3 = database, 5 = .h5)
        2 60 $CSV_FILE

        %Where to put peakflow data
        %(0 = no output, 1 = .pea file, 2 = database)
        0 

        %.sav files for hydrographs and peak file (meas.sav)
        %(0 = save no data, 1 = .sav file, 2 = .dbc file, 3 = all links)
        1 $SAV_FILE
        0

        %Snapshot information (0 = none, 1 = .rec single, 2 = .rec multiple, ...)
        0

        %Filename for scratch work
        $HPC_SCRATCH_DIR

        %Numerical solver settings follow

        %facmin, facmax, fac
        .1 10.0 .9

        %Solver flag (0 = data below, 1 = .rkd)
        0
        %Numerical solver index (0-3 explicit, 4 = implicit)
        2
        %Error tolerances (abs, rel, abs dense, rel dense)
        1E-2 1E-2 1E-2 1E-2 1E-2 1E-2 1E-2 1E-2 1E-2 1E-2
        1E-2 1E-2 1E-2 1E-2 1E-2 1E-2 1E-2 1E-2 1E-2 1E-2
        1E-2 1E-2 1E-2 1E-2 1E-2 1E-2 1E-2 1E-2 1E-2 1E-2
        1E-2 1E-2 1E-2 1E-2 1E-2 1E-2 1E-2 1E-2 1E-2 1E-2

        # %End of file
    """)
    
    start_time = da_config['assimilation_window']['start']
    end_time = da_config['assimilation_window']['end']
    
    template_vars = {
        "MODEL_NUM": hlm_config["model_num"], "START_TIME": start_time, "END_TIME": end_time,
        "GLOBAL_PARAMS": "11 1 50 3 1 20 35 0 5 0 20 1.0",
        "RVR_FILE": hlm_config['rvr'],
        "PRM_FILE": presim_prm_path,
        "INPUT_REC_FILE": os.path.join(os.path.dirname(presim_prm_path), "presim_initial.rec"),
        "RAIN_DIR": hlm_config['rain_dir'],
        "EVAPO_FILE": hlm_config['evapo'],
        "TEMP_FILE": hlm_config['temp'],
        "EPOCH_START": str(int(time_to_epoch(start_time))), 
        "EPOCH_END": str(int(time_to_epoch(end_time))),
        "CSV_FILE": output_csv_path,
        "SAV_FILE": os.path.join(config['paths']['tmp_dir'], "meas.sav"),
        "HPC_SCRATCH_DIR": os.path.join(hlm_config['scratch_dir'], "presim_run"),
    }
    
    gbl_content = Template(gbl_template_str).safe_substitute(template_vars)
    with open(presim_gbl_path, "w") as f:
        f.write(gbl_content)

def _create_presim_job_script(config: dict, job_script_path: str, gbl_path: str):
    """Creates the HPC shell script for the presimulation run."""
    # Use specialized settings for the large presimulation job
    hpc_settings = config['hlm_model']['presim_hpc_settings']
    num_slots = hpc_settings['num_parallel_slots']
    mem_req = hpc_settings['memory_request']
    scratch_dir = os.path.join(config['hlm_model']['scratch_dir'], "presim_run")
    executable_path = os.path.join(config['login_node_root'], 'exec/asynch/bin/asynch')
    with open(job_script_path, 'w') as f:
        f.write('#!/bin/bash -l\n') # -l ensures login environment is sourced
        f.write('#$ -N DA_presimulation_job\n')
        f.write(f'#$ -pe {config["hlm_model"]["parallel_argument"]} {num_slots}\n')
        f.write(f'#$ -l mf={mem_req}\n')
        f.write('#$ -q IFC\n')
        f.write('#$ -j y\n')
        # Execute from the directory where the GBL file is, so relative paths work
        f.write(f'#$ -cwd \n')
        f.write(f'#$ -o {os.path.join(config["paths"]["tmp_dir"], "hpc_logs")}/presim.out\n')
        f.write(f'#$ -e {os.path.join(config["paths"]["tmp_dir"], "hpc_logs")}/presim.err\n')
        f.write('\nmodule reset\nmodule load openmpi\n\n')
        f.write(f'mkdir -p "{scratch_dir}"\n\n')
        f.write(f'mpirun -np {num_slots} {executable_path} {gbl_path}\n')
        f.write(f'\nrm -r "{scratch_dir}"\n')

def _create_initial_rec_file(config: dict, hlm_runner, output_rec_path: str):
    """Creates a .rec file from the initial .uini file for the presimulation."""
    initial_uini_path = config['hlm_model']['initial_uini']
    with open(initial_uini_path, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
    
    state_values = np.array([float(v) for v in lines[2].split()])
    if len(state_values) != 5:
        raise ValueError(f"Expected 5 state values in {initial_uini_path}, but found {len(state_values)}.")

    n_links = hlm_runner.n_links
    # Create a full state matrix by tiling the single row of initial states for all links
    initial_state_matrix = np.tile(state_values, (n_links, 1))
    
    write_rec_file(output_rec_path, config['hlm_model']['model_num'], hlm_runner.sorted_link_ids, initial_state_matrix)
    print(f"Created initial .rec file for presimulation at: {output_rec_path}")

def _wait_for_files(file_paths: List[str], timeout: int = 1800, tmp_dir: str = None, read_content: bool = True) -> List[np.ndarray]:
    """
    Waits for a list of files to exist, be non-empty, and have consistent sizes.
    Can also check for corresponding .error files.
    If `read_content` is False, it only checks for file existence.
    """
    start_time = time.time()
    num_files = len(file_paths)

    while True:
        elapsed = int(time.time() - start_time)
        if elapsed > timeout:
            missing_files = [p for p in file_paths if not os.path.isfile(p)]
            raise TimeoutError(f"Timeout ({timeout}s) exceeded. Missing {len(missing_files)} files: {missing_files[:5]}")

        if tmp_dir:
            worker_error_files = [f for f in os.listdir(tmp_dir) if f.endswith('.input.error')]
            if worker_error_files:
                error_msgs = []
                for err_file in worker_error_files:
                    with open(os.path.join(tmp_dir, err_file), 'r') as f:
                        error_msgs.append(f"Error in DA input worker (see {err_file}):\n---\n{f.read()}\n---")
                raise RuntimeError("Errors detected in DA input generation workers:\n" + "\n".join(error_msgs))

            sge_error_files = [f for f in os.listdir(tmp_dir) if f.endswith('.err') and os.path.getsize(os.path.join(tmp_dir, f)) > 0]
            if sge_error_files:
                error_msgs = []
                for err_file in sge_error_files:
                    with open(os.path.join(tmp_dir, err_file), 'r') as f:
                        error_msgs.append(f"Error in SGE job (see {err_file}):\n---\n{f.read()}\n---")
                raise RuntimeError("Errors detected in SGE jobs:\n" + "\n".join(error_msgs))

        missing_indices = [i for i, p in enumerate(file_paths) if not os.path.isfile(p)]
        if missing_indices:
            msg = f"⏳ {elapsed}s: Waiting for {len(missing_indices)}/{num_files} files to appear..."
            sys.stdout.write("\r" + msg + " "*20)
            sys.stdout.flush()
            time.sleep(10)
            continue
        
        # If we only need to check for existence.
        if not read_content:
            print(f"\n✅ All {num_files} required files have been created after {elapsed} seconds.")
            return
            
        read_values = []
        error_indices = []
        empty_indices = []
        for i, p in enumerate(file_paths):
            try:
                # Custom handling for CSV vs other files
                if p.endswith('.csv'):
                    # Presimulation CSVs may have headers to strip
                    with open(p, 'r') as f:
                        lines = f.readlines()
                    if len(lines) > 2:
                        # The actual data starts from the 3rd line.
                        # We can process it in memory without creating a temp file.
                        data_str = "".join(lines[2:])
                        data = np.genfromtxt(data_str.splitlines(), delimiter=',')
                        if data.size == 0:
                            empty_indices.append(i)
                        else:
                            # Also remove the last column if it's all zeros (common HLM artifact)
                            if data.ndim == 2 and np.all(data[:, -1] == 0):
                                data = data[:, :-1]
                            read_values.append(data)
                    else: # File is too short to contain data
                        empty_indices.append(i)
                else:
                    # Default behavior for other files (e.g., .rec)
                    data = np.load(p) if p.endswith('.npy') else np.genfromtxt(p)
                    if data.size == 0:
                        empty_indices.append(i)
                    read_values.append(data)

            except Exception as e:
                sys.stdout.write(f"\r❌ Error reading {p}: {e}. Retrying... {' '*20}")
                sys.stdout.flush()
                time.sleep(10)
                break
        else: # This block runs only if the for loop completes without `break`
            if empty_indices:
                msg = f"⏳ {elapsed}s: {len(empty_indices)}/{num_files} files are empty. Waiting..."
                sys.stdout.write("\r" + msg + " "*20)
                sys.stdout.flush()
                time.sleep(10)
                continue

            sizes = {arr.shape for arr in read_values if arr.ndim > 0}
            if len(sizes) > 1:
                msg = f"⏳ {elapsed}s: File shape mismatch. Shapes: {sizes}. Waiting..."
                sys.stdout.write("\r" + msg + " "*20)
                sys.stdout.flush()
                time.sleep(10)
                continue
            
            print(f"\n✅ All {num_files} files are ready after {elapsed} seconds.")
            return read_values

def run_hpc_presimulation(config: dict, hlm_runner):
    """
    Orchestrates the presimulation run to generate synthetic observation data.
    """
    print("\n--- Starting HPC Task: Presimulation for Synthetic Data Generation ---")
    tmp_dir = config['paths']['tmp_dir']
    presim_dir = os.path.join(tmp_dir, 'presim_run')
    os.makedirs(presim_dir, exist_ok=True)

    # 1. Create the PRM file with the truth parameter value
    truth_alpha = config['parameters']['truth_alpha_value']
    truth_params_active = np.full((1, hlm_runner.n_divisions), truth_alpha)
    active_param_indices = [hlm_runner.cr_param_index]
    
    presim_prm_path = os.path.join(presim_dir, "presim.prm")
    create_prm_from_division_params(
        config['hlm_model'], hlm_runner.link_to_division_map, 
        truth_params_active, active_param_indices, presim_prm_path
    )
    print(f"Created presimulation PRM file with alpha={truth_alpha} at {presim_prm_path}")

    # 2. Create the initial .rec file needed for the simulation
    presim_rec_path = os.path.join(presim_dir, "presim_initial.rec")
    _create_initial_rec_file(config, hlm_runner, presim_rec_path)

    # 3. Create the GBL file pointing to the correct inputs/outputs
    output_csv_path = config['da_settings']['synthetic_obs_path']
    if os.path.exists(output_csv_path):
        os.remove(output_csv_path) # Ensure clean start
    presim_gbl_path = os.path.join(presim_dir, "presim.gbl")
    _create_presim_gbl_file(config, presim_prm_path, output_csv_path, presim_gbl_path)

    # 4. Create and submit the HPC job script
    job_script_path = os.path.join(presim_dir, "submit_presim.sh")
    _create_presim_job_script(config, job_script_path, presim_gbl_path)
    
    job_cmd = f"qsub {job_script_path}"
    print(f"Submitting presimulation job: {job_cmd}")
    subprocess.run(job_cmd, shell=True, check=True)

    # 5. Wait for the synthetic data file to be created and process it
    _wait_for_files([output_csv_path], timeout=3600, read_content=True) # Wait and also strip headers
    print(f"Synthetic observation data successfully generated at: {output_csv_path}")

def create_hpc_job_script(config: dict, tmp_dir: str):
    """(Forecast Operator)Creates the batch job file for submitting an HPC job array."""
    hlm_config = config['hlm_model']
    # Correctly point to the subdirectories
    log_dir = os.path.join(tmp_dir, 'hpc_logs') # Log files go here
    forecast_files_dir = os.path.join(tmp_dir, 'forecast_files') # GBL files are here
    job_file_path = os.path.join(tmp_dir, 'submit_forecast_ensemble.job')

    with open(job_file_path, 'w') as f:
        f.write('#!/bin/bash\n')
        f.write('#$ -N DA_ensemble_forecast\n')
        f.write(f'#$ -pe {hlm_config["parallel_argument"]} {hlm_config["num_parallel_slots"]}\n')
        f.write('#$ -q IFC\n')
        f.write('#$ -cwd\n')
        f.write('#$ -j y\n') # Joins stdout and stderr
        # f.write(f'#$ -o /dev/null\n')
        # f.write(f'#$ -e /dev/null\n')
        f.write(f'#$ -o {log_dir}/$JOB_ID.$SGE_TASK_ID.forecast.out\n') # Enable logging for debug
        f.write(f'#$ -e {log_dir}/$JOB_ID.$SGE_TASK_ID.forecast.err\n')
        f.write('\n')
        f.write('module reset\nmodule load openmpi\n\n')
        f.write('ensemble_id=$(($SGE_TASK_ID - 1))\n')
        f.write(f'scratch_path="{hlm_config["scratch_dir"]}/$ensemble_id"\n')
        f.write('mkdir -p "$scratch_path"\n\n')

        # Path must be valid on the compute node
        compute_root = config['compute_node_root']
        executable_path = os.path.join(compute_root, 'exec/asynch/bin/asynch')
        f.write(f'mpirun -np {hlm_config["num_parallel_slots"]} {executable_path} {forecast_files_dir}/$ensemble_id.gbl\n')
        f.write('\nrm -r "$scratch_path"\n')
    return job_file_path

def create_hpc_prepare_script(config: dict, tmp_dir: str):
    """Creates the batch job file for submitting the parallel file preparation job array."""
    # Correctly point to the subdirectories
    compute_root = config['compute_node_root']
    log_dir = os.path.join(tmp_dir, 'hpc_logs')
    job_file_path = os.path.join(tmp_dir, 'submit_prepare_forecast_inputs.job')
    worker_script_path = os.path.join(compute_root, 'DA', 'generate_da_input_worker.py')
    python_executable = config['hpc_python_path']
    hlm_config = config['hlm_model']

    with open(job_file_path, 'w') as f:
        f.write('#!/bin/bash\n')
        f.write('#$ -N DA_prepare_forecast_inputs\n')
        f.write(f'#$ -pe {hlm_config["parallel_argument"]} {hlm_config["num_parallel_slots"]}\n')
        f.write('#$ -j y\n')
        f.write('#$ -cwd\n')
        f.write(f'#$ -o {log_dir}/$JOB_ID.$SGE_TASK_ID.prepare_forecast.out\n') # Enable logging for debug
        f.write(f'#$ -e {log_dir}/$JOB_ID.$SGE_TASK_ID.prepare_forecast.err\n')
        f.write('#$ -l mf=2G\n') # Modest memory for file I/O
        f.write('#$ -q IFC\n')
        f.write('\n')
        f.write('module reset\n')
        f.write('\n')
        # The worker script needs the project root in its path to find modules like io_ifc
        f.write(f'export PYTHONPATH=$PYTHONPATH:{compute_root}\n')
        f.write(f'{python_executable} {worker_script_path} {tmp_dir}\n')
    return job_file_path

def run_hpc_ensemble_forecast(config: dict, ens_size: int) -> List[np.ndarray]:
    """
    Submits and manages a forecast step for an entire ensemble on HPC.
    Returns the final physical state vector q for each member.
    """
    tmp_dir = config['paths']['tmp_dir']
    # Correctly point to the subdirectory where forecast results (.rec) will be placed
    forecast_files_dir = os.path.join(tmp_dir, 'forecast_files')
    
    # 1. Create the HPC job script first, then submit it.
    job_script_path = create_hpc_job_script(config, tmp_dir)
    job_cmd = f"qsub -t 1-{ens_size} {job_script_path}"
    print(f"Submitting HPC forecast job: {job_cmd}")
    subprocess.run(job_cmd, shell=True, check=True)

    # 2. Wait for all .rec output files to be created.
    rec_paths = [os.path.join(forecast_files_dir, f"{j}.rec") for j in range(ens_size)] # This path is now correct
    _wait_for_files(rec_paths, timeout=3600, tmp_dir=tmp_dir, read_content=False) # Just wait for existence

    # 3. Process results: Parse each output .rec file to get the final state matrix.
    final_states = []
    for rec_path in rec_paths:
        try:
            state_matrix = parse_rec_file(rec_path)
            final_states.append(state_matrix)
        except Exception as e:
            print(f"Error parsing {rec_path}: {e}. Appending None.")
            final_states.append(None) # Handle potential file read errors

    # 4. Cleanup
    for j in range(ens_size):
        # Correctly clean up files from the subdirectory
        for suffix in [f'{j}.rec', f'{j}.gbl']:
             if os.path.isfile(p := os.path.join(forecast_files_dir, suffix)): os.remove(p)

    return final_states

def run_hpc_file_preparation(config: dict, n_ens: int, q_ensemble: np.ndarray, alpha_ensemble: np.ndarray, shared_data: dict):
    """
    Submits an HPC job array to generate all .prm and .rec files in parallel for the forecast step.
    """
    tmp_dir = config['paths']['tmp_dir']
    # Correctly point to the subdirectory for forecast-related files
    forecast_files_dir = os.path.join(tmp_dir, 'forecast_files')

    print(f"\n--- Starting HPC Task: Preparing inputs for {n_ens} members ---")

    # --- CRITICAL: Clean up artifacts from the previous forecast step ---
    print("[Forecast Prepare] Ensuring clean environment in tmp directory...")
    for k in range(n_ens):
        member_dir = os.path.join(forecast_files_dir, str(k)) # This path is correct
        if os.path.isdir(member_dir):
            shutil.rmtree(member_dir)
    
    # Clean up serialized data and old job scripts/logs
    files_to_clean = ['da_job_data.pkl', 'q_ensemble.npy', 'alpha_ensemble.npy']
    for filename in files_to_clean:
        try:
            # These files are now inside the forecast_files subdirectory
            os.remove(os.path.join(forecast_files_dir, filename))
        except FileNotFoundError:
            pass # It's okay if the file doesn't exist yet
    # Also clean up the job script from the root of tmp_dir
    try:
        os.remove(os.path.join(tmp_dir, 'submit_prepare_forecast_inputs.job'))
    except FileNotFoundError:
        pass
            
    # 1. Serialize shared data and member-specific data
    with open(os.path.join(forecast_files_dir, 'da_job_data.pkl'), 'wb') as f:
        pickle.dump(shared_data, f)
    np.save(os.path.join(forecast_files_dir, 'q_ensemble.npy'), q_ensemble)
    np.save(os.path.join(forecast_files_dir, 'alpha_ensemble.npy'), alpha_ensemble)

    # 2. Create and submit the job array script
    job_script_path = create_hpc_prepare_script(config, tmp_dir) # This path is correct
    job_cmd = f"qsub -t 1-{n_ens} {job_script_path}"
    print(f"Submitting HPC file preparation job: {job_cmd}")
    subprocess.run(job_cmd, shell=True, check=True)
    
    # 3. Wait for all files to be created. We check for one file per member.
    files_to_check = [os.path.join(forecast_files_dir, str(k), "params.prm") for k in range(n_ens)] # This path is correct
    try:
        _wait_for_files(files_to_check, timeout=600, tmp_dir=tmp_dir, read_content=False)
    except (TimeoutError, RuntimeError) as e:
        print(f"\n❌ Failed to generate all input files: {e}")
        sys.exit(1)
    print("Finished parallel generation of .prm and .rec files.")