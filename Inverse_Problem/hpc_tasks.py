import os, sys
import numpy as np
import time
import pickle
from typing import List, Tuple, Dict

# Import dependencies for the HPC task functions
from utils import get_ids, get_subwatershed
from io_ifc import create_presim_gbl, create_presim_job_file, create_prm_from_division_params, create_prm_generation_job_file


def _wait_for_files(file_paths: List[str], timeout: int = 1800, check_for_errors: bool = False, tmp_dir: str = None) -> List[np.ndarray]:
    """
    Waits for a list of files to exist, be non-empty, and have consistent sizes.
    Can also check for corresponding .error files.
    """
    start_time = time.time()
    num_files = len(file_paths)

    while True:
        elapsed = int(time.time() - start_time)
        if elapsed > timeout:
            raise TimeoutError(f"Timeout ({timeout}s) exceeded while waiting for files.")

        if check_for_errors and tmp_dir:
            # Check for errors from Python workers (e.g., PRM generation)
            worker_error_files = [f for f in os.listdir(tmp_dir) if f.endswith('.error')]
            if worker_error_files:
                error_msgs = []
                for err_file in worker_error_files:
                    with open(os.path.join(tmp_dir, err_file), 'r') as f:
                        error_msgs.append(f"Error in {err_file}: {f.read()}")
                raise RuntimeError("Errors detected in worker processes:\n" + "\n".join(error_msgs))

            # Check for errors from SGE jobs (non-empty .err files)
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

        # For PRM generation, we only need to check for existence, not content.
        if check_for_errors:
            print(f"\n✅ All {num_files} files are ready after {elapsed} seconds.")
            return None # Return None as we are not reading files here

        read_values = []
        empty_indices = []
        for i, p in enumerate(file_paths):
            try:
                data = np.genfromtxt(p, delimiter=',', skip_header=2)
                if data.size == 0:
                    empty_indices.append(i)
                read_values.append(data)
            except Exception as e:
                sys.stdout.write(f"\r❌ Error reading {p}: {e}. Retrying... {' '*20}")
                sys.stdout.flush()
                time.sleep(10)
                break 
        else:
            if empty_indices:
                msg = f"⏳ {elapsed}s: {len(empty_indices)}/{num_files} files are empty. Waiting..."
                sys.stdout.write("\r" + msg + " "*20)
                sys.stdout.flush()
                time.sleep(10)
                continue

            sizes = [arr.size for arr in read_values]
            if len(set(sizes)) > 1:
                msg = f"⏳ {elapsed}s: File size mismatch. Sizes: {sizes}. Waiting..."
                sys.stdout.write("\r" + msg + " "*20)
                sys.stdout.flush()
                time.sleep(10)
                continue
            
            print(f"\n✅ All {num_files} files are ready after {elapsed} seconds.")
            return read_values

def run_hpc_presimulation_for_synthetic_data(test_dict: Dict) -> None:
    """
    Submits a single HPC job to run a pre-simulation with reference parameters.
    This is used to generate the synthetic observation data for the EKI experiment.
    It creates all necessary files, submits the job, and waits for the output.
    """
    print("\n--- Starting HPC Task: Pre-Simulation for Synthetic Data Generation ---")

    presim_dir = os.path.join(os.path.dirname(test_dict['meas_series']), 'presim_run')
    os.makedirs(presim_dir, exist_ok=True)
    print(f"Created presimulation directory: {presim_dir}")

    prm_link_ids = get_ids(test_dict)
    sparse_parent, link_to_division_map = get_subwatershed(test_dict, prm_link_ids)
    num_divisions = sparse_parent.shape[0]

    cr_ref_config = test_dict.get('Cr_ref')
    if cr_ref_config is None:
        raise ValueError("Cr_ref must be defined in config for simulated data experiments.")
    
    if isinstance(cr_ref_config, (int, float)):
        cr_ref_vec = np.full(num_divisions, float(cr_ref_config))
    elif isinstance(cr_ref_config, list):
        if len(cr_ref_config) == num_divisions:
            cr_ref_vec = np.array(cr_ref_config)
        else:
            raise ValueError(f"Size of Cr_ref list ({len(cr_ref_config)}) does not match number of divisions ({num_divisions}).")
    else:
        raise TypeError("Cr_ref must be a number or a list.")

    try:
        prm_names = test_dict['prm_names']
        cr_param_index = prm_names.index('$Cr$')
    except (ValueError, KeyError):
        raise ValueError("'$Cr$' must be present in 'prm_names' for simulated data experiments.")

    physical_params_div_active = cr_ref_vec.reshape(1, num_divisions)
    active_param_indices = [cr_param_index]

    presim_prm_path = os.path.join(presim_dir, "presim.prm")
    create_prm_from_division_params(
        test_dict, link_to_division_map, physical_params_div_active, active_param_indices, presim_prm_path
    )

    presim_gbl_path = os.path.join(presim_dir, "presim.gbl")
    output_csv = test_dict['meas_series']
    create_presim_gbl(test_dict, presim_prm_path, presim_gbl_path, output_csv)
    
    job_file_path = create_presim_job_file(test_dict, presim_dir, presim_gbl_path)

    if os.path.isfile(output_csv):
        print(f"Removing old simulation output file: {output_csv}")
        os.remove(output_csv)
        
    job_cmd = f"qsub {job_file_path}"
    print(f"Submitting pre-simulation job: {job_cmd}")
    os.system(job_cmd)
    
    try:
        _wait_for_files([output_csv], timeout=1800)
        with open(output_csv, 'r') as f:
            lines = f.readlines()
        if len(lines) > 2:
            lines_trimmed = lines[2:]
            with open(output_csv, 'w') as f:
                f.writelines(lines_trimmed)
            print(f"Removed header lines from {output_csv}")
    except TimeoutError as e:
        print(f"\n❌ {e}")
        sys.exit(1)

def run_hpc_prm_generation_ensemble(test_dict: dict, X_ensemble: np.ndarray, ens: int, n_divisions: int, link_to_division_map: dict) -> None:
    """
    Submits an HPC job array to generate all .prm files for a given ensemble.
    It serializes shared data, submits the job, and waits for all files to be created.
    """
    tmp_dir = test_dict['tmp_dir']
    print(f"\n--- Starting HPC Task: Distributing {ens} .prm file generation tasks ---")
    
    for k in range(ens):
        prm_file_path = os.path.join(tmp_dir, f"{k}.prm")
        error_file_path = os.path.join(tmp_dir, f"{k}.prm.error")
        if os.path.exists(prm_file_path): os.remove(prm_file_path)
        if os.path.exists(error_file_path): os.remove(error_file_path)

    prm_dist_bool = [str(val).lower() == 'true' for val in test_dict["prm_dist"]]
    active_param_indices = [i for i, is_active in enumerate(prm_dist_bool) if is_active]

    shared_data = {
        'test_dict': test_dict, 'link_to_division_map': link_to_division_map,
        'n_divisions': n_divisions, 'active_param_indices': active_param_indices
    }
    with open(os.path.join(tmp_dir, 'prm_job_data.pkl'), 'wb') as f:
        pickle.dump(shared_data, f)
    np.save(os.path.join(tmp_dir, 'X_ensemble.npy'), X_ensemble)

    job_file_path = create_prm_generation_job_file(test_dict, ens)
    job_cmd = f"qsub {job_file_path}"
    os.system(job_cmd)

    prm_files_to_check = [os.path.join(tmp_dir, f"{k}.prm") for k in range(ens)]
    try:
        _wait_for_files(prm_files_to_check, timeout=600, check_for_errors=True, tmp_dir=tmp_dir)
    except (TimeoutError, RuntimeError) as e:
        print(f"\n❌ Failed to generate all .prm files: {e}")
        sys.exit(1)

    os.remove(os.path.join(tmp_dir, 'prm_job_data.pkl'))
    os.remove(os.path.join(tmp_dir, 'X_ensemble.npy'))
    os.remove(job_file_path)
    print("Finished generating .prm files.")

def run_hpc_simulation_ensemble(ens: int, X: np.ndarray, tmp_dir: str, idx_meas: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Submits an HPC job array to run the IFC model simulation for an entire ensemble.
    It waits for all simulation outputs (.csv) and then processes them.

    Args:
        ens (int): The number of ensemble members.
        X (np.ndarray): The parameter ensemble array. Used here for context but not direct calculation.
        tmp_dir (str): The temporary directory where job files and intermediate outputs are stored.
        idx_meas (np.ndarray): An array of column indices indicating which links correspond to measurement locations.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing two arrays:
        
        - Y (np.ndarray): The processed simulation output array, prepared for the data assimilation step (EnKF).
                          It contains only the data from the measurement locations (`idx_meas`) and is flattened.
                          Shape: `(n_timesteps * n_gauges, n_ensembles)`.
        - Y_plot (np.ndarray): The raw, complete simulation output from all ensemble members for all links specified
                               in the run's `meas.sav` file. This is used for plotting and saving the full particle state.
                               Shape: `(n_ensembles, n_timesteps, n_links)`.
    """
    job_cmd = f"qsub -t 1-{ens} {os.path.join(tmp_dir, 'submit_job.job')}"
    os.system(job_cmd)
    
    csv_paths = [os.path.join(tmp_dir, f"{j}.csv") for j in range(ens)]
    try:
        read_values = _wait_for_files(csv_paths, timeout=1800, check_for_errors=True, tmp_dir=tmp_dir)
    except (TimeoutError, RuntimeError) as e:
        print(f"\n❌ {e}")
        sys.exit(1)
            
    read_values_fixed = [res[:, :-1] for res in read_values]
    read_values_measured = [res[:, idx_meas] for res in read_values_fixed]
    Y = np.concatenate([np.reshape(rm, (-1, 1)) for rm in read_values_measured], axis=1)
    Y_plot = np.array(read_values_fixed)
    
    # Cleanup intermediate files from the simulation job array
    for j in range(ens):
        # SGE task IDs are 1-based, our ensemble/file IDs are 0-based
        sge_task_id = j + 1
        for suffix in [f'{j}.csv', f'{sge_task_id}.out', f'{sge_task_id}.err']:
            file_to_remove = os.path.join(tmp_dir, suffix)
            if os.path.isfile(file_to_remove):
                os.remove(file_to_remove)
    
    return Y, Y_plot