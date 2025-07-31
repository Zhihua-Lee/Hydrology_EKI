import os, sys
import numpy as np
import time
import pickle
from typing import List, Tuple, Dict, Union
from io_ifc import create_presim_prm_from_template, create_presim_gbl, create_presim_job_file, create_prm_from_division_params
from utils import get_ids, get_subwatershed
from latent import transform_latent_to_physical

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
            error_files = [f for f in os.listdir(tmp_dir) if f.endswith('.error')]
            if error_files:
                error_msgs = []
                for err_file in error_files:
                    with open(os.path.join(tmp_dir, err_file), 'r') as f:
                        error_msgs.append(f"Error in {err_file}: {f.read()}")
                raise RuntimeError("Errors detected in worker processes:\n" + "\n".join(error_msgs))

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

def generate_synthetic_data(test_dict: Dict) -> None:
    """
    Generates synthetic hydrograph data to be used as the 'true' observation.
    This involves running a pre-simulation with known reference parameters by dynamically
    creating all necessary configuration and job files.
    """
    print("\n--- Starting Pre-Simulation for Synthetic Data Generation ---")

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

    presim_prm_path = os.path.join(presim_dir, "presim.prm")
    create_presim_prm_from_template(test_dict['prm'], presim_prm_path, link_to_division_map, cr_ref_vec)

    presim_gbl_path = os.path.join(presim_dir, "presim.gbl")
    output_csv = test_dict['meas_series']
    create_presim_gbl(test_dict, presim_prm_path, presim_gbl_path, output_csv)
    
    job_file_path = create_presim_job_file(test_dict, presim_dir, presim_gbl_path)

    if os.path.isfile(output_csv):
        print(f"Removing old simulation output file: {output_csv}")
        os.remove(output_csv)
        
    job_cmd = f"qsub {job_file_path}"
    print(f"Submitting simulation job: {job_cmd}")
    os.system(job_cmd)
    
    try:
        _wait_for_files([output_csv])
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

def run_test(ens: int, X: np.ndarray, tmp_dir: str, idx_meas: np.ndarray) -> Tuple[np.ndarray]:
    """
    Run the test (ODE simulation) with a given ensemble size, latent parameter ensemble,
    temporary directory, and measurement indices.
    """
    job_cmd = f"qsub -t 1-{ens} {os.path.join(tmp_dir, 'submit_job.job')}"
    os.system(job_cmd)
    
    csv_paths = [os.path.join(tmp_dir, f"{j}.csv") for j in range(ens)]
    try:
        read_values = _wait_for_files(csv_paths)
    except (TimeoutError, RuntimeError) as e:
        print(f"\n❌ {e}")
        sys.exit(1)
            
    read_values_fixed = [res[:, :-1] for res in read_values]
    read_values_measured = [res[:, idx_meas] for res in read_values_fixed]
    Y = np.concatenate([np.reshape(rm, (-1, 1)) for rm in read_values_measured], axis=1)
    
    Y_plot_mean = np.mean(np.array(read_values_fixed), axis=0)
    Y_plot_std = np.std(np.array(read_values_fixed), axis=0)
    Y_plot = np.array(read_values_fixed)
    
    X_plot_mean = np.mean(X, axis=1, keepdims=True)
    X_plot_std = np.std(X, axis=1, keepdims=True)
    
    for csv_path in csv_paths:
        if os.path.isfile(csv_path):
            os.remove(csv_path)
    
    return Y, Y_plot, Y_plot_mean, Y_plot_std, X_plot_mean, X_plot_std

def _create_prm_generation_job_file(test_dict: dict, ens: int) -> str:
    """
    Creates the HPC batch job script for generating .prm files in parallel.
    """
    tmp_dir = test_dict['tmp_dir']
    job_file_path = os.path.join(tmp_dir, 'submit_prm_job.job')
    worker_script_path = os.path.join(test_dict['project_root'], 'Inverse_Problem', 'generate_prm_worker.py')

    with open(job_file_path, 'w') as f:
        f.write('#!/bin/bash\n')
        f.write('#$ -N prm_generation\n')
        f.write('#$ -j y\n')
        f.write('#$ -cwd\n')
        f.write(f'#$ -t 1-{ens}\n')
        f.write('#$ -l mf=2G\n') # Request modest memory for this simple task
        f.write('#$ -q IFC\n')
        # f.write(f'#$ -o {tmp_dir}$TASK_ID.out\n')
        # f.write(f'#$ -e {tmp_dir}$TASK_ID.err\n')
        f.write('#$ -o /dev/null\n')
        f.write('#$ -e /dev/null\n')
        f.write('\n')
        f.write('module reset\n')
        # f.write('module load python\n') # Ensure python environment is loaded
        f.write('\n')
        # Pass the temporary directory as an argument to the worker script
        python_executable = test_dict['hpc_python_path']
        f.write(f'{python_executable} {worker_script_path} {tmp_dir}\n')
    return job_file_path

def generate_prm_files_for_ensemble(
    test_dict: dict,
    X_ensemble: np.ndarray,
    ens: int,
    n_divisions: int,
    link_to_division_map: dict
) -> None:
    """
    Generates all .prm files for a given ensemble by submitting an HPC job array.
    
    This function first cleans up any old .prm files from the temporary directory
    to prevent race conditions where the file-waiting logic sees old files and
    returns prematurely. It then serializes shared data, submits a job array
    to the HPC scheduler, and waits for all new .prm files to be created.

    Args:
        X_ensemble (np.ndarray): The complete latent parameter ensemble.
                                 Shape: (n_active_params, n_divisions, n_ens).
    """
    tmp_dir = test_dict['tmp_dir']
    print(f"Distributing {ens} .prm file generation tasks to HPC cluster...")
    
    # 1. Pre-cleanup Step to prevent race conditions.
    # Safely remove any old .prm or .prm.error files from a previous iteration
    # to ensure the _wait_for_files logic must wait for new file creation.
    print("Cleaning up old .prm files before generation...")
    for k in range(ens):
        prm_file_path = os.path.join(tmp_dir, f"{k}.prm")
        error_file_path = os.path.join(tmp_dir, f"{k}.prm.error")
        try:
            os.remove(prm_file_path)
        except FileNotFoundError:
            pass  # File didn't exist, which is fine.
        try:
            os.remove(error_file_path)
        except FileNotFoundError:
            pass # Error file didn't exist, which is fine.

    # 2. Serialize shared data for worker processes
    shared_data = {
        'test_dict': test_dict,
        'link_to_division_map': link_to_division_map,
        'n_divisions': n_divisions
    }
    with open(os.path.join(tmp_dir, 'prm_job_data.pkl'), 'wb') as f:
        pickle.dump(shared_data, f)
    
    np.save(os.path.join(tmp_dir, 'X_ensemble.npy'), X_ensemble)

    # 3. Create the job submission script
    job_file_path = _create_prm_generation_job_file(test_dict, ens)

    # 4. Submit the job array
    job_cmd = f"qsub {job_file_path}"
    print(f"Submitting PRM generation job: {job_cmd}")
    os.system(job_cmd)

    # 5. Wait for all .prm files to be created by the job array
    prm_files_to_check = [os.path.join(tmp_dir, f"{k}.prm") for k in range(ens)]
    try:
        _wait_for_files(prm_files_to_check, timeout=600, check_for_errors=True, tmp_dir=tmp_dir)
    except (TimeoutError, RuntimeError) as e:
        print(f"\n❌ Failed to generate all .prm files: {e}")
        sys.exit(1)

    # 6. Clean up temporary data files
    os.remove(os.path.join(tmp_dir, 'prm_job_data.pkl'))
    os.remove(os.path.join(tmp_dir, 'X_ensemble.npy'))
    os.remove(job_file_path)

    print("Finished generating .prm files.")