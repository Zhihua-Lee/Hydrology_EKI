import os, sys
import numpy as np
import time
from typing import List, Tuple, Dict, Union
from io_ifc import create_presim_prm_from_template, create_presim_gbl, create_presim_job_file
from utils import get_ids, get_subwatershed

def _wait_for_files(file_paths: List[str], timeout: int = 1800) -> List[np.ndarray]:
    """
    Waits for a list of files to exist, be non-empty, and have consistent sizes.

    Args:
        file_paths (List[str]): A list of file paths to wait for.
        timeout (int): Timeout in seconds.

    Returns:
        List[np.ndarray]: A list of numpy arrays read from the files.
        
    Raises:
        TimeoutError: If the files are not ready within the timeout period.
        FileNotFoundError: If a file is not found after waiting.
    """
    start_time = time.time()
    num_files = len(file_paths)

    while True:
        elapsed = int(time.time() - start_time)
        if elapsed > timeout:
            raise TimeoutError(f"Timeout ({timeout}s) exceeded while waiting for files.")

        missing_indices = [i for i, p in enumerate(file_paths) if not os.path.isfile(p)]
        if missing_indices:
            msg = f"⏳ {elapsed}s: Waiting for {len(missing_indices)}/{num_files} files to appear..."
            sys.stdout.write("\r" + msg + " "*20)
            sys.stdout.flush()
            time.sleep(10)
            continue

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
        else: # This block executes if the for loop completes without a break
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

    # Define the directory for presimulation files
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

    # 1. Create the .prm file for the presimulation
    presim_prm_path = os.path.join(presim_dir, "presim.prm")
    template_prm_path = test_dict['prm'] # The original template
    create_presim_prm_from_template(template_prm_path, presim_prm_path, link_to_division_map, cr_ref_vec)

    # 2. Create the .gbl file for the presimulation
    presim_gbl_path = os.path.join(presim_dir, "presim.gbl")
    output_csv = test_dict['meas_series']
    create_presim_gbl(test_dict, presim_prm_path, presim_gbl_path, output_csv)
    
    # 3. Create the job submission file
    job_file_path = create_presim_job_file(test_dict, presim_dir, presim_gbl_path)

    if os.path.isfile(output_csv):
        print(f"Removing old simulation output file: {output_csv}")
        os.remove(output_csv)
        
    # 4. Submit the job
    job_cmd = f"qsub {job_file_path}"
    print(f"Submitting simulation job: {job_cmd}")
    os.system(job_cmd)
    
    try:
        _wait_for_files([output_csv])
        # The simulation output might contain header lines that need to be removed.
        with open(output_csv, 'r') as f:
            lines = f.readlines()
        if len(lines) > 2:
            lines_trimmed = lines[2:]
            with open(output_csv, 'w') as f:
                f.writelines(lines_trimmed)
            print(f"Removed header lines from {output_csv}")
        else:
            print(f"File has {len(lines)} lines. Assuming no header to remove.")
    except TimeoutError as e:
        print(f"\n❌ {e}")
        sys.exit(1)


def run_test(ens: int, X: np.ndarray, tmp_dir: str, idx_meas: np.ndarray) -> Tuple[np.ndarray]:
    """
    Run the test (ODE simulation) with a given ensemble size, latent parameter ensemble,
    temporary directory, and measurement indices.
    
    Args:
        ens (int): Number of ensemble members.
        X (np.ndarray): Latent parameter ensemble.
        tmp_dir (str): Temporary directory path (e.g., "tmp/5570910/").
        idx_meas (np.ndarray): Array containing measurement indices.

    Returns:
        Tuple[np.ndarray]: A tuple containing simulation results and statistics.
    """
    
    # 1. Submit the job array using the 'submit_job.job' script.
    job_cmd = f"qsub -t 1:{ens} {tmp_dir}submit_job.job"
    os.system(job_cmd)
    
    # 2. Wait for all result CSV files to be ready.
    csv_paths = [os.path.join(tmp_dir, f"{j}.csv") for j in range(ens)]
    try:
        read_values = _wait_for_files(csv_paths)
    except TimeoutError as e:
        print(f"\n❌ {e}")
        # Consider whether to exit or handle this more gracefully
        sys.exit(1)
            
    # 3. Process the read data after successful retrieval:
    # 3.1 Remove the last column (bug associated with written csv file, extra empty column)
    read_values_fixed = [res[:, :-1] for res in read_values]
    
    # 3.2 Extract measurement data based on specified measurement indices.
    read_values_measured = [res[:, idx_meas] for res in read_values_fixed]
    Y = np.concatenate([np.reshape(rm, (-1, 1)) for rm in read_values_measured], axis=1)
    
    # 3.3 Compute mean, standard deviation, and full list of results at plotting locations
    Y_plot_mean = np.mean(np.array(read_values_fixed), axis=0)
    Y_plot_std = np.std(np.array(read_values_fixed), axis=0)
    Y_plot = np.array(read_values_fixed)
    
    # 3.4 Compute the mean and standard deviation of the latent variables.
    X_plot_mean = np.mean(X, axis=1, keepdims=True)
    X_plot_std = np.std(X, axis=1, keepdims=True)
    
    # 3.5 Remove the temporary CSV files.
    for csv_path in csv_paths:
        if os.path.isfile(csv_path):
            os.remove(csv_path)
    
    return Y, Y_plot, Y_plot_mean, Y_plot_std, X_plot_mean, X_plot_std