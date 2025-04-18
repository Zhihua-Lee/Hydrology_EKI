import os
import numpy as np
import time
from typing import List, Tuple, Dict, Union


def run_test(ens: int, X: np.ndarray, tmp_dir: str, idx_meas: np.ndarray) -> Tuple[np.ndarray]:
    """
    Run the test (ODE simulation) with a given ensemble size, latent parameter ensemble,
    temporary directory, and measurement indices.
    
    Args:
        ens (int): Number of ensemble members.
        X (np.ndarray): Latent parameter ensemble.
        tmp_dir (str): Temporary directory path (e.g., "tmp/5570910/").
        idx_meas (np.ndarray): Array containing measurement indices.
    csv_missing_indices
    Returns:
        Tuple[np.ndarray]: A tuple containing simulation results and statistics.
    """
    
    # 1. Submit the job array using the 'submit_job.job' script.
    job_cmd = f"qsub -t 1:{ens} {tmp_dir}submit_job.job"
    procs = os.system(job_cmd)
    
    # 2. Attempt to read result CSV files; if missing, empty, or mismatched, wait and retry.
    while True:
        csv_missing_indices = []
        csv_zero_size_indices = []
        read_values = []
        
        try:
            # 2.1 Loop over each ensemble member and attempt to read its CSV file.
            for j in range(ens):
                csv_path = tmp_dir + str(j) + ".csv"
                if not os.path.isfile(csv_path):
                    csv_missing_indices.append(j)
                    continue
                
                data_j = np.genfromtxt(csv_path, delimiter=',', skip_header=2)
                if data_j.size == 0:
                    csv_zero_size_indices.append(j)
                read_values.append(data_j)
            
            # 2.2 If any files are missing or empty, print which ones and wait.
            if csv_missing_indices or csv_zero_size_indices:
                msg = (
                    f"⏳ Waiting for ensembles... "
                    f"Missing: {len(csv_missing_indices)} / {ens}, "
                    f"Empty: {len(csv_zero_size_indices)} / {ens}"
                )
                sys.stdout.write("\r" + msg + " " * 10)  # clear previous line tail
                sys.stdout.flush()
                time.sleep(10)
                continue
            
            # 2.3 Check that all files have the same size and are not all zero.
            count_all = [arr.size for arr in read_values]
            max_size = max(count_all)
            min_size = min(count_all)
            if (max_size - min_size) == 0:
                if max_size == 0:
                    sys.stdout.write("\r⏳ Waiting... All CSVs are zero-sized.           ")
                    sys.stdout.flush()
                    time.sleep(10)
                else:
                    # All files exist, are non-empty, and sizes are consistent.
                    print("\n✅ CSVs are ready.")
                    break
            else:
                msg = f"⏳ Waiting... File size mismatch. Sizes: {count_all}"
                sys.stdout.write("\r" + msg + " " * 10)
                sys.stdout.flush()
                time.sleep(10)
        
        except Exception as e:
            sys.stdout.write(f"\r❌ Error while reading CSV files: {e}. Retrying in 10s...     ")
            sys.stdout.flush()
            time.sleep(10)
            
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
    for j in range(ens):
        csv_path = tmp_dir + str(j) + ".csv"
        if os.path.isfile(csv_path):
            os.remove(csv_path)
    
    return Y, Y_plot, Y_plot_mean, Y_plot_std, X_plot_mean, X_plot_std



# def run_test(ens: int, X: np.ndarray, tmp_dir: str, idx_meas: np.ndarray) -> Tuple[np.ndarray]:
#     """
#     Run the test(ODE simulation) with given ensemble size, latent parameter ensemble, temporary directory, and measurement indices.

#     Args:
#         ens (int): Number of ensemble members.
#         X (np.ndarray): Latent parameter ensemble.
#         tmp_dir (str): Temporary directory path.
#         idx_meas (np.ndarray): Array containing measurement indices.

#     Returns:
#         Tuple[np.ndarray]: A tuple containing simulation results and statistics.
#     """
    
#     # Runs test utilizing 'submit_job.job' script, submiting array job
#     job = "qsub -t 1:" + str(ens) + ' ' + tmp_dir + 'submit_job.job'
#     procs = os.system(job) # running bash file array "job"

#     # Tries to read results files, retries every 100 seconds
#     while True:
#         try:
#             read_values = [np.genfromtxt(tmp_dir + str(j) + ".csv", delimiter=',', skip_header=2) for j in range(ens)] 
#             count_all = np.array([a.size for a in read_values])

#             #makes sure results are all the same size and not all 0
#             if (np.max(count_all) - np.min(count_all)) == 0:
#                 if np.any(np.max(count_all) == 0):
#                     pass
#                 else:
#                     break
#         except:
#             print("not finished, waiting 100 seconds")
#             time.sleep(100)
#             #TODO: probably should ensure this doesnt go forever if unmonitored

#     # Removes last column (bug associated with written csv file, extra empty column)
#     read_values_fixed = [results[:, :-1] for results in read_values]
    
#     # Gets results at measured locations
#     read_values_measured = [results[:, idx_meas] for results in read_values_fixed]
#     Y = np.concatenate([np.reshape(results, (-1, 1)) for results in read_values_measured], 1)
    
#     # Calculates mean, standard deviation, and full list of results at plotting locations
#     Y_plot_mean = np.mean(np.array(read_values_fixed), 0)
#     Y_plot_std = np.std(np.array(read_values_fixed), axis=0)
#     Y_plot = np.array(read_values_fixed)
    
#     # Calculates the mean and standard deviation of latent variables
#     X_plot_mean = np.mean(X, axis=1, keepdims=True)
#     X_plot_std = np.std(X, axis=1, keepdims=True)

#     # Remove temporary CSV files
#     for j in range(ens):
#         os.remove(tmp_dir + str(j) + ".csv")

#     return Y, Y_plot, Y_plot_mean, Y_plot_std, X_plot_mean, X_plot_std

