# File: DA/kalman_update.py

import numpy as np
from typing import List
from state_vector import StateVector
from analysis_operator import AnalysisOperator
import os
import pickle
import subprocess
from hpc_da_tasks import _wait_for_files

class KalmanUpdate:
    def __init__(self, config: dict):
        self.config = config
        print("KalmanUpdate initialized.")
    
    def _create_hpc_analysis_script(self, config: dict, tmp_dir: str):
        hlm_config = config.get('hlm_model', {})
        """Creates the batch job file for the parallel analysis step."""
        compute_root = config['compute_node_root']
        job_file_path = os.path.join(tmp_dir, 'submit_analysis.job')
        log_dir = os.path.join(tmp_dir, 'hpc_logs')
        worker_script_path = os.path.join(compute_root, 'DA', 'analysis_window_worker.py')
        python_executable = config['hpc_python_path']

        with open(job_file_path, 'w') as f:
            f.write('#!/bin/bash\n')
            f.write('#$ -N DA_analysis\n')
            f.write('#$ -j y\n')
            f.write('#$ -cwd\n')
            # --- START MODIFICATION ---
            # 从配置中添加并行环境请求
            # 注意: analysis step 里的 hlm_runner 每次只跑一个 member，但这个 member 内部可能需要并行
            num_slots = hlm_config.get('num_parallel_slots', 1)
            pe_arg = hlm_config.get('parallel_argument', 'smp')
            if num_slots > 1:
                f.write(f'#$ -pe {pe_arg} {num_slots}\n')
            # --- END MODIFICATION ---
            # --- START MODIFICATION ---
            # 注释掉 /dev/null 的重定向
            # f.write('#$ -o /dev/null\n')
            # f.write('#$ -e /dev/null\n')

            # 将输出重定向到 tmp_dir 中，并用任务ID命名文件
            # $SGE_TASK_ID 是SGE在作业运行时会自动设置的环境变量
            f.write(f'#$ -o {log_dir}/$JOB_ID.$SGE_TASK_ID.analysis.out\n')
            f.write(f'#$ -e {log_dir}/$JOB_ID.$SGE_TASK_ID.analysis.err\n')
            # --- END MODIFICATION ---
            f.write('#$ -l mf=4G\n')
            f.write('#$ -q IFC\n')
            f.write('\n')
            f.write('module reset\n')
            f.write('module load openmpi\n')
            f.write(f'export PYTHONPATH=$PYTHONPATH:{compute_root}\n')
            f.write(f'{python_executable} {worker_script_path} {tmp_dir}\n')
        return job_file_path

    def _run_hpc_analysis_simulations(self, config: dict, analysis_jobs: List[dict], t_current: int) -> np.ndarray:
        """Submits and manages the HPC job array for the analysis step."""
        tmp_dir = config['paths']['tmp_dir'] 
        n_ens = len(analysis_jobs)

        # This is the data that analysis_window_worker.py expects
        job_data = {'config': config, 'analysis_jobs': analysis_jobs}
        with open(os.path.join(tmp_dir, 'analysis_job_data.pkl'), 'wb') as f:
            pickle.dump(job_data, f)

        job_script_path = self._create_hpc_analysis_script(config, tmp_dir)
        job_cmd = f"qsub -t 1-{n_ens} {job_script_path}"
        subprocess.run(job_cmd, shell=True, check=True)

        # Build filenames pointing to the structured results subdirectory
        results_dir = os.path.join(tmp_dir, 'analysis_results')
        result_files = [os.path.join(results_dir, f'analysis_sim_t{t_current:03d}_m{i:03d}.npy') for i in range(n_ens)]
        _wait_for_files(result_files, timeout=3600, tmp_dir=tmp_dir, read_content=False)

        # Clean up worker error files if any were created
        for i in range(n_ens):
            error_file = os.path.join(results_dir, f'analysis_t{t_current:03d}_m{i:03d}.error')
            if os.path.exists(error_file): os.remove(error_file)

        # Load all results and stack them into the Y_sim matrix
        Y_sim_list = [np.load(f) for f in result_files]
        Y_sim_matrix = np.array(Y_sim_list).T
        return Y_sim_matrix

    def _update_standard(self, X_f, Y_f, y_obs, R):
        ensemble_size = X_f.shape[1]
        obs_size = len(y_obs)
        x_mean = np.mean(X_f, axis=1, keepdims=True)
        y_mean = np.mean(Y_f, axis=1, keepdims=True)
        pert_vec = np.random.normal(0, 1, (obs_size, ensemble_size))
        # y_pert = y_obs + np.sqrt(R) @ pert_vec
        # --- START OF MODIFICATION ---
        # Get the diagonal of R as a 1D vector for robust broadcasting
        R_diag = np.diag(R)
        # Generate perturbed observations using element-wise multiplication (broadcasting)
        # This is equivalent to `L @ pert_vec` for a diagonal R, but more robust.
        y_pert = y_obs + np.sqrt(R_diag)[:, np.newaxis] * pert_vec
        # --- END OF MODIFICATION ---
        X_prime = (X_f - x_mean) / np.sqrt(ensemble_size - 1)
        Y_prime = (Y_f - y_mean) / np.sqrt(ensemble_size - 1)
        K = np.linalg.solve((Y_prime @ Y_prime.T + R).T, (X_prime @ Y_prime.T).T).T
        X_a = X_f + K @ (y_pert - Y_f)
        return X_a

    def _update_svd(self, X_f, Y_f, y_obs, R_diag):
        ensemble_size = X_f.shape[1]
        obs_size = Y_f.shape[0]
        x_mean = np.mean(X_f, axis=1, keepdims=True)
        y_mean = np.mean(Y_f, axis=1, keepdims=True)
        X_prime = X_f - x_mean
        Y_prime = Y_f - y_mean
        pert_vec = np.random.normal(0, 1, (obs_size, ensemble_size))
        y_pert = y_obs + np.sqrt(R_diag)[:, np.newaxis] * pert_vec
        d = y_pert - Y_f

        # --- ROBUSTNESS FIX for Broadcasting ---
        # The original `(Y_prime.T / R_diag)` is fragile when obs_dim is 1.
        # We rewrite the operation Y'ᵀR⁻¹Y' as Y'ᵀ @ (R⁻¹Y'), which can be implemented
        # with robust row-wise broadcasting: Y_prime.T @ (Y_prime / R_diag[:, np.newaxis])
        M = Y_prime.T @ (Y_prime / R_diag[:, np.newaxis]) + (ensemble_size - 1) * np.eye(ensemble_size)
        
        # Apply the same robust broadcasting to the Y'ᵀR⁻¹d term.
        update_ens_space = np.linalg.solve(M, Y_prime.T @ (d / R_diag[:, np.newaxis]))
        update = X_prime @ update_ens_space
        X_a = X_f + update
        return X_a

    def run_update_step(self, forecast_ensemble: List[StateVector], y_obs_window: np.ndarray,
                        analysis_operator: AnalysisOperator, t_current: int) -> np.ndarray:
        
        # 1. Prepare job descriptions for all ensemble members
        print("[Analysis] Preparing job descriptions for parallel window simulations...")
        analysis_jobs = [analysis_operator.prepare_analysis_simulation_job(vec, t_current) for vec in forecast_ensemble]
        
        # 2. Run all window simulations in parallel on HPC to get the simulated observation ensemble (Y_sim)
        Y_sim = self._run_hpc_analysis_simulations(self.config, analysis_jobs, t_current)
        
        # 3. Assemble state matrix (X_f) and observation vector (y_obs)
        X_f = np.array([vec.full_vector for vec in forecast_ensemble]).T
        y_obs = y_obs_window.reshape(-1, 1)

        # Conditionally print debug info based on config flag
        if self.config.get('logging', {}).get('debug_mode', False):
            print(f"\n--- DEBUG (t={t_current}) ---")
            print(f"Shape of Y_sim (simulated obs): {Y_sim.shape}")
            print(f"Shape of y_obs (real obs): {y_obs.shape}")
            print(f"Shape of X_f (state vector): {X_f.shape}")
        
        # --- MODIFIED: Correctly construct the observation error covariance matrix R ---
        da_config = self.config.get('da_settings', {})
        real_time_var = da_config.get('real_time_obs_error_var', 1.0)
        smoother_var = da_config.get('smoother_obs_error_var', 15.0)
        
        # Get the number of gauges from the analysis operator's configured indices.
        n_gauges = len(analysis_operator.observation_indices)
        obs_dim = Y_sim.shape[0]

        # Initialize all variances to the smoother (historical) variance.
        r_variances = np.full(obs_dim, smoother_var)

        # If we have gauges and observations, find the real-time indices and update their variance.
        if n_gauges > 0 and obs_dim > 0:
            # The number of time steps in the assimilation window (e.g., N_t + 1)
            n_steps_in_window = obs_dim // n_gauges
            # For each gauge, its real-time observation is the last one in its data block.
            for i in range(n_gauges):
                real_time_idx = (i * n_steps_in_window) + (n_steps_in_window - 1)
                if real_time_idx < obs_dim:
                    r_variances[real_time_idx] = real_time_var
        
        R = np.diag(r_variances)
        R_diag = r_variances

        ensemble_size = X_f.shape[1]
        
        # --- MODIFICATION: Always use the more numerically stable SVD-based update ---
        # This avoids inverting a potentially large (obs_dim x obs_dim) matrix,
        # which can become ill-conditioned even when obs_dim is smaller than ensemble_size.
        # Instead, it inverts a smaller (ensemble_size x ensemble_size) matrix.
        print(f"Using numerically stable SVD-based update (y_dim={obs_dim}, ens_size={ensemble_size}).")
        X_a = self._update_svd(X_f, Y_sim, y_obs, R_diag)
        
        print("Update step complete.")
        return X_a.T