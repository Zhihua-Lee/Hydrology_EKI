# File: DA/analysis_window_worker.py

import os
import sys
import numpy as np
import pickle
import subprocess
from hlm_runner import HLMRunner
from analysis_operator import AnalysisOperator

def main():
    try:
        task_id = int(os.environ.get('SGE_TASK_ID', os.environ.get('SLURM_ARRAY_TASK_ID')))
        member_id = task_id - 1
    except (ValueError, TypeError):
        sys.exit("Error: Could not get a valid task ID.")

    if len(sys.argv) < 2:
        sys.exit("Error: Path to tmp_dir not provided.")
    tmp_dir = sys.argv[1]

    try:
        with open(os.path.join(tmp_dir, 'analysis_job_data.pkl'), 'rb') as f:
            job_data = pickle.load(f)
        
        config = job_data['config']
        analysis_jobs = job_data['analysis_jobs']
        job_info = analysis_jobs[member_id] # Get the specific job for this worker

    except Exception as e:
        sys.exit(f"Error loading data for member {member_id}: {e}")
    
    try:
        # Re-instantiate the HLMRunner on the worker node
        hlm_runner = HLMRunner(config)
        
        # Instantiate a dummy DataHandler, not used but needed for AnalysisOperator constructor
        class DummyDataHandler: pass
        analysis_operator = AnalysisOperator(config, DummyDataHandler(), hlm_runner)
        
        # --- Execute the window simulation ---
        # The HLMRunner's run_window_simulation contains the Python-level loop
        # that calls the robust, local mpirun execution for each step.
        simulated_window_states = hlm_runner.run_window_simulation(
            q_initial_matrix=job_info['q_initial_matrix'],
            param_sequence=job_info['param_sequence'],
            start_time=job_info['start_time']
        )
        
        # --- Extract observable part and save result ---
        # The result of run_window_simulation is a list of (n_links, 5) matrices
        Y_sim_window = np.array([analysis_operator.extract_observation_from_state(q_matrix) for q_matrix in simulated_window_states])
        # CRITICAL: Flatten in 'F' (column-major) order to match the structure of y_obs from DataHandler.
        Y_sim_vector = Y_sim_window.flatten('F')
        
        # Construct a unique filename for the analysis simulation result (Y_sim)
        results_dir = os.path.join(tmp_dir, 'analysis_results')
        t_current = job_info['t_current']
        if config.get('logging', {}).get('debug_mode', False):
            print(f"DEBUG worker {member_id} at t={t_current}: saving analysis simulation vector of length {len(Y_sim_vector)}")
        
        output_path = os.path.join(results_dir, f'analysis_sim_t{t_current:03d}_m{member_id:03d}.npy')
        np.save(output_path, Y_sim_vector)

    except Exception as e:
        error_msg = f"Error during analysis window simulation for member {member_id}: {e}"
        # --- START MODIFICATION ---
        if isinstance(e, subprocess.CalledProcessError):
            # 如果是子进程错误，记录详细的 stdout 和 stderr
            error_msg += f"\n--- STDOUT ---\n{e.stdout}"
            error_msg += f"\n--- STDERR ---\n{e.stderr}"
        # --- END MODIFICATION ---
        results_dir = os.path.join(tmp_dir, 'analysis_results')
        t_err_val = job_info.get("t_current", "unknown")
        with open(os.path.join(results_dir, f'analysis_t{t_err_val:03d}_m{member_id:03d}.error'), 'w') as f:
            f.write(error_msg)
        sys.exit(error_msg)

if __name__ == "__main__":
    main()