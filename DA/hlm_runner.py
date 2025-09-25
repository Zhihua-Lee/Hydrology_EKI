# File: DA/hlm_runner.py

import os
import shutil
import numpy as np
import pandas as pd
from typing import List

from scipy.sparse import coo_matrix
from io_ifc import get_ids, get_subwatershed, create_prm_from_division_params, _create_single_gbl, load_usgs_mapping, parse_rec_file, write_rec_file
# +++ NEW: Import the transformation function +++
from latent import transform_latent_to_physical

class HLMRunner:
    def __init__(self, config: dict):
        print("Initializing HLMRunner...")
        self.config = config
        self.hlm_config = config.get('hlm_model', {})
        self.paths_config = config.get('paths', {})
        self.tmp_dir = self.paths_config.get('tmp_dir')
        
        self.sorted_link_ids = get_ids(self.hlm_config)
        self.n_links = len(self.sorted_link_ids)
        
        try:
            prm_names = config['parameters']['prm_names']
            self.cr_param_index = prm_names.index('$Cr$')
        except (ValueError, KeyError):
            raise ValueError("'$Cr$' must be present in 'prm_names' in the config.")
        
        if self.hlm_config.get('watershed_csv'):
            self.division_to_link_map, self.link_to_division_map = get_subwatershed(self.hlm_config, self.sorted_link_ids)
        else:
            rows = np.zeros(self.n_links, dtype=int)
            cols = np.arange(self.n_links)
            self.division_to_link_map = coo_matrix((np.ones(self.n_links), (rows, cols)), shape=(1, self.n_links))
            self.link_to_division_map = {link_id: 0 for link_id in self.sorted_link_ids}
        self.n_divisions = self.division_to_link_map.shape[0]
        
        # The meas.sav file is now created externally before HLMRunner is initialized.
        self.meas_sav_path = os.path.join(self.tmp_dir, "meas.sav")
        print("HLMRunner initialized successfully.")

    def prepare_gbl_for_run(self, member_id: int, start_time: pd.Timestamp):
        """
        Prepares just the .gbl file for a single HLM run in forecast operator.
        Assumes that the large .prm and .rec files have already been created
        in a member-specific subdirectory by a parallel worker.
        """
        end_time = start_time + pd.Timedelta(hours=1)
        forecast_files_dir = os.path.join(self.tmp_dir, 'forecast_files')
        # The worker places its output in a subdirectory within the forecast_files directory
        run_dir = os.path.join(forecast_files_dir, str(member_id))
        
        gbl_config = self.hlm_config.copy()
        gbl_config.update({
            "time_start": start_time.strftime('%Y-%m-%d %H:%M'),
            "time_end": end_time.strftime('%Y-%m-%d %H:%M'),
        })
        # Manually add necessary root paths and model paths to the config dict being passed down.
        gbl_config['login_node_root'] = self.config.get('login_node_root')
        gbl_config['compute_node_root'] = self.config.get('compute_node_root')
        gbl_config.update(self.config.get('hlm_model', {})) # Ensure rvr, etc., are present
        
        _create_single_gbl(
            test_dict=gbl_config,
            # output_gbl_path=os.path.join(self.tmp_dir, f"{member_id}.gbl"),
            # # CRITICAL: For qsub jobs, we must also use absolute paths to eliminate
            # # any ambiguity related to the job's execution environment and CWD.
            # prm_file_path=gbl_config.get(['prm']),
            # input_rec_path=os.path.abspath(os.path.join(run_dir, "state.rec")),
            # output_rec_path=os.path.abspath(os.path.join(self.tmp_dir, f"{member_id}.rec")),
            # sav_file_path=os.path.abspath(self.meas_sav_path),
            # scratch_dir_path=os.path.join(self.hlm_config.get('scratch_dir'), str(member_id)),
            output_gbl_path=os.path.join(forecast_files_dir, f"{member_id}.gbl"),
            prm_file_path=os.path.join(run_dir, "params.prm"),
            input_rec_path=os.path.join(run_dir, "state.rec"),
            output_rec_path=os.path.join(forecast_files_dir, f"{member_id}.rec"),
            sav_file_path=self.meas_sav_path,
            scratch_dir_path=os.path.join(self.hlm_config.get('scratch_dir'), str(member_id)),
            target_env='login'
        )

    def _create_gbl_file(self, gbl_path, prm_path, input_rec_path, output_rec_path, start_time, end_time, member_id_str):
        """A private helper to create a single GBL file on HPC."""

        # Initialize the config by copying the hlm_model section, which contains 'rvr', 'evapo', etc.
        gbl_config = self.hlm_config.copy()
        
        # Add time-specific and other necessary top-level information.
        gbl_config.update({
            "time_start": start_time.strftime('%Y-%m-%d %H:%M'),
            "time_end": end_time.strftime('%Y-%m-%d %H:%M'),
            "model_num": self.hlm_config.get('model_num', 'DA_RUN'),
            # Pass the root paths required by the GBL creation utility
            'compute_node_root': self.config.get('compute_node_root'),
            'login_node_root': self.config.get('login_node_root')
        })
        

        _create_single_gbl(
            test_dict=gbl_config,
            output_gbl_path=gbl_path,
            # For local subprocess calls, we MUST use absolute paths.
            prm_file_path=os.path.abspath(prm_path),
            input_rec_path=os.path.abspath(input_rec_path),
            output_rec_path=os.path.abspath(output_rec_path),
            sav_file_path=os.path.abspath(self.meas_sav_path),
            # Scratch dir should also be member-specific
            scratch_dir_path=os.path.join(self.hlm_config.get('scratch_dir'), member_id_str),
            target_env='compute'
        )
    
    def run_window_simulation(self, q_initial_matrix: np.ndarray, param_sequence: List[float], start_time: pd.Timestamp) -> np.ndarray:
        """
        Runs the HLM model over a window of time with a sequence of parameters.
        This function is executed by the analysis_window_worker on a compute node.
        It performs a series of single-step simulations where each step depends
        on the output of the previous one.

        Args:
            q_initial_matrix (np.ndarray): The initial physical state matrix (n_links, 5) at the beginning of the window.
            param_sequence (List[float]): A list of parameters to use for each step.
            start_time (pd.Timestamp): The start time of the simulation window.

        Returns:
            np.ndarray: A (n_steps, n_links, 5) array of the simulated physical states over the window.
        """
        window_outputs = []
        current_q_matrix = q_initial_matrix
        current_time = start_time

        for i, alpha in enumerate(param_sequence):
            # This part runs locally ON THE COMPUTATION NODE inside the analysis_worker.
            # Each step uses the output of the previous step as its input.

            # --- DETAILED DEBUGGING ---
            # +++ FIX: Wrap debug prints in a conditional block and fix formatting +++
            if self.config.get('logging', {}).get('debug_mode', False):
                print(f"  [Worker] Window Sim Step {i}: Start Time={current_time}, Mean_Latent_Alpha={np.mean(alpha):.4f}")
                if np.any(np.isnan(current_q_matrix)):
                    print(f"  [Worker] WARNING: Input q_matrix for step {i} contains NaN values!")

            next_q_matrix = self._run_single_step_local(current_q_matrix, alpha, current_time)
            
            if next_q_matrix.size == 0 or np.any(np.isnan(next_q_matrix)):
                print(f"  [Worker] ERROR: Output q_matrix for step {i} is empty or contains NaN. Aborting window simulation.")
                # Return what we have so far to avoid a crash, even if it's incomplete
                return np.array(window_outputs)
            # --- END DEBUGGING ---

            window_outputs.append(next_q_matrix)
            # Update status and time to proceed to the next step.
            current_q_matrix = next_q_matrix
            current_time += pd.Timedelta(hours=1)
        return np.array(window_outputs)

    def _run_single_step_local(self, q_matrix: np.ndarray, alpha_param: float, start_time: pd.Timestamp) -> np.ndarray:
        """
        A local, synchronous version of run_single_step for use inside the analysis_worker.
        It now uses .rec files for both input and output.
        MODIFIED: It now accepts a LATENT alpha_param and transforms it to physical space.
        """
        # Create a unique temporary directory for this specific simulation step
        # to avoid race conditions. We use the process ID to guarantee uniqueness.
        pid = os.getpid()
        run_dir = os.path.join(self.tmp_dir, f"local_run_{pid}")
        os.makedirs(run_dir, exist_ok=True)
 
        # 1. Create step-specific input files
        prm_path = os.path.join(run_dir, "params.prm")
        
        # +++ NEW: Transform latent alpha to physical before creating PRM +++
        # `param_sequence` from `analysis_operator` gives a list of latent alpha vectors.
        # `alpha_param` here is one of those vectors, shape (n_divisions,).
        prm_dist_bool = [str(val).lower() == 'true' for val in self.config['parameters']["prm_dist"]]
        active_param_indices = [i for i, is_active in enumerate(prm_dist_bool) if is_active]
        
        # Reshape the latent vector into the 2D shape expected by the transformer
        # Shape: (n_divisions,) -> (1, n_divisions) for one active parameter
        latent_alpha_2d = alpha_param.reshape(1, -1)
        
        # Perform the transformation
        physical_alpha_2d = transform_latent_to_physical(
            self.config['parameters'],
            latent_alpha_2d,
            n_divisions=self.n_divisions,
            active_param_indices=active_param_indices
        )

        create_prm_from_division_params(self.hlm_config, self.link_to_division_map, physical_alpha_2d, active_param_indices, prm_path)
 
        input_rec_path = os.path.join(run_dir, "state.rec")
        write_rec_file(input_rec_path, self.hlm_config['model_num'], self.sorted_link_ids, q_matrix)

        # 2. Prepare GBL file to point to the new .rec files
        gbl_path = os.path.join(run_dir, "local.gbl")
        output_rec_path = os.path.join(run_dir, "output.rec")
        end_time = start_time + pd.Timedelta(hours=1)
        unique_scratch_id = f"local_worker_{pid}_step" # Make scratch unique per step too
        self._create_gbl_file(gbl_path, prm_path, input_rec_path, output_rec_path, start_time, end_time, unique_scratch_id)
        
        # 3. Execute locally using a robust bash wrapper
        # This runs on a compute node, so use compute_node_root
        compute_root = self.config.get('compute_node_root')
        executable_path = os.path.join(compute_root, 'exec/asynch/bin/asynch')
        num_slots = self.hlm_config.get('num_parallel_slots', 1)
        mpirun_executable = self.hlm_config.get('mpirun_path', 'mpirun')

        # --- Ultimate Robust Command Execution ---
        # We use 'bash -l -c "..."' to ensure a full login environment is sourced.
        # The '-l' flag is CRITICAL as it simulates a login shell, which correctly
        # initializes the 'module' command and all necessary environment variables
        # like LD_LIBRARY_PATH, required by the HLM executable.
        command_str = (
            f"module load openmpi && "
            # f"{executable_path} {os.path.abspath(gbl_path)}"
            f"{mpirun_executable} -np {num_slots} {executable_path} {os.path.abspath(gbl_path)}"
        )

        try:
            import subprocess
            # Note: We pass the command as a list ['bash', '-l', '-c', command_str]
            # to avoid any shell injection vulnerabilities and for clarity.
            # CRITICAL: We also set cwd=run_dir to handle any potential relative paths
            # inside the GBL that are relative to the GBL's own location.
            subprocess.run(['bash', '-l', '-c', command_str], check=True, cwd=run_dir, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"Local HLM model run failed for window simulation!\n"
                  f"Stderr: {e.stderr}\nStdout: {e.stdout}")
            raise

        # 4. Read .rec output and cleanup 
        output_matrix = parse_rec_file(output_rec_path)
        shutil.rmtree(run_dir, ignore_errors=True)
        return output_matrix

    def prepare_analysis_window_files(self, member_id: int, q_initial_matrix: np.ndarray, param_sequence: List[float], start_time: pd.Timestamp):
        """
        Prepares ALL files needed for a long-run window simulation for the Analysis step.
        This includes the initial .rec, all intermediate .prm files, and the final .gbl.
        This version uses a multi-step simulation within a single HLM run.
        """
        # HLM asynch does not support dynamically changing PRM files mid-run.
        # The only robust way is to run a sequence of single-step simulations.
        # This confirms the local loop is the necessary (though slow) implementation.
        # The logic below simulates this for a single member for file preparation.
        
        window_dir = os.path.join(self.tmp_dir, f"analysis_run_{member_id}")
        if os.path.exists(window_dir): shutil.rmtree(window_dir)
        os.makedirs(window_dir)

        current_q_matrix = q_initial_matrix
        current_time = start_time
        simulated_states = []

        for i, alpha in enumerate(param_sequence):
            step_dir = os.path.join(window_dir, f"step_{i}")
            os.makedirs(step_dir)
            
            # 1. Create step-specific input files
            prm_path = os.path.join(step_dir, "params.prm")
            cr_params = np.full((1, self.n_divisions), alpha)
            create_prm_from_division_params(self.hlm_config, self.link_to_division_map, cr_params, [self.cr_param_index], prm_path)
            
            input_rec_path = os.path.join(step_dir, "input.rec")
            write_rec_file(input_rec_path, self.hlm_config['model_num'], self.sorted_link_ids, current_q_matrix)
            
            output_rec_path = os.path.join(step_dir, "output.rec")
            gbl_path = os.path.join(step_dir, "run.gbl")
            
            # 2. Create the GBL for this single step
            gbl_config = self.hlm_config.copy()
            step_start_time = current_time
            step_end_time = current_time + pd.Timedelta(hours=1)
            gbl_config.update({
                "time_start": step_start_time.strftime('%Y-%m-%d %H:%M'),
                "time_end": step_end_time.strftime('%Y-%m-%d %H:%M'),
            })
            _create_single_gbl(
                test_dict=gbl_config, output_gbl_path=gbl_path,
                prm_file_path=prm_path, input_rec_path=input_rec_path,
                output_rec_path=output_rec_path, sav_file_path=self.meas_sav_path,
                scratch_dir_path=os.path.join(self.hlm_config.get('scratch_dir'), f"analysis_{member_id}_step_{i}")
            )
            
            # This function now only PREPARES files, it does not run anything.
            # The actual execution will be done by the analysis worker.
            
            # To get the next q_matrix for the next step's input, we'd need to run this.
            # This confirms that a simple, single HPC job per window is not feasible
            # without a model that supports dynamic parameter files.
            # We must stick to the Python-level loop on the worker.
            