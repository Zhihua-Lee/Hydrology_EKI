import numpy as np
import pandas as pd
from state_vector import StateVector
from typing import List
from hlm_runner import HLMRunner
import hpc_da_tasks # Import the new HPC task coordinator
# We will need the model runner interface, which can be adapted from io_ifc.py
# from io_ifc import HLMRunner 

class ForecastOperator:
    """
    Handles the forecast step of the data assimilation process.

    This operator implements the state propagation model M(.), which evolves the
    augmented state vector from the analysis at time t to the forecast at time t+1.
    It runs the physical model for one step and evolves the parameter state history.
    """
    def __init__(self, config: dict, hlm_runner: HLMRunner):
        """
        Initializes the forecast operator using a configuration dictionary.

        Args:
            config (dict): The main configuration dictionary for the DA run.
            hlm_runner (HLMRunner): An initialized HLMRunner instance for model interaction.
        """
        self.config = config
        da_config = config.get('da_settings', {})
        hlm_config = config.get('hlm_model', {})

        self.max_param_history = da_config.get('max_param_history', 10)
        self.alpha_noise_std = da_config.get('alpha_process_noise_std', 0.01)
        self.q_noise_std = da_config.get('physical_state_process_noise_std', 0.0)
        
        # The HLM model runner is passed in for actual model execution.
        self.hlm_runner = hlm_runner
        print("ForecastOperator initialized.")

    def run_forecast(self, analysis_ensemble: List[StateVector], t: int) -> List[StateVector]:
        """
        Performs the forecast step for an entire ensemble of state vectors.
        This corresponds to the equation: X_{t+1|t} = M(X_{t|t}) + w_t.

        Args:
            analysis_ensemble (List[StateVector]): The ensemble of analysis state
                                                   vectors from time t (X_{t|t}).
            t (int): The current time step.

        Returns:
            List[StateVector]: The ensemble of forecast state vectors for time t+1 (X_{t+1|t}).
        """
        print(f"\n--- Starting Forecast Step for Time {t} ---")
        n_ens = len(analysis_ensemble)

        # --- STAGE 1: Gather data for all ensemble members ---
        # Instead of writing files serially, we first collect all necessary data in memory.
        q_ensemble_list = []
        alpha_ensemble_list = []
        start_time = pd.to_datetime(self.config['da_settings']['assimilation_window']['start']) + pd.Timedelta(hours=t)

        for state_vec_analysis in analysis_ensemble:
            # state_vec_analysis.get_current_parameter() returns the latest alpha history entry.
            # Shape is (n_divisions,)
            current_alpha_vector = state_vec_analysis.get_current_parameter()
            n_divisions = current_alpha_vector.shape[0]

            # Evolve parameter state via random walk: alpha_{r,t+1} = alpha_{r,t} + w_t
            noise = np.random.normal(0, self.alpha_noise_std, size=n_divisions)
            next_alpha_vector = current_alpha_vector + noise
            
            # The StateVector stores the discharge vector, which we pass to the worker.
            q_ensemble_list.append(state_vec_analysis.q)
            alpha_ensemble_list.append(next_alpha_vector)
        
        # Convert lists to numpy arrays for efficient serialization
        q_ensemble = np.array(q_ensemble_list)
        alpha_ensemble = np.array(alpha_ensemble_list)

        # --- STAGE 2: Execute parallel file preparation on HPC ---
        # Read the constant states from the centralized config to pass to the workers.
        constant_states = np.array(self.config['hlm_model']['constant_physical_states'])
        shared_data_for_workers = {
            'config': self.config,
            'link_to_division_map': self.hlm_runner.link_to_division_map,
            'n_divisions': self.hlm_runner.n_divisions,
            'cr_param_index': self.hlm_runner.cr_param_index,
            'sorted_link_ids': self.hlm_runner.sorted_link_ids,
            'constant_states': constant_states,
        }
        hpc_da_tasks.run_hpc_file_preparation(self.config, n_ens, q_ensemble, alpha_ensemble, shared_data_for_workers)

        # --- STAGE 3: Prepare .gbl files (this is fast and can remain serial) ---
        print("[Forecast] Preparing .gbl files for all members...")
        for i in range(n_ens):
             # Only prepare the small .gbl file now. The .prm and .uini are already on disk.
             self.hlm_runner.prepare_gbl_for_run(member_id=i, start_time=start_time)

        # --- STAGE 4: Execute the entire HLM simulation ensemble on HPC ---
        print(f"[Forecast] Submitting ensemble of {n_ens} members to HPC...")
        final_q_states = hpc_da_tasks.run_hpc_ensemble_forecast(self.config, n_ens)

        # --- STAGE 5: Collect results and assemble the forecast state vectors ---
        print("[Forecast] Assembling forecast state vectors from HPC results...")
        forecast_ensemble = []
        for i, state_vec_analysis in enumerate(analysis_ensemble):
            # HLM returns many state variables, we only need the first one (discharge) for the StateVector.
            # --- Defensive Check ---
            # Check if the returned state from HPC is a valid 2D matrix.
            # A 1D array indicates a failed HLM run that produced an empty/invalid .rec file.
            state_matrix = final_q_states[i]
            if state_matrix.ndim != 2 or state_matrix.shape[1] != 5:
                raise RuntimeError(f"Forecast for ensemble member {i} at time {t} failed. "
                                 f"The HLM model run likely produced an invalid .rec file, possibly due to invalid initial conditions (e.g., NaN/inf). "
                                 f"Received shape: {state_matrix.shape}")
            
            next_q_discharge = state_matrix[:, 0]
            # Add process noise to the physical state (multiplicative noise) to represent model error
            next_q_discharge = np.maximum(0, next_q_discharge * np.random.normal(1.0, self.q_noise_std, size=next_q_discharge.shape))

            next_alpha_vector = alpha_ensemble[i] # Get the corresponding alpha vector
            
            # --- 2. Update and assemble the new parameter history ---
            new_alpha_history = np.insert(state_vec_analysis.alpha_r_history, 0, next_alpha_vector, axis=0)
            
            # Trim the history to maintain the maximum window size (N_max).
            # We need to store one extra history point because the analysis window simulation
            # for a window of size N_max requires N_max+1 parameters.
            new_alpha_history = new_alpha_history[:self.max_param_history + 1]
            
            forecast_vec = StateVector(next_q_discharge, new_alpha_history)
            forecast_ensemble.append(forecast_vec)
            
        return forecast_ensemble