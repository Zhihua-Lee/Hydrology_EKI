# File: DA/analysis_operator.py

import numpy as np
import pandas as pd
from state_vector import StateVector
from typing import List
from hlm_runner import HLMRunner
from io_ifc import load_usgs_mapping, get_ids

class AnalysisOperator:
    def __init__(self, config: dict, data_handler, hlm_runner: HLMRunner):
        self.config = config
        da_config = config.get('da_settings', {})
        hlm_config = config.get('hlm_model', {})
        self.max_param_history = da_config.get('max_param_history', 10)
        self.data_handler = data_handler
        self.hlm_runner = hlm_runner
        
        obs_config = self.config.get('observations', {})
        sorted_link_ids = get_ids(hlm_config)
        usgs_to_link_id, _, _ = load_usgs_mapping(obs_config)
        assimilation_usgs_ids = obs_config.get('real_time_usgs_gauges', [])
        assimilation_link_ids = [usgs_to_link_id[uid] for uid in assimilation_usgs_ids]
        link_id_to_index = {link_id: i for i, link_id in enumerate(sorted_link_ids)}
        self.observation_indices = [link_id_to_index[lid] for lid in assimilation_link_ids]
        
        print(f"AnalysisOperator will extract observations from state vector indices: {self.observation_indices}")
        print("AnalysisOperator initialized.")

    def prepare_analysis_simulation_job(self, forecast_vec: StateVector, t_current: int) -> dict:
        """
        Prepares the job description for a single member's analysis window simulation.
        This function conceptually defines the operator H(X_f), which maps a forecast
        state X_f to a simulated observation window Y_sim.

        Args:
            forecast_vec (StateVector): The forecast state vector (X_f) for one member.
            t_current (int): The current time step index.

        Returns:
            dict: A dictionary containing all necessary info for the HPC worker.
        """
        # V3 LOGIC:
        # N_t is the number of steps in the re-run window.
        # At t=1, N_t=1. At t=2, N_t=2, etc., up to the max history length.
        N_t = min(t_current, self.max_param_history)
        # The re-run simulation starts from the physical state at t_current - N_t.
        start_q_time_index = t_current - N_t
        initial_q_for_sim = self.data_handler.get_historical_analysis_state(start_q_time_index)
        # The parameter sequence has length N_t. It's the N_t most recent parameters.
        param_sequence = forecast_vec.alpha_r_history[:N_t][::-1]

        # Conditionally print debug info based on config flag
        if self.config.get('logging', {}).get('debug_mode', False):
            print(f"DEBUG (t={t_current}): N_t={N_t}, start_q_t={start_q_time_index}, history_len={len(forecast_vec.alpha_r_history)}, param_sequence_len={len(param_sequence)}")

        start_time_window = pd.to_datetime(self.config['da_settings']['assimilation_window']['start']) + pd.Timedelta(hours=start_q_time_index)
        
        # This dictionary contains all info a worker needs for one window simulation
        job_info = {
            'q_initial_matrix': initial_q_for_sim,
            'param_sequence': param_sequence,
            'start_time': start_time_window,
            't_current': t_current  # Pass current time step index to the worker
        }
        return job_info

    def extract_observation_from_state(self, q_matrix: np.ndarray) -> np.ndarray:
        """Extracts discharge at observation locations from a full state matrix."""
        # q_matrix is (n_links, 5), we need discharge (col 0) at specific links
        return q_matrix[self.observation_indices, 0]