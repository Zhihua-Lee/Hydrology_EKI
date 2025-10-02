# File: DA/data_handler.py

from typing import Dict
from state_vector import StateVector
import numpy as np

class DataHandler:
    def __init__(self, config: dict, n_links: int, n_gauges: int, observation_timeseries: np.ndarray):
        print("Initializing DataHandler...")
        self.config = config
        self.n_links = n_links
        da_config = self.config.get('da_settings', {})
        self.n_gauges = n_gauges
        self.max_param_history = da_config.get('max_param_history', 10)
        self.analysis_states_history: Dict[int, np.ndarray] = {}
        self.full_observation_timeseries = observation_timeseries
        print("DataHandler initialized and observation data loaded.")

    def store_analysis_state(self, t: int, state_vector: StateVector):
        """Stores the physical component (discharge vector) of the analysis state."""
        self.analysis_states_history[t] = state_vector.q

    def get_historical_analysis_state(self, t: int) -> np.ndarray:
        """
        Retrieves a historical physical state and assembles it into a full (n_links, 5) matrix.
        """
        # Read the constant states from the centralized config to ensure consistency.
        constant_states = np.array(self.config['hlm_model']['constant_physical_states'])
        
        # With the V3 algorithm, the main loop starts at t=1, and the earliest state
        # requested will be for t=0. This check is now a safeguard for unexpected logic errors.
        if t < 0:
            raise ValueError(f"Error: Requested historical state for a negative time index t={t}, which is not allowed.")
        
        discharge_vector = self.analysis_states_history.get(t)
        if discharge_vector is None:
             # CRITICAL FIX: Do not silently fail by returning a zero matrix.
             # This indicates a critical failure in the DA logic (e.g., a state was not
             # stored correctly in a previous step). The program must stop.
             raise KeyError(f"Fatal: Historical analysis state for time step t={t} not found in DataHandler history.")
        
        # Assemble the full state matrix for the HLM simulation
        full_matrix = np.zeros((self.n_links, 5))
        full_matrix[:, 0] = discharge_vector
        full_matrix[:, 1:] = np.tile(constant_states, (self.n_links, 1))
        return full_matrix

    def get_observations_for_window(self, t_current: int) -> np.ndarray:
        """
        Extracts the window of real observations corresponding to the analysis window.
        This window grows dynamically with time, consistent with the formulation.
        
        Args:
            t_current (int): The current time step index.

        Returns:
            np.ndarray: A flattened 1D array of observations from t-N_t to t.
                        Shape: ((N_t + 1) * n_gauges,).
        """
        # V3 LOGIC FIX: The number of observation points must match the number of
        # simulation steps in the analysis window. A simulation with N_t parameters
        # produces N_t states to be compared against observations.
        N_t = min(t_current, self.max_param_history)
        start_idx = t_current - N_t + 1
        end_idx = t_current + 1

        if self.n_gauges == 0:
            return np.array([])

        # The total number of timesteps for a single gauge in the flattened array.
        # The timeseries is flattened Fortran-style, so data for each gauge is contiguous.
        n_timesteps_per_gauge = len(self.full_observation_timeseries) // self.n_gauges

        window_slices = []
        for i in range(self.n_gauges):
            # Calculate the starting offset for the current gauge's data block
            gauge_offset = i * n_timesteps_per_gauge
            # Extract the window for this specific gauge and append it
            gauge_window = self.full_observation_timeseries[gauge_offset + start_idx : gauge_offset + end_idx]
            window_slices.append(gauge_window)

        return np.concatenate(window_slices).flatten()