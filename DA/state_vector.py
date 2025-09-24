import numpy as np
from typing import List

class StateVector:
    """

    Represents the augmented state vector for the DA system.

    This class manages the physical state (e.g., streamflow) and the
    history of the rainfall correction parameter over a moving window.
    """
    def __init__(self, physical_state: np.ndarray, param_history: np.ndarray):
        """
        Initializes the augmented state vector.

        Args:
            physical_state (np.ndarray): The vector of the model's physical
                                         states (e.g., streamflow at all links),
                                         representing `q_t`.
            param_history (np.ndarray): A vector containing the history of
                                        the parameter (e.g., alpha_r).
                                        IMPORTANT: The history is ordered from most
                                        recent to oldest, i.e., `[alpha_r,t, alpha_r,t-1, ...]`.
        """
        self.q = physical_state
        self.alpha_r_history = param_history

    def get_current_parameter(self) -> float:
        """
        Returns the most recent parameter value, alpha_{r,t}.

        This is the parameter value used to drive the model forward for the next
        time step in the forecast operator.
        """
        # The current parameter is the first element in the history
        return self.alpha_r_history[0] if len(self.alpha_r_history) > 0 else None

    @property
    def full_vector(self) -> np.ndarray:
        """Returns the complete augmented state vector as a single numpy array."""
        return np.concatenate([self.q, self.alpha_r_history])

    def __repr__(self) -> str:
        return (f"StateVector(q_shape={self.q.shape}, "
                f"alpha_history_len={len(self.alpha_r_history)})")

    @staticmethod
    def reconstruct_ensemble_from_matrix(
        analysis_matrix: np.ndarray, 
        n_physical_states: int, 
            n_param_history: int
    ) -> List['StateVector']:
        """
        Reconstructs an ensemble (list of StateVector objects) from a raw analysis matrix.

        This is a crucial bridge between the mathematical Kalman update step, which
        operates on a plain NumPy array, and the object-oriented DA framework.
        It converts the updated matrix back into a list of StateVector objects
        that can be passed to the next forecast step.

        Args:
            analysis_matrix (np.ndarray): The 2D array from the Kalman updater,
                                          where each row is a full augmented
                                          state vector for one ensemble member.
                                          Shape: (num_ensembles, n_physical_states + n_param_history).
            n_physical_states (int): The size of the physical state vector (q).
            n_param_history (int): The length of the parameter history vector.

        Returns:
            List[StateVector]: A list of reconstructed StateVector objects for the new analysis ensemble.
        """
        reconstructed_ensemble = []
        num_ensembles = analysis_matrix.shape[0]

        for i in range(num_ensembles):
            full_vector = analysis_matrix[i]
            
            # Slice the flat vector back into its constituent parts
            physical_state = full_vector[:n_physical_states]
            param_history = full_vector[n_physical_states:]
            
            # Basic validation
            if len(param_history) != n_param_history:
                raise ValueError(
                    f"Reconstruction error: expected param history of length {n_param_history}, "
                    f"but got {len(param_history)}."
                )
            
            reconstructed_ensemble.append(
                StateVector(physical_state=physical_state, param_history=param_history)
            )
            
        return reconstructed_ensemble