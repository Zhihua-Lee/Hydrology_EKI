import json
import numpy as np
from typing import List, Tuple, Dict, Union

def convert_logical(str_list: list) -> list:
    """
    Convert a list of strings representing logical values to a list of Python booleans.

    Args:
        str_list (list): A list of strings representing logical values ("true" or "false").

    Returns:
        list: A list of Python booleans (True or False) corresponding to the logical values.
    """
    # Convert each string element in the input list to lower case and parse it as a JSON value
    # This will convert strings "true" and "false" to Python booleans True and False, respectively
    return [json.loads(i.lower()) for i in str_list]
    

def create_latent(test_dict: dict, division_to_link_map: np.ndarray, ens: int) -> np.ndarray:
    """
    Initializes the latent parameter ensemble (X) for the EKI process.

    This function generates the initial set of parameters in a latent (unbounded, normalized)
    space. These parameters correspond to watershed divisions, not individual links.

    Args:
        test_dict (dict): The main configuration dictionary.
        division_to_link_map (np.ndarray): A sparse matrix mapping divisions to links.
                                           Its shape (n_divisions, n_links) provides the number of divisions.
        ens (int): The number of ensemble members to generate.

    Returns:
        np.ndarray: The initial latent parameter ensemble array (X_post).
                    Shape: (n_latent_params, n_divisions, n_ens).
    """
    include_parameters = convert_logical(test_dict["prm_dist"])
    n_divisions = division_to_link_map.shape[0]
    n_active_params = sum(include_parameters)

    # Generate all random values from a standard normal distribution at once.
    # The distribution is centered at 0 with a standard deviation from the config.
    latent_mat = np.random.normal(0, test_dict['sig_P0'], (n_active_params, n_divisions, ens))

    if np.isnan(latent_mat).any():
        print("Warning: NaN found in the initial latent matrix created by create_latent!")

    return latent_mat

def unbounded_to_bounded(x: np.ndarray, lb: float, ub: float) -> np.ndarray:
    """
    Convert unbounded values to bounded values using a sigmoid transformation.

    Args:
        x (np.ndarray): Input array of unbounded values.
        lb (float): Lower bound of the desired bounded range.
        ub (float): Upper bound of the desired bounded range.

    Returns:
        np.ndarray: An array of bounded values mapped to the range [lb, ub].
    """
    # TODO: Include other transformed parameter distributions
    # Apply a sigmoid transformation to map the unbounded values to the range [0, 1]
    x_on_0_1 = (np.tanh(x) + 1) / 2.0

    # Scale and shift the values to the desired bounded range [lb, ub]
    res = lb + (x_on_0_1) * (ub - lb)

    # Check for NaN occurrences
    if np.isnan(res).any():
        print(f"Warning: NaN found in unbounded_to_bounded for lb={lb}, ub={ub}!")
        # Optional: You could also fill or clip NaN values here

    return res


def transform_latent_to_physical(
    test_dict: dict, 
    X_ensemble: np.ndarray, # Latent params. Shape: (n_active_params, n_divisions, [n_ens])
    n_divisions: int,
    active_param_indices: list[int] # List of original indices of active params
) -> np.ndarray:
    """
    Transforms latent variables into bounded, physical-space parameters at the DIVISION level.
    This function is vectorized and handles both a single member (2D) and a full ensemble (3D).
    It only returns the transformed values for the parameters that are active in the EKI.

    Args:
        test_dict (dict): Configuration dictionary.
        X_ensemble (np.ndarray): A 2D or 3D array of latent variables.
                                 Shape: `(n_active_params, n_divisions)` for a single member,
                                 or `(n_active_params, n_divisions, n_ens)` for the full ensemble.
        n_divisions (int): The number of watershed divisions.
        active_param_indices (list[int]): List of original indices (0-12) for active parameters.

    Returns:
        np.ndarray: A dense 2D or 3D array of active physical parameters. Shape is identical to X_ensemble.
    """
    lower_bounds = test_dict['prm_lb']
    upper_bounds = test_dict['prm_ub']
    
    num_active_params = len(active_param_indices)

    # Accommodate both 2D (single member) and 3D (full ensemble) inputs
    was_2d = X_ensemble.ndim == 2
    X_proc = np.expand_dims(X_ensemble, axis=2) if was_2d else X_ensemble
    n_ens = X_proc.shape[2]

    # Create a dense output array, matching the latent input shape
    physical_params_out = np.zeros_like(X_proc)

    for active_idx, original_idx in enumerate(active_param_indices):
        # Extract the 2D slice (divisions, ens) for the current active parameter
        latent_slice_2d = X_proc[active_idx, :, :]
        
        # Get bounds for the original parameter index and apply the transformation
        lb = float(lower_bounds[original_idx])
        ub = float(upper_bounds[original_idx])
        
        physical_values_2d = unbounded_to_bounded(latent_slice_2d, lb, ub)
        physical_params_out[active_idx, :, :] = physical_values_2d

    # Return array with shape corresponding to the input shape
    return np.squeeze(physical_params_out, axis=2) if was_2d else physical_params_out