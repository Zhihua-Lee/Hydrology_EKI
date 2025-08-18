import json 
import numpy as np
from typing import List, Tuple, Dict, Union
from metric_operator import event_meas_op

# ==============================================================================
# EKI Core
# ==============================================================================

def perturb_ensemble(X: np.ndarray, test_dict: dict, division_to_link_map: np.ndarray) -> np.ndarray:
    """
    Perturbs the latent parameter ensemble (X) by adding random noise.
 
    This function adds Gaussian noise to the existing latent parameter ensemble.
    The standard deviation of the noise for each parameter type is specified in the configuration.
    This step is essential for exploring the parameter space in the EKI.
 
    Args:
        X (np.ndarray): The latent parameter ensemble to be perturbed.
                        Shape: (n_latent_params, n_divisions, n_ens).
        test_dict (dict): The main configuration dictionary.
        division_to_link_map (np.ndarray): A sparse matrix mapping divisions to links.
                                           Used here to get the number of divisions.
 
    Returns:
        np.ndarray: The perturbed latent parameter ensemble.
                    Shape is the same as the input X.
    """
    prm_std = test_dict['prm_std']
    include_parameters = [json.loads(i.lower()) for i in test_dict["prm_dist"]]
    
    n_active_params, n_divisions, n_ens = X.shape
    
    # Create the noise array, scaling the standard deviation for each active parameter.
    noise = np.zeros_like(X)
    active_idx = 0
    for i, do_pert in enumerate(include_parameters):
        if do_pert:
            std = float(prm_std[i])
            noise[active_idx, :, :] = np.random.normal(0, std, (n_divisions, n_ens))
            active_idx += 1
            
    return X + noise

def _EnKF_standard(X_pre: np.ndarray, Y_pre: np.ndarray, y: np.ndarray, R_diag: np.ndarray) -> np.ndarray:
    """
    Perform the Standard Perturbed Observation Ensemble Kalman Filter (EnKF) update step
    using direct inversion. This method is efficient for low-dimensional observations.

    Args:
        X_pre (np.ndarray): Prior ensemble of latent parameters, flattened into a 2D array.
                            Shape: (n_params * n_divisions, n_ens).
        Y_pre (np.ndarray): Prior ensemble of model outputs (observations).
                            Shape: (n_obs, n_ens).
        y (np.ndarray): Actual observations (measurement).
                        Shape: (n_obs, 1).
        R_diag (np.ndarray): Diagonal elements of the measurement noise covariance matrix (R).
                             Shape: (n_obs,).

    Returns:
        np.ndarray: Posterior ensemble of latent parameters after the EnKF update.
                    Shape is the same as the input X_pre.
    """
    ens = X_pre.shape[1]
    y_num = len(y)

    #Computes state and measurement means
    xbar = np.mean(X_pre, axis=1, keepdims=True)
    ybar = np.mean(Y_pre, axis=1, keepdims=True)
    
    #Gets measurement perturbation and perturbs
    pert_vec = np.random.normal(0, 1, (y_num, ens))
    R = np.diag(R_diag)
    y_pert = y + np.sqrt(R) @ pert_vec
    
    #Computes Kalman Gain 
    X = (X_pre - xbar) / np.sqrt(ens - 1)
    Y = (Y_pre - ybar) / np.sqrt(ens - 1)
    # This is the expensive step for high-dimensional y: (Y @ Y.T) is (y_dim, y_dim)
    K = np.linalg.solve((Y @ Y.T + R).T, (X @ Y.T).T).T
    
    #Updates states (parameter vector)
    X_post = X_pre + K @ (y_pert - Y_pre)
    return X_post

def _EnKF_svd(X_pre: np.ndarray, Y_pre: np.ndarray, y: np.ndarray, R_diag: np.ndarray) -> np.ndarray:
    """
    Perform the EnKF update step using an SVD-based method to avoid direct inversion
    of the large observation covariance matrix. This is efficient for high-dimensional
    observations where y_dim >> ensemble_size.

    Args:
        X_pre (np.ndarray): Prior ensemble of latent parameters, flattened into a 2D array.
                            Shape: (n_params * n_divisions, n_ens).
        Y_pre (np.ndarray): Prior ensemble of model outputs. 
                            Shape: (n_obs, n_ens).
        y (np.ndarray): Actual observations. 
                        Shape: (n_obs, 1).
        R_diag (np.ndarray): Diagonal of obs noise covariance.
                             Shape: (n_obs,).

    Returns:
        np.ndarray: Posterior ensemble of latent parameters.
                    Shape is the same as the input X_pre.
    """

    n_ens = X_pre.shape[1]
    n_y = Y_pre.shape[0]

    # Compute ensemble means and deviations
    x_mean = np.mean(X_pre, axis=1, keepdims=True)
    y_mean = np.mean(Y_pre, axis=1, keepdims=True)
    X_prime = X_pre - x_mean
    Y_prime = Y_pre - y_mean

    # Perturb observations for each ensemble member
    pert_vec = np.random.normal(0, 1, (n_y, n_ens))
    # Efficiently perturb observations without creating a full R matrix
    y_pert = y + np.sqrt(R_diag)[:, np.newaxis] * pert_vec
    
    # Innovation (difference between perturbed obs and predictions)
    d = y_pert - Y_pre

    # Use SVD to solve the update equation efficiently
    # This avoids forming the (n_y, n_y) matrix Y*Y' + R
    
    # The core of the SVD method is to operate in the ensemble space (n_ens, n_ens)
    # This matrix is much smaller than the observation space matrix
    # Use element-wise division instead of full R_inv matrix multiplication
    M = (Y_prime.T / R_diag) @ Y_prime + (n_ens - 1) * np.eye(n_ens)
    
    # Solve for the update in ensemble space
    # (Y' R_inv d) has shape (n_ens, n_ens)
    update_ens_space = np.linalg.solve(M, (Y_prime.T / R_diag) @ d)

    # Project the update back to the parameter space
    update = X_prime @ update_ens_space
    
    X_post = X_pre + update
    return X_post

def EnKF(X_pre: np.ndarray, Y_pre: np.ndarray, y: np.ndarray, R_diag: np.ndarray) -> np.ndarray:
    """
    Dispatcher for the Ensemble Kalman Filter (EnKF) update step.

    This function checks the dimensions of the observation space (n_y) versus the
    ensemble size (n_en) and dynamically chooses the most efficient algorithm.
    - For n_y > n_en, it uses an SVD-based method (_EnKF_svd) to avoid inverting a large matrix.
    - Otherwise, it uses the standard direct inversion method (_EnKF_standard).

    Args:
        X_pre (np.ndarray): Prior ensemble of latent parameters, flattened into a 2D array.
                            Shape: (n_params * n_divisions, n_ens).
        Y_pre (np.ndarray): Prior ensemble of model outputs (observations).
                            Shape: (n_obs, n_ens).
        y (np.ndarray): Actual observations (measurement).
                        Shape: (n_obs, 1).
        R_diag (np.ndarray): Diagonal elements of the measurement noise covariance matrix (R).
                             Shape: (n_obs,).

    Returns:
        np.ndarray: Posterior ensemble of latent parameters after the EnKF update.
                    Shape is the same as the input X_pre.
    """
    n_y = Y_pre.shape[0]
    n_en = X_pre.shape[1]

    # Check if the observation dimension is significantly larger than the ensemble size
    if n_y > n_en:
        print(f"High-dimensional observation detected (y_dim={n_y}, ens_size={n_en}). Using SVD-based EnKF for efficiency.")
        return _EnKF_svd(X_pre, Y_pre, y, R_diag)
    else:
        print(f"Standard-dimensional observation (y_dim={n_y}, ens_size={n_en}). Using direct inversion EnKF.")
        return _EnKF_standard(X_pre, Y_pre, y, R_diag)

def EnKF_step(y: np.ndarray, X: np.ndarray, Y: np.ndarray, R: np.ndarray, test_dict: Dict[str, Union[str, float]], i: int) -> np.ndarray:
    """
    Perform an EnKF step based on the type of measurement specified in the test dictionary.

    Args:
        y (np.ndarray): 1D array representing the observation/measurement data.
                        Shape: (n_obs, 1).
        X (np.ndarray): The prior latent parameter ensemble. This MUST be a 2D array where
                        parameters have been flattened for each ensemble member.
                        Shape: (n_params * n_divisions, n_ens).
        Y (np.ndarray): 2D array representing the ensemble forecast time series.
                        Shape: (n_obs, n_ens).
        R (np.ndarray): 1D array representing the diagonal of the measurement error covariance.
                        Shape: (n_obs,).
        test_dict (Dict[str, Union[str, float]]): Test dictionary containing configuration parameters.
        i (int): Index of the EnKF step.

    Returns:
        np.ndarray: The updated ensemble of state vectors after the EnKF step.
                   Shape is the same as the input X.
    """

    # If using 'metric' only, just use metric and thresh every other iteration
    if test_dict["meas_type"] == 'metric':
        print("EnKF step: using metric")
        y_use, Y_use, R_use = event_meas_op(y, Y, R, test_dict)
        X_post = EnKF(X, Y_use, y_use, R_use)
        # print('i=',i,':',np.linalg.norm(X_post-X)/np.linalg.norm(X))
    
    # If using 'threshed series' only, just use y values larger than thresh_val
    elif test_dict["meas_type"] == 'threshed_series':
        print("EnKF step: using threshed_series")
        thresh_val = test_dict['thresh_val']
        idx_use = np.where(y > thresh_val) # (N_use, 1)
        thresh_idx = idx_use[0] # (N_use)
        y_use = y[thresh_idx, :]
        R_use = R[thresh_idx]
        Y_use = Y[thresh_idx, :]
        X_post = EnKF(X, Y_use, y_use, R_use)
    
    # If using 'metric & threshed series', switch between metric and thresh every other iteration
    elif test_dict["meas_type"] == 'metric+threshed_series':
        print("EnKF step: using metric+threshed_series")
        if np.mod(i, 2) == 0:
            y_use, Y_use, R_use = event_meas_op(y, Y, R, test_dict)
            X_post = EnKF(X, Y_use, y_use, R_use)
            # print('i=',i,':',np.linalg.norm(X_post-X)/np.linalg.norm(X))
        else:
            thresh_val = test_dict['thresh_val']
            idx_use = np.where(y > thresh_val)
            thresh_idx = idx_use[0]
            y_use = y[thresh_idx, :]
            R_use = R[thresh_idx]
            Y_use = Y[thresh_idx, :]
            X_post = EnKF(X, Y_use, y_use, R_use)
            # print('i=',i,':',np.linalg.norm(X_post-X)/np.linalg.norm(X))
    
    # Otherwise using 'series', just use standard EKI for obs series
    else:
        print("EnKF step: using series")
        X_post = EnKF(X, Y, y, R)

        # # ① Observed peak
        # y_val = float(np.max(y))
        # y_peak = np.array([[y_val]], dtype=float)
        # print(f"🟠 Observed peak: {y_val:.3f}")
    
        # # ② Peak of each member
        # Y_vals = np.max(Y, axis=0)
        # Y_peak = Y_vals.reshape(1, -1)
        # print(f"🔵 Simulated peaks (ensemble): {np.round(Y_vals, 3)}")
    
        # # ③ Corresponding observation error (could be changed to np.max(R))
        # R_val = float(R[np.argmax(y)])
        # R_peak = np.array([R_val], dtype=float)
        # # print(f"⚠️  Observation error used (R): {R_val:.3f}")
    
        # # Kalman update
        # X_post = EnKF(X, Y_peak, y_peak, R_peak)
    return X_post



