import json 
import copy
import numpy as np
from typing import List, Tuple, Dict, Union

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
    R = np.diag(R_diag)
    pert_vec = np.random.normal(0, 1, (n_y, n_ens))
    y_pert = y + np.sqrt(R) @ pert_vec
    
    # Innovation (difference between perturbed obs and predictions)
    d = y_pert - Y_pre

    # Use SVD to solve the update equation efficiently
    # This avoids forming the (n_y, n_y) matrix Y*Y' + R
    R_inv = np.diag(1.0 / R_diag)
    
    # The core of the SVD method is to operate in the ensemble space (n_ens, n_ens)
    # This matrix is much smaller than the observation space matrix
    M = Y_prime.T @ R_inv @ Y_prime + (n_ens - 1) * np.eye(n_ens)
    
    # Solve for the update in ensemble space
    # (Y' R_inv d) has shape (n_ens, n_ens)
    update_ens_space = np.linalg.solve(M, (Y_prime.T @ R_inv @ d))

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
        y_use, Y_use, R_use = event_meas_op(y, Y, R)
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
            y_use, Y_use, R_use = event_meas_op(y, Y, R)
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

        # # ① 观测峰值
        # y_val = float(np.max(y))
        # y_peak = np.array([[y_val]], dtype=float)
        # print(f"🟠 Observed peak: {y_val:.3f}")
    
        # # ② 每个成员的峰值
        # Y_vals = np.max(Y, axis=0)
        # Y_peak = Y_vals.reshape(1, -1)
        # print(f"🔵 Simulated peaks (ensemble): {np.round(Y_vals, 3)}")
    
        # # ③ 对应观测误差（可改为 np.max(R)）
        # R_val = float(R[np.argmax(y)])
        # R_peak = np.array([R_val], dtype=float)
        # # print(f"⚠️  Observation error used (R): {R_val:.3f}")
    
        # # Kalman 更新
        # X_post = EnKF(X, Y_peak, y_peak, R_peak)
    return X_post


# ==============================================================================
# Event-Based Metric Operators
# ==============================================================================

def max_event_op(max_values_idx: List[List[int]], Y_pre: np.ndarray) -> np.ndarray:
    """
    Retrieve the events with maximum values from the input time series.

    Args:
        max_values_idx (List[List[int]]): List of lists containing the indices of maximum values in each event.
        Y_pre (np.ndarray): 2D array representing the original time series.

    Returns:
        np.ndarray: A 2D array containing the events with maximum values.
    """
    idxs = np.array(max_values_idx).squeeze()
    result = Y_pre[idxs, :]
    return result

def mean_event_op(event_list: List[List[int]], Y_pre: np.ndarray) -> np.ndarray:
    """
    Calculate the mean of the events found in the input ensemble time series.

    Args:
        event_list (List[List[int]]): List of lists containing the indices of each event.
        Y_pre (np.ndarray): 2D array representing the ensemble of time series.

    Returns:
        np.ndarray: A 2D array containing the mean of each event.
    """
    E = len(event_list) # Number of events
    N = Y_pre.shape[1] # Ensemble size
    result = np.zeros((E, N))
    for i, event_indices in enumerate(event_list):
        result[i, :] = np.mean(Y_pre[event_indices, :], axis=0)
    return result

def std_event_op(event_list: List[List[int]], Y_pre: np.ndarray) -> np.ndarray:
    """
    Calculate the standard deviation of the events found in the input ensemble time series.

    Args:
        event_list (List[List[int]]): List of lists containing the indices of each event.
        Y_pre (np.ndarray): 2D array representing the ensemble of time series.

    Returns:
        np.ndarray: A 2D array containing the standard deviation of each event.
    """
    E = len(event_list) # Number of events
    N = Y_pre.shape[1] # Ensemble size
    result = np.zeros((E, N))
    for i, event_indices in enumerate(event_list):
        result[i, :] = np.std(Y_pre[event_indices, :], axis=0)
    return result

def mean_y_event_op(event_list: List[List[int]], Y_pre: np.ndarray) -> np.ndarray:
    """
    Calculate the weighted mean of the location (in time) of the events found in the input ensemble time series.

    Args:
        event_list (List[List[int]]): List of lists containing the indices of each event.
        Y_pre (np.ndarray): 2D array representing the ensemble of time series.

    Returns:
        np.ndarray: A 2D array containing the mean time value of each event.
    """
    E = len(event_list) # Number of events
    N = Y_pre.shape[1] # Ensemble size
    result = np.zeros((E, N))
    
    #Weighted average (over time) weighted by value
    for i, event_indices in enumerate(event_list):
        weights = Y_pre[event_indices, :]
        weight_sum = np.sum(weights, axis=0)
        event = np.array(event_indices).reshape(-1, 1)
        weighted_sum = np.sum(event * weights, axis=0)
        result[i, :] = weighted_sum / weight_sum
    return result

def std_y_event_op(event_list: List[List[int]], Y_pre: np.ndarray) -> np.ndarray:
    """
    Calculate the weighted standard deviation of the location (in time) of the events found in the input ensemble time series.

    Args:
        event_list (List[List[int]]): List of lists containing the indices of each event.
        Y_pre (np.ndarray): 2D array representing the ensemble of time series.

    Returns:
        np.ndarray: A 2D array containing the standard deviation of the time location of each event.
    """
    E = len(event_list)
    N = Y_pre.shape[1]
    result = np.zeros((E, N))
    
    #Weighted average and std (over time) weighted by value
    mean_y = mean_y_event_op(event_list, Y_pre)
    for i, event_indices in enumerate(event_list):
        weights = Y_pre[event_indices, :]
        event = np.array(event_indices).reshape(-1, 1)
        denominator = (((len(event) - 1.0) / len(event)) * np.sum(weights, axis=0))
        result[i, :] = np.sqrt(np.sum(weights * (event - mean_y[i:i+1, :]) ** 2, axis=0) / denominator)
    return result

def slope_event_op(slope_idx: List[List[int]], Y_pre: np.ndarray) -> np.ndarray:
    """
    Calculate the slope of events for each column in the input time series.

    Args:
        slope_idx (List[List[int]]): List of lists containing the indices for calculating the slopes of events.
        Y_pre (np.ndarray): 2D array representing the original time series.

    Returns:
        np.ndarray: A 2D array containing the calculated slopes of events for each column in the time series.
    """
    E = len(slope_idx)
    N = Y_pre.shape[1]
    result = np.zeros((E, N))
    for i, event_indices in enumerate(slope_idx):
        x = np.array(event_indices).flatten()
        y = Y_pre[x, :]
        for j in range(N):
            result[i, j], _ = np.polyfit(x, y[:, j], 1)
    return result


# ==============================================================================
# Event Finding & Metric Calculation
# ==============================================================================

def find_events(y: np.ndarray, min_dist: int, min_thresh: float, min_length: int) -> Tuple[List[List[int]], List[List[float]]]:
    """
    Find events in a time series based on given conditions.

    Args:
        y (np.ndarray): Input time series.
        min_dist (int): Minimum distance allowed between two events.
        min_thresh (float): Minimum threshold value for identifying an event.
        min_length (int): Minimum length of an event (number of consecutive points).

    Returns:
        Tuple[List[List[int]], List[List[float]]]: A tuple containing two lists:
        1. List of lists containing the indices of each event found in the input time series.
        2. List of lists containing the corresponding values of each event found.
    """
    # TODO: make this work with several different sensors, currently only works with a single sensor,
    # can work asis with multiple, but will have weird effects at the interface between two vectorized
    # time series
    
    # Initialize event list
    event_list = []
    event_val_list = []
    check_list = copy.deepcopy(y) #measurement values
    check_list_idx = np.arange(len(y)) #measurement locations
    min_thresh = np.maximum(1e-4, min_thresh) #ensures 0 values are always excluded

    
    while len(check_list) > 0: # While there are still elements in the list
        # Finds the current largest value in list
        max_val_idx = np.argmax(check_list) 
        idx = check_list_idx[max_val_idx]
        y_idx = check_list[max_val_idx]

        # If this value is smaller than the minimum value, break
        if y[idx] < min_thresh:
            break

        #If there is no events currently, make a new event, add value to event
        if not event_list:
            event_list.append([idx])
            event_val_list.append([y_idx])
        else:
            #Check to see which value in the event list the value is indexwise closest to
            for i, event in enumerate(event_list):
                min_diff = min([abs(e - idx) for e in event])
                
                #If within the minimum index distance, add to that event, then stop
                if min_diff < min_dist:
                    event_list[i] = event + [idx]
                    event_val_list[i] = event_val_list[i] + [y_idx]
                    break
                #If we made it to the end and we havent added it yet, the value gets its own event
                elif i == len(event_list) - 1:
                    event_list.append([idx])
                    event_val_list.append([y_idx])
                    break
                    
        #Remove the value from the remaining values
        check_list = np.delete(check_list, max_val_idx)
        check_list_idx = np.delete(check_list_idx, max_val_idx)

    # Check all the events created, remove all the really short events
    i = 0
    while i < len(event_list):
        if len(event_list[i]) < min_length:
            event_list.pop(i)
            event_val_list.pop(i)
        else:
            i += 1

    return event_list, event_val_list


def find_metric_values(event_list, event_val_list):
    """
    Calculate various metrics for each event found in the input time series.

    Args:
        event_list (List[List[int]]): List of lists containing the indices of each event.
        event_val_list (List[List[float]]): List of lists containing the corresponding values of each event.

    Returns:
        Tuple: A tuple containing various metrics for each event:
        1. List of lists containing the indices of the peak values of each event.
        2. List of lists containing the peak values of each event.
        3. List of lists containing the mean values of each event.
        4. List of lists containing the slope values of each event.
        5. List of lists containing the y-intercept values of each event.
        6. List of lists containing the indices used to calculate the slope of each event.
        7. List of lists containing the standard deviation values of each event.
        8. List of lists containing the mean y-values of each event.
        9. List of lists containing the standard deviation of y-values of each event.
    """
    max_values = []
    mean_values = []
    slope_values = []
    slope_idx = []
    int_values = []
    max_values_idx = []
    std_values = []
    mean_y_values = []
    std_y_values = []
    # Note: this function only works on a vector, not a ensemble, see below for that. 
    # TODO: combine this function to work for both
    
    #Calculates metrics
    for events, values in zip(event_list, event_val_list):
        max_val_idx = np.argmax(values)
        x = np.array([e for i,e in enumerate(events) if e >= events[max_val_idx]])
        # y = np.log(np.array([v for i,v in enumerate(values) if events[i] >= events[max_val_idx]]))
        # slope, inter = np.polyfit(x, y, 1)
        # Consistently filter for recession limb points with positive values to avoid mismatched array lengths
        recession_mask = (np.array(events) >= events[max_val_idx]) & (np.array(values) > 0)
        x = np.array(events)[recession_mask]
        y_vals = np.array(values)[recession_mask]

        if len(x) < 2: # Check if enough points exist for a linear fit
            slope, inter = np.nan, np.nan
        else:
            y = np.log(y_vals)
            slope, inter = np.polyfit(x, y, 1)
        max_val_idx = np.argmax(values)
        max_value = values[max_val_idx]
        mean_value = np.mean(values)
        std_value = np.std(values)
        mean_y_value = np.sum(np.array(events)*values)/np.sum(values)
        std_y_value = np.sqrt(np.sum(values*(np.array(events)-mean_y_value.T)**2)/(((len(values)-1.0)/len(values))*np.sum(values)))
        
        corresponding_index = events[max_val_idx]
        max_values.append([max_value])
        mean_values.append([mean_value])
        slope_values.append([slope])
        slope_idx.append([x])
        int_values.append([inter])
        max_values_idx.append([corresponding_index])
        std_values.append([std_value])
        mean_y_values.append([mean_y_value])
        std_y_values.append([std_y_value])
    return max_values_idx, max_values, mean_values, slope_values, int_values, slope_idx, std_values, mean_y_values, std_y_values

def event_meas_op(y: np.ndarray, Y_pre: np.ndarray, R: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform various operations on events based on input data and parameters.

    Args:
        y (np.ndarray): 1D array representing the original time series.
        Y_pre (np.ndarray): 2D array representing the ensemble forecast time series.
        R (np.ndarray): diagonal of the measurement error covariance.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing the calculated event properties for observation,
                                                  ensemble forecast, and measurement error covariance.
    """
    #TODO: enable using more than just thses metrics. I.E allow a feature to specify the specific metrics included
    
    # Get the "metric" measurement 
    N_y = Y_pre.shape[0]
    n_samp = 1000
    min_dist = 24
    min_thresh = np.percentile(y[y > 0], 25)
    min_length = 72
    y_event_idx_list, y_event_list = find_events(y.flatten(), min_dist, min_thresh, min_length)
    y_max_idx, y_max, y_mean, y_slope, _, y_slope_idx, std_values, mean_y_values, std_y_values = find_metric_values(y_event_idx_list, y_event_list)
    y_event = np.concatenate((y_max, y_mean, std_values, mean_y_values, std_y_values))

    # Get the "metric" operator
    Y_pre_max = max_event_op(y_max_idx, Y_pre)
    Y_pre_mean = mean_event_op(y_event_idx_list, Y_pre)
    Y_pre_std = std_event_op(y_event_idx_list, Y_pre)
    Y_pre_y_mean = mean_y_event_op(y_event_idx_list, Y_pre)
    Y_pre_y_std = std_y_event_op(y_event_idx_list, Y_pre)
    Y_pre_event = np.concatenate((Y_pre_max, Y_pre_mean, Y_pre_std, Y_pre_y_mean, Y_pre_y_std))

    # Perturb the measurement measurements for emperical approximation of "event" covariance
    # Note: we are making the approximation of both uncorrelated metrics and gaussian metrics, 
    # this is required by EKI but there are other choices that could be made, like keeping R_event full rank
    # or making a different choice of gaussian approximation.
    y_pert_unbounded = y.reshape(-1, 1) + np.sqrt(R).reshape(-1, 1) * np.random.normal(0, 1, (N_y, n_samp))
    y_pert = np.maximum(y_pert_unbounded, 0)
    y_pert_max = max_event_op(y_max_idx, y_pert)
    y_pert_mean = mean_event_op(y_event_idx_list, y_pert)
    y_pert_std = std_event_op(y_event_idx_list, y_pert)
    y_pert_y_mean = mean_y_event_op(y_event_idx_list, y_pert)
    y_pert_y_std = std_y_event_op(y_event_idx_list, y_pert)
    y_pert_event = np.concatenate((y_pert_max, y_pert_mean, y_pert_std, y_pert_y_mean, y_pert_y_std))
    C_yy = np.cov(y_pert_event)
    R_event = np.diag(C_yy)

    return y_event, Y_pre_event, R_event


# ==============================================================================
# Data Subsampling
# ==============================================================================

def subsample_data(data: np.ndarray, test_dict: dict, id_list: List[int], file_order: np.ndarray) -> Tuple[np.ndarray]:
    """
    Subsample data based on the given test dictionary, list of IDs, and file order.

    Args:
        data (np.ndarray): The data to subsample.
        test_dict (dict): Test dictionary containing required parameters.
        id_list (List[int]): List of IDs for filtering the data.
        file_order (np.ndarray): Array containing the order of files.

    Returns:
        Tuple[np.ndarray]: A tuple containing the subsampled data and the corresponding list of IDs.
    """
    
    # Gets id values saved to meas sav file
    tmp_dir = test_dict["tmp_dir"]
    sav_name = tmp_dir + "meas.sav"
    sav_vals = np.array(np.genfromtxt(sav_name, delimiter=','), ndmin=1)
    
    # Gets data at id values
    new_lines = []
    sav_ids = []
    for i, id_val in enumerate(sav_vals):
        if id_val in id_list:
            sav_ids.append(id_val)
            new_lines.append(data[:, i:i + 1])
            
    # returns data values and ids
    ids_meas = np.array(sav_ids)
    meas_vals = np.concatenate(new_lines, 1)

    return meas_vals, ids_meas