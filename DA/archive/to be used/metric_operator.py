import copy
import numpy as np
from typing import List, Tuple, Dict, Union

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

def event_meas_op(y: np.ndarray, Y_pre: np.ndarray, R: np.ndarray, test_dict: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform event-based metric operations on multi-gauge data.

    This function processes each gauge's time series independently to find events
    and calculate metrics, then concatenates the results. It assumes the input
    arrays `y`, `Y_pre`, and `R` are flattened in Fortran ('F') order, meaning
    data for each gauge is contiguous.

    Args:
        y (np.ndarray): Flattened observation vector from all gauges. Shape: `(n_timesteps * n_gauges, 1)`.
        Y_pre (np.ndarray): Flattened model output ensemble. Shape: `(n_timesteps * n_gauges, n_ens)`.
        R (np.ndarray): Flattened diagonal of the measurement error covariance. Shape: `(n_timesteps * n_gauges,)`.
        test_dict (dict): The main configuration dictionary.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing the concatenated event metrics for
        the observation, the model ensemble, and the estimated metric error covariance diagonal.
    """
    # --- Configuration ---
    usgs_gauge_ids = test_dict['meas_usgs']
    if isinstance(usgs_gauge_ids, str):
        usgs_gauge_ids = [usgs_gauge_ids]
    n_gauges = len(usgs_gauge_ids)
    
    if n_gauges == 0:
        raise ValueError("Metric processing requires at least one gauge in 'meas_usgs'.")

    total_obs_len = y.shape[0]
    if total_obs_len % n_gauges != 0:
        raise ValueError(f"Total observation length ({total_obs_len}) is not divisible by the number of gauges ({n_gauges}).")
    n_timesteps = total_obs_len // n_gauges

    event_params = test_dict.get('event_finding', {})
    min_dist = event_params.get('min_dist', 24)
    min_thresh_pct = event_params.get('min_thresh_pct', 25)
    min_length = event_params.get('min_length', 72)
    n_samp = 1000
    
    all_y_events, all_Y_pre_events, all_R_events_diags = [], [], []

    for i in range(n_gauges):
        start_idx, end_idx = i * n_timesteps, (i + 1) * n_timesteps
        y_gauge, Y_pre_gauge, R_gauge = y[start_idx:end_idx, :], Y_pre[start_idx:end_idx, :], R[start_idx:end_idx]

        min_thresh = np.percentile(y_gauge[y_gauge > 0], min_thresh_pct) if np.any(y_gauge > 0) else 0
        y_event_idx_list, y_event_list = find_events(y_gauge.flatten(), min_dist, min_thresh, min_length)
        
        if not y_event_idx_list:
            print(f"Warning: No events found for gauge {i+1} ({usgs_gauge_ids[i]}). Skipping its metrics.")
            continue

        y_max_idx, y_max, y_mean, _, _, _, std_values, mean_y_values, std_y_values = find_metric_values(y_event_idx_list, y_event_list)
        y_event_gauge = np.concatenate((y_max, y_mean, std_values, mean_y_values, std_y_values))
        all_y_events.append(y_event_gauge)
        
        Y_pre_event_gauge = np.concatenate((
            max_event_op(y_max_idx, Y_pre_gauge), mean_event_op(y_event_idx_list, Y_pre_gauge),
            std_event_op(y_event_idx_list, Y_pre_gauge), mean_y_event_op(y_event_idx_list, Y_pre_gauge),
            std_y_event_op(y_event_idx_list, Y_pre_gauge)
        ))
        all_Y_pre_events.append(Y_pre_event_gauge)

        y_pert_unbounded = y_gauge + np.sqrt(R_gauge)[:, np.newaxis] * np.random.normal(0, 1, (n_timesteps, n_samp))
        y_pert = np.maximum(y_pert_unbounded, 0)
        y_pert_event = np.concatenate((
            max_event_op(y_max_idx, y_pert), mean_event_op(y_event_idx_list, y_pert),
            std_event_op(y_event_idx_list, y_pert), mean_y_event_op(y_event_idx_list, y_pert),
            std_y_event_op(y_event_idx_list, y_pert)
        ))
        C_yy_gauge = np.cov(y_pert_event)
        all_R_events_diags.append(np.diag(C_yy_gauge))

    if not all_y_events:
        print("Warning: No events were found for any gauge. Returning empty metric arrays.")
        return np.array([]), np.array([]), np.array([])
        
    return np.concatenate(all_y_events, axis=0), np.concatenate(all_Y_pre_events, axis=0), np.concatenate(all_R_events_diags, axis=0)
