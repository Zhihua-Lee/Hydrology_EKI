from dateutil import parser
from scipy.sparse import coo_matrix
from typing import List, Tuple, Dict, Union
import numpy as np
import os
import pandas as pd

import struct
import re # Import regex module for year matching

#TODO: convert this into something that can be stored in a seperate file, so that it can be generalized

# ==============================================================================
# Utility Functions
# ==============================================================================

def get_rainfall_for_lid_from_config(target_lid, start_time_str, end_time_str, rain_dir):
    """
    Parses binary rainfall files to get data for a target LID within a time window.
    Handles both hierarchical (rain_dir/YYYY/file) and flat (rain_dir/file) structures.

    Parameters:
        target_lid (int): The Link ID (LID) to extract rainfall data for.
        start_time_str (str): Start time string (YYYY-MM-DD HH:MM or similar).
        end_time_str (str): End time string (YYYY-MM-DD HH:MM or similar).
        rain_dir (str): Base directory containing the rainfall data.

    Returns:
        pd.DataFrame: DataFrame with 'Time' index and 'Rainfall' column (mm/h).
                      Returns an empty DataFrame if no data is found or rain_dir is invalid.
    """
    rainfall_data = []
    if not os.path.isdir(rain_dir):
        print(f"Warning: Rainfall directory not found or invalid: {rain_dir}") # Keep important warnings
        return pd.DataFrame(rainfall_data, columns=["Time", "Rainfall"]).set_index("Time")

    try:
        start_time = pd.to_datetime(start_time_str)
        end_time = pd.to_datetime(end_time_str)
    except ValueError as e:
        print(f"Error: Could not parse start/end time strings: {e}") # Keep important errors
        return pd.DataFrame(rainfall_data, columns=["Time", "Rainfall"]).set_index("Time")

    # --- Detect directory structure ---
    dirs_to_scan = []
    potential_year_dirs = []
    for item in os.listdir(rain_dir):
        item_path = os.path.join(rain_dir, item)
        if os.path.isdir(item_path) and re.fullmatch(r'(19|20)\d{2}', item):
            potential_year_dirs.append(item_path)

    years_in_range_set = {str(y) for y in range(start_time.year, end_time.year + 1)}
    relevant_year_dirs = [p for p in potential_year_dirs if os.path.basename(p) in years_in_range_set]

    if relevant_year_dirs:
        # Use hierarchical structure
        dirs_to_scan = relevant_year_dirs
    else:
        # Use flat structure
        dirs_to_scan = [rain_dir]
    # --- End Structure Detection ---

    found_any_data = False

    # --- Scan Directories and Process Files ---
    for data_dir in dirs_to_scan:
        try:
            files_in_dir = [
                f for f in os.listdir(data_dir)
                if os.path.isfile(os.path.join(data_dir, f)) and f.isdigit()
            ]
            files = sorted(files_in_dir)
        except OSError as e:
            print(f"Warning: Could not list files in {data_dir}: {e}") # Keep important warnings
            continue

        if not files:
            continue

        for file in files:
            try:
                timestamp_s = int(file)
                timestamp = pd.to_datetime(timestamp_s, unit="s")

                # Efficiently skip files outside the primary time range
                if timestamp > end_time:
                    continue
                if timestamp < start_time:
                    continue

                # Process file if within range
                file_path = os.path.join(data_dir, file)
                with open(file_path, "rb") as f:
                    raw_data = f.read()

                if len(raw_data) < 8:
                    # print(f"Warning: Skipping potentially corrupt file (size < 8 bytes): {file_path}") # Keep warnings
                    continue

                raw_data = raw_data[4:]

                rainfall_value = 0.0
                found_lid = False
                try:
                    for lid, rainfall in struct.iter_unpack("if", raw_data):
                        if lid == target_lid:
                            rainfall_value = rainfall
                            found_lid = True
                            break
                except struct.error as se:
                     print(f"Warning: Struct unpacking error in file {file_path}: {se}. Skipping file.") # Keep warnings
                     continue

                rainfall_data.append((timestamp, rainfall_value))
                found_any_data = True

            except ValueError:
                # Warning for non-integer filenames (though isdigit should prevent this)
                print(f"Warning: Could not parse filename to int: {file} in {data_dir}") # Keep warnings
            except MemoryError:
                 print(f"Error: MemoryError reading file {file_path}. Skipping file.") # Keep errors
            except OSError as e:
                 print(f"Error: OSError reading file {file_path}: {e}. Skipping file.") # Keep errors
            except Exception as e:
                print(f"Error processing file {os.path.join(data_dir, file)}: {type(e).__name__} - {e}") # Keep errors

    # --- Final DataFrame Creation ---
    if not found_any_data:
        # No need to print if empty, function will just return empty DF
        return pd.DataFrame(columns=["Time", "Rainfall"]).set_index("Time")

    try:
        rainfall_df = pd.DataFrame(rainfall_data, columns=["Time", "Rainfall"])
        if not rainfall_df.empty:
            rainfall_df.set_index("Time", inplace=True)
            rainfall_df = rainfall_df.sort_index()
            rainfall_df = rainfall_df[~rainfall_df.index.duplicated(keep='first')]
            rainfall_df = rainfall_df.loc[start_time:end_time] # Precise final filtering
        else:
             return pd.DataFrame(columns=["Time", "Rainfall"]).set_index("Time")

    except Exception as e:
        print(f"Error creating/processing final DataFrame: {e}") # Keep errors
        return pd.DataFrame(columns=["Time", "Rainfall"]).set_index("Time")

    return rainfall_df

def load_usgs_mapping(test_dict: dict) -> Tuple[Dict[str, int], Dict[int, str], np.ndarray]:
    """
    Reads the mapping between USGS stations and link IDs from a CSV file specified in the config.
    It generates the usgs_2_id, id_2_usgs dictionaries, and the file_order array.

    The configuration dictionary must contain the key "usgs_csv", pointing to the CSV file path.
    The CSV file must have 'STAID' and 'LINKNO' columns.

    Returns:
        tuple: (usgs_2_id, id_2_usgs, file_order)
    """
    # Get CSV file path from test_dict
    usgs_csv_path = test_dict.get("usgs_csv")
    gauges_lid_sav_path = test_dict.get("link_sav")
    if not usgs_csv_path or not os.path.exists(usgs_csv_path):
        raise FileNotFoundError(f"USGS mapping CSV file not found: {usgs_csv_path}")
    
    return load_usgs_mapping_from_path(usgs_csv_path, gauges_lid_sav_path) 

def load_usgs_mapping_from_path(usgs_csv_path: str, gauges_lid_sav_path: str) -> Tuple[Dict[str, int], Dict[int, str], np.ndarray]:
    """
    Loads USGS station to link ID mappings from specified file paths.

    Args:
        usgs_csv_path (str): Path to the CSV file containing 'STAID' and 'LINKNO' columns.
        gauges_lid_sav_path (str): Path to the .sav file that defines the order of link IDs in output files.

    Returns:
        Tuple[Dict[str, int], Dict[int, str], np.ndarray]: A tuple containing:
            - usgs_2_id (Dict[str, int]): A dictionary mapping USGS station ID (str) to model link ID (int).
            - id_2_usgs (Dict[int, str]): The reverse mapping from model link ID (int) to USGS station ID (str).
            - file_order (np.ndarray): An array of link IDs representing the column order in output files.
    """
    # Read CSV data, assuming 'STAID' is the station number and 'LINKNO' is the link id
    df = pd.read_csv(usgs_csv_path, dtype=str).set_index('STAID')
    
    # Construct the USGS to link id mapping (converting to integer)
    usgs_2_id = df['LINKNO'].astype(int).to_dict()
    
    # 反向映射：link id 到 USGS
    id_2_usgs = {v: k for k, v in usgs_2_id.items()}

     # Read file_order (i.e., link id sequence) from the .sav file
    if not os.path.exists(gauges_lid_sav_path):
        raise FileNotFoundError(f"meas_sav file not found: {gauges_lid_sav_path}")
    with open(gauges_lid_sav_path, 'r') as f:
        lines = f.readlines()
    # Extract the first number (link id) from each line, skipping empty lines
    file_order = np.array([int(line.strip().split()[0]) for line in lines if line.strip() != ""])

    
    # # Define file_order, maintaining the order of 'LINKNO' from the CSV
    # file_order = df['LINKNO'].astype(int).values

    return usgs_2_id, id_2_usgs, file_order

def get_ids(test_dict: dict) -> List[int]:
    """
    Get the list of IDs from the PRM file specified in the test dictionary.
    This version is more robust against formatting errors in the ID lines.
    """
    prm_name = test_dict['prm']
    with open(prm_name, 'r') as f:
        prm_lines = [line for line in f.readlines() if line.strip()]
    
    id_list_prm = []
    # The lines with IDs should be at indices 1, 3, 5, ...
    for line in prm_lines[1::2]:
        try:
            # Split the line by spaces and try to convert the first element to an integer
            first_element = line.strip().split()[0]
            id_list_prm.append(int(first_element))
        except (ValueError, IndexError) as e:
            # Handle cases where the line is empty, not a number, or has other issues
            print(f"Warning: Could not parse an ID from line: '{line.strip()}'. Skipping. Error: {e}")
            
    id_list = np.sort(id_list_prm)
    return id_list


def get_subwatershed(test_dict: dict, sorted_link_ids: List[int]) -> Tuple[coo_matrix, Dict[int, int]]:
    """
    Constructs the mapping between watershed divisions and individual river links.

    This function reads a CSV file that defines how larger watershed areas (divisions)
    are broken down into smaller, individual river segments (links). It produces two key outputs:
    a sparse matrix that acts as a transformation operator and a direct dictionary mapping.

    Args:
        test_dict (dict): The configuration dictionary, containing keys like 'watershed_csv' 
                          and 'watershed_depth'.
        sorted_link_ids (List[int]): The authoritative, sorted list of all link IDs that the 
                                     model will simulate. This ensures consistency.

    Returns:
        Tuple[coo_matrix, Dict[int, int]]: A tuple containing:
        
        - division_to_link_map (coo_matrix): A sparse matrix of shape (n_divisions, n_links).
          A '1' at `matrix[i, j]` indicates that link `j` is part of division `i`.
          This matrix is primarily used via its transpose to efficiently broadcast or aggregate 
          parameters between the division level and the link level.
          
        - link_to_division_map (Dict[int, int]): A dictionary that provides a direct lookup from 
          any individual `link_id` (int) to its corresponding `division_id` (int).
    """
    watershed_csv = test_dict["watershed_csv"]
    watershed_depth = test_dict["watershed_depth"]
    
    # 1. Load the raw watershed division data from the CSV file.
    watershed_vals = np.genfromtxt(watershed_csv, delimiter=',', skip_header=True)
    id_subwatershed = watershed_vals[:, 0]
    idx_sort = np.argsort(id_subwatershed)
    id_list_from_file = id_subwatershed[idx_sort]

    # 2. Select the correct column for the desired watershed hierarchy depth.
    depth_to_col_map = {4: 1, 5: 2, 6: 3, 7: 4, 8: 5}
    if watershed_depth not in depth_to_col_map:
        raise ValueError(f"Unsupported watershed_depth: {watershed_depth}. Must be one of {list(depth_to_col_map.keys())}.")
    idx_col = depth_to_col_map[watershed_depth]
    
    # Division IDs from the file (adjusting for 1-based indexing).
    division_ids_from_file = (watershed_vals[idx_sort, idx_col] - 1).astype(int)

    # 3. Filter the data to include only the links present in the authoritative `sorted_link_ids`.
    id_tmp = []
    id_div_tmp = []
    for i, link_id in enumerate(id_list_from_file):
        if link_id in sorted_link_ids:
            id_tmp.append(link_id)
            id_div_tmp.append(division_ids_from_file[i])
    
    id_links_final = np.array(id_tmp)
    id_divisions_orig = np.array(id_div_tmp)

    # 4. Re-index the division IDs to be sequential (0, 1, 2, ...).
    # This is crucial for creating a minimal-sized sparse matrix.
    unique_orig_divs = np.sort(np.unique(id_divisions_orig))
    orig_to_new_map = {orig_div: new_idx for new_idx, orig_div in enumerate(unique_orig_divs)}
    
    id_divisions_new = np.array([orig_to_new_map[orig_div] for orig_div in id_divisions_orig])

    # 5. Create the final outputs.
    # The dictionary maps each link ID to its *new*, sequential division index.
    link_to_division_map = {int(link): int(new_div) for link, new_div in zip(id_links_final, id_divisions_new)}

    # The sparse matrix uses the re-indexed divisions and the final, sorted link list.
    n_links = len(sorted_link_ids)
    n_divisions = len(unique_orig_divs)
    
    # Create a mapping from link_id to its index in the sorted list for matrix column lookup.
    link_id_to_idx = {link_id: i for i, link_id in enumerate(sorted_link_ids)}
    
    row_vals = [link_to_division_map[link_id] for link_id in sorted_link_ids if link_id in link_to_division_map]
    col_vals = [link_id_to_idx[link_id] for link_id in sorted_link_ids if link_id in link_to_division_map]
    val_vals = np.ones(len(row_vals))

    division_to_link_map = coo_matrix((val_vals, (row_vals, col_vals)), shape=(n_divisions, n_links))

    return division_to_link_map, link_to_division_map

def load_and_process_observations(test_dict: Dict, file_order: np.ndarray, usgs_to_link_id: Dict, sorted_link_ids: List[int]) -> Tuple:
    """
    Loads and processes the observation data (either real or simulated).
    This function handles file reading, data cleaning, time filtering, and column selection.

    Args:
        test_dict (Dict): The main configuration dictionary for the experiment.
        file_order (np.ndarray): An array defining the column order in the observation data files.
        usgs_to_link_id (Dict): A mapping from USGS gauge IDs to internal model link IDs.
        sorted_link_ids (List[int]): A sorted list of all link IDs used in the model.

    Returns:
        Tuple: A tuple containing:
            - data_use (np.ndarray): The processed observation data for the specific gauge.
            - data_plot (np.ndarray): Subsampled data for plotting.
            - sav_ids (np.ndarray): The sensor IDs corresponding to the subsampled data.
            - col_idx_in_sav (np.ndarray): The column index in the .sav file for the observation gauge.
    """
    print("\n--- Loading and Processing Observation Data ---")
    data_file = test_dict['meas_series']
    # Handle single string or list of strings for backward compatibility and new feature
    usgs_gauge_ids = test_dict['meas_usgs']
    if isinstance(usgs_gauge_ids, str):
        usgs_gauge_ids = [usgs_gauge_ids]

    using_simulated_data = test_dict['using_simulated_data']

    if using_simulated_data:
        print("Processing SIMULATED observation data...")
        df = pd.read_csv(data_file, header=None, dtype=str, na_values=[''], encoding='utf-8').fillna("0")
        if df.iloc[-1].str.strip().eq("").all() or df.iloc[-1].eq("0").all():
            df = df.iloc[:-1, :]
        if df.iloc[:, -1].str.strip().eq("").all() or df.iloc[:, -1].eq("0").all():
            df = df.iloc[:, :-1]
        data_tmp = df.astype(float).to_numpy()
    else:
        print("Processing REAL observation data...")
        df = pd.read_csv(data_file, index_col=0, dtype=str, na_values=[''], encoding='utf-8').fillna("0")
        df.index = df.index.str.replace(r'-\d\d:\d\d', '', regex=True)
        df.index = pd.to_datetime(df.index, errors='coerce')
        if df.index.isna().any():
            print("Warning: Some indices could not be parsed to datetime!")
            df = df[~df.index.isna()]
        
        start_time = pd.to_datetime(test_dict['time_start'])
        end_time = pd.to_datetime(test_dict['time_end'])
        print(f"Filtering data between {start_time} and {end_time}.")
        df_filtered = df.loc[start_time:end_time]
        data_tmp = df_filtered.astype(float).to_numpy()
        
    print("Processed data shape:", data_tmp.shape)
    
    # --- MODIFIED LOGIC FOR MULTIPLE GAUGES ---
    all_data_use_cols = []
    print(f"Attempting to load data for assimilation gauges: {usgs_gauge_ids}")
    for usgs_id in usgs_gauge_ids:
        try:
            link_id = usgs_to_link_id[usgs_id]
            col_idx = np.where(file_order == link_id)[0]
            if col_idx.size > 0:
                # Extract the column as a (T, 1) array and append
                all_data_use_cols.append(data_tmp[:, col_idx])
                print(f"  - Found data for gauge {usgs_id} (LID: {link_id}) at column index {col_idx[0]} in the observation source file.")
            else:
                print(f"  - Warning: Could not find link ID {link_id} for gauge {usgs_id} in the data file's column order.")
        except KeyError:
            print(f"  - Warning: Gauge ID '{usgs_id}' not found in the usgs_to_link_id mapping.")

    if not all_data_use_cols:
        raise ValueError("No valid observation data could be loaded for any of the specified 'meas_usgs'.")
    
    # Concatenate the time series from all selected gauges column-wise -> (n_timesteps, n_gauges)
    data_use_all_gauges = np.concatenate(all_data_use_cols, axis=1)
    # Reshape into a single long vector in Fortran order, making data from each gauge contiguous.
    # This is crucial for multi-gauge 'metric' assimilation. (e.g., [g1_t1...g1_tN, g2_t1...g2_tN, ...])
    data_use = data_use_all_gauges.reshape(-1, 1, order='F')
    print(f"Final assimilation vector 'y' shape: {data_use.shape} (concatenated from {len(usgs_gauge_ids)} gauges)")
    
    data_plot, sav_ids = subsample_data(data_tmp, test_dict, sorted_link_ids, file_order)
    
    # Find the column indices of the assimilation gauges within the model output files (whose columns are ordered by meas.sav)
    assimilation_link_ids = [usgs_to_link_id[gid] for gid in usgs_gauge_ids if gid in usgs_to_link_id]
    # Preserve the order from the config file, which is crucial for matching Y and y.
    indices = [np.where(sav_ids == lid)[0][0] for lid in assimilation_link_ids if np.where(sav_ids == lid)[0].size > 0]
    col_idx_in_sav = np.array(indices)

    print(f"Column indices for assimilation gauges in model output files: {col_idx_in_sav}")
    
    return data_use, data_plot, sav_ids, col_idx_in_sav

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