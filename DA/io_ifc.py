import numpy as np
import pandas as pd
from string import Template
import shutil
import struct
import re
from textwrap import dedent
from scipy.sparse import coo_matrix
from tqdm import tqdm

from latent import transform_latent_to_physical
from typing import List, Dict, Tuple, Union
from utils import time_to_epoch

import os
from textwrap import dedent

# ==============================================================================
# Utility Functions (Moved from data_handler)
# ==============================================================================
def get_rainfall_for_lids(target_lids: set, start_time_str: str, end_time_str: str, rain_dir: str) -> Dict[int, pd.DataFrame]:
    """
    Efficiently loads rainfall time series for a specific set of link IDs.
    It iterates through the data files once to extract all required series.
    Handles both hierarchical (rain_dir/YYYY/file) and flat (rain_dir/file) structures.

    Args:
        target_lids (set): A set of link IDs (int) to extract rainfall data for.
        start_time_str (str): Start time for the data window (e.g., "YYYY-MM-DD HH:MM").
        end_time_str (str): End time for the data window.
        rain_dir (str): Base directory containing the rainfall data.

    Returns:
        Dict[int, pd.DataFrame]: A dictionary where keys are link IDs and values are
                                 DataFrames with a 'Time' index and 'Rainfall' column.
    """
    if not os.path.isdir(rain_dir):
        print(f"Warning: Rainfall directory not found: {rain_dir}")
        return {}

    start_time = pd.to_datetime(start_time_str)
    end_time = pd.to_datetime(end_time_str)

    # --- Find all relevant rainfall files ---
    files_to_process = []
    years_in_range = {str(y) for y in range(start_time.year, end_time.year + 1)}
    
    potential_year_dirs = [os.path.join(rain_dir, item) for item in os.listdir(rain_dir) if os.path.isdir(os.path.join(rain_dir, item)) and re.fullmatch(r'(19|20)\d{2}', item)]
    
    dirs_to_scan = [p for p in potential_year_dirs if os.path.basename(p) in years_in_range]
    if not dirs_to_scan:
        dirs_to_scan = [rain_dir]

    for data_dir in dirs_to_scan:
        for filename in os.listdir(data_dir):
            if filename.isdigit():
                timestamp = pd.to_datetime(int(filename), unit='s')
                if start_time <= timestamp <= end_time:
                    files_to_process.append(os.path.join(data_dir, filename))

    if not files_to_process:
        print("Warning: No rainfall files found in the specified time range.")
        return {}

    # --- Initialize a dictionary to hold the data lists for each target LID ---
    rainfall_data_temp = {lid: [] for lid in target_lids}

    # --- Main Processing Loop ---
    for file_path in files_to_process:
        timestamp = pd.to_datetime(int(os.path.basename(file_path)), unit='s')
        try:
            with open(file_path, "rb") as f:
                raw_data = f.read()[4:]  # Skip header

            for lid, rainfall in struct.iter_unpack("if", raw_data):
                if lid in target_lids:
                    rainfall_data_temp[lid].append((timestamp, rainfall))
        except Exception as e:
            print(f"Warning: Could not process file {file_path}. Error: {e}")

    # --- Convert lists to sorted DataFrames ---
    final_rainfall_data = {}
    for lid, data_list in rainfall_data_temp.items():
        if data_list:
            df = pd.DataFrame(data_list, columns=['Time', 'Rainfall']).set_index('Time').sort_index()
            final_rainfall_data[lid] = df[~df.index.duplicated(keep='first')]
    
    return final_rainfall_data

def load_and_aggregate_rainfall_by_division(start_time_str, end_time_str, rain_dir, link_to_division_map):
    """
    Efficiently loads all rainfall data within a time window and aggregates it by sub-watershed division.
    This function reads each binary rainfall file only once.
    """
    if not os.path.isdir(rain_dir):
        print(f"Warning: Rainfall directory not found: {rain_dir}")
        return {}

    try:
        start_time = pd.to_datetime(start_time_str)
        end_time = pd.to_datetime(end_time_str)
    except ValueError as e:
        print(f"Error: Could not parse start/end time strings: {e}")
        return {}

    duration_hours = (end_time - start_time).total_seconds() / 3600.0
    if duration_hours <= 0:
        print(f"Warning: Time duration is non-positive ({duration_hours} hours). Cannot calculate average rate.")
        duration_hours = 1 

    # --- Find all relevant rainfall files ---
    files_to_process = []
    years_in_range = {str(y) for y in range(start_time.year, end_time.year + 1)}
    
    dirs_to_scan = []
    potential_year_dirs = []
    for item in os.listdir(rain_dir):
        item_path = os.path.join(rain_dir, item)
        if os.path.isdir(item_path) and re.fullmatch(r'(19|20)\d{2}', item):
            potential_year_dirs.append(item_path)

    years_in_range_set = {str(y) for y in range(start_time.year, end_time.year + 1)}
    relevant_year_dirs = [p for p in potential_year_dirs if os.path.basename(p) in years_in_range_set]

    if relevant_year_dirs:
        dirs_to_scan = relevant_year_dirs
    else:
        dirs_to_scan = [rain_dir]

    for data_dir in dirs_to_scan:
        for filename in os.listdir(data_dir):
            if filename.isdigit():
                timestamp = pd.to_datetime(int(filename), unit='s')
                if start_time <= timestamp <= end_time:
                    files_to_process.append(os.path.join(data_dir, filename))

    if not files_to_process:
        print("Warning: No rainfall files found in the specified time range.")
        return {}

    # --- Process files and aggregate rainfall ---
    num_divisions = max(link_to_division_map.values()) + 1
    division_rainfall_totals = np.zeros(num_divisions)
    
    print("Aggregating rainfall data by division...")
    for file_path in tqdm(files_to_process, desc="Aggregating Rainfall"):
        try:
            with open(file_path, "rb") as f:
                raw_data = f.read()
            
            if len(raw_data) < 8: continue
            raw_data = raw_data[4:]

            for lid, rainfall in struct.iter_unpack("if", raw_data):
                division_id = link_to_division_map.get(lid)
                if division_id is not None:
                    division_rainfall_totals[division_id] += rainfall
        except Exception as e:
            print(f"Warning: Could not process file {file_path}. Error: {e}")
            
    return {i: total / duration_hours for i, total in enumerate(division_rainfall_totals) if total > 0}

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
    usgs_csv_path = test_dict['usgs_csv']
    gauges_lid_sav_path = test_dict['link_sav']
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
    
    # Reverse mapping: link ID to USGS
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

def write_rec_file(file_path: str, model_num: str, link_ids: np.ndarray, state_matrix: np.ndarray):
    """
    Writes a physical state matrix to a .rec file.
    
    Args:
        file_path (str): The output file path.
        model_num (str): The model number (e.g., '602').
        link_ids (np.ndarray): Array of link IDs.
        state_matrix (np.ndarray): The physical state matrix. Shape: (n_links, 5).
    """
    with open(file_path, 'w') as f:
        f.write(f"{model_num}\n")
        f.write(f"{len(link_ids)}\n")
        f.write("0.0\n") # Default time value
        
        for i, link_id in enumerate(link_ids):
            f.write(f"{link_id}\n")
            # Format and write the 5 state variables for this link
            state_line = " ".join([f"{val:.6e}" for val in state_matrix[i, :]])
            f.write(state_line + "\n")

def parse_rec_file(rec_path: str) -> np.ndarray:
    """
    Parses an HLM .rec file and returns the full physical state matrix.

    Args:
        rec_path (str): Path to the .rec file.

    Returns:
        np.ndarray: A 2D numpy array of the physical states, shape (n_links, n_state_variables).
    """
    with open(rec_path, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
    
    # Skip header (3 lines) and extract state lines
    state_lines = lines[3::2]
    
    # Defensively parse lines, ignoring any that are empty after stripping whitespace.
    states = [np.fromstring(line, sep=' ') for line in state_lines if line]
    return np.array(states)

def load_and_process_observations(
    data_file_path: str,
    obs_config: Dict, 
    da_window_config: Dict,
    file_order: np.ndarray, 
    usgs_to_link_id: Dict, 
    sorted_link_ids: List[int],
    using_simulated_data: bool = False
) -> Tuple:
    """
    Loads and processes the observation data (either real or simulated).
    This function handles file reading, data cleaning, time filtering, and column selection.

    Args:
        data_file_path (str): The full path to the observation data file (real or synthetic).
        obs_config (Dict): The 'observations' section of the main config.
        da_window_config (Dict): The 'assimilation_window' section of the main config.
        file_order (np.ndarray): An array defining the column order in the observation data files.
        usgs_to_link_id (Dict): A mapping from USGS gauge IDs to internal model link IDs.
        sorted_link_ids (List[int]): A sorted list of all link IDs used in the model.
        using_simulated_data (bool): Flag to determine data processing logic. Defaults to False.

    Returns:
        Tuple: A tuple containing:
            - data_use (np.ndarray): The processed observation data for the specific gauge.
            - data_plot (np.ndarray): Subsampled data for plotting.
            - sav_ids (np.ndarray): The sensor IDs corresponding to the subsampled data.
            - col_idx_in_sav (np.ndarray): The column index in the .sav file for the observation gauge.
    """
    print("\n--- Loading and Processing Observation Data ---")
    # Handle single string or list of strings for backward compatibility and new feature
    usgs_gauge_ids = obs_config['real_time_usgs_gauges']
    if isinstance(usgs_gauge_ids, str):
        usgs_gauge_ids = [usgs_gauge_ids]

    if using_simulated_data:
        # Robust loading for HLM's quirky CSV format, matching EKI project's handling
        print("Processing SIMULATED observation data...")
        df = pd.read_csv(data_file_path, skiprows=2, header=None, dtype=str, na_values=[''], encoding='utf-8').fillna("0")
        # Drop the first column, which is an invalid index/duplicate from HLM
        if df.shape[1] > len(file_order):
            df = df.iloc[:, 0:]
        if df.iloc[-1].str.strip().eq("").all() or df.iloc[-1].eq("0").all():
            df = df.iloc[:-1, :]
        if df.iloc[:, -1].str.strip().eq("").all() or df.iloc[:, -1].eq("0").all():
            df = df.iloc[:, :-1]
    else:
        print("Processing REAL observation data...")
        df = pd.read_csv(data_file_path, index_col=0, dtype=str, na_values=[''], encoding='utf-8').fillna("0")

    # --- Time Filtering for REAL data ---
    # Simulated data is assumed to be generated for the correct window and does not need filtering.
    # Real data has a datetime index that needs parsing and filtering.
    if not using_simulated_data and not pd.api.types.is_numeric_dtype(df.index):
        df.index = df.index.str.replace(r'-\d\d:\d\d', '', regex=True)
        df.index = pd.to_datetime(df.index, errors='coerce')
        df = df[~df.index.isna()] # Drop rows where date parsing failed
        # Filter real data by time window
        start_time = pd.to_datetime(da_window_config['start'])
        end_time = pd.to_datetime(da_window_config['end'])
        print(f"Filtering data between {start_time} and {end_time}.")
        df = df.loc[start_time:end_time]

    data_tmp = df.astype(float).to_numpy()
        
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
    
    # NOTE: subsample_data requires 'tmp_dir' which is not in obs_config.
    # For now, we assume this function is not critical for DA logic and pass a placeholder.
    # This might need a more robust solution if subsampling is used later.
    data_plot, sav_ids = subsample_data(data_tmp, {'tmp_dir': './tmp/DA_run/'}, sorted_link_ids, file_order)
    
    # Find the column indices of the assimilation gauges within the model output files (whose columns are ordered by meas.sav)
    assimilation_link_ids = [usgs_to_link_id[gid] for gid in usgs_gauge_ids if gid in usgs_to_link_id]
    # Preserve the order from the config file, which is crucial for matching Y and y.
    indices = [np.where(sav_ids == lid)[0][0] for lid in assimilation_link_ids if np.where(sav_ids == lid)[0].size > 0]
    col_idx_in_sav = np.array(indices)

    print(f"Column indices for assimilation gauges in model output files: {col_idx_in_sav}")
    
    return data_use, data_plot, sav_ids, col_idx_in_sav

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

# def create_meas_sav(test_dict: dict, model_link_ids: list) -> None:
#     """
#     Create a filtered SAV file based on the EKI project's logic.
#     It reads an existing SAV file (with USGS IDs), maps them to model link IDs,
#     and filters them against the model's actual link IDs.

#     Args:
#         test_dict (dict): The main configuration dictionary.
#         model_link_ids (list): List of model link IDs (from .prm) for filtering.
#     """
#     # Adapt to the DA project's nested config structure
#     obs_config = test_dict.get('observations', {})
#     paths_config = test_dict.get('paths', {})
    
#     # Get necessary parameters from config
#     original_sav_path = obs_config['meas_sav']
#     tmp_dir = paths_config['tmp_dir']

#     # Load the mapping from USGS IDs to model link IDs
#     usgs_to_link_id, _, _ = load_usgs_mapping(test_dict)

#     # Read the original SAV file (contains USGS string IDs)
#     with open(original_sav_path, 'r') as f:
#         sav_lines = [line.strip() for line in f if line.strip()]
    
#     new_lines = []
#     for gauge_id_str in sav_lines:
#         # Check if this gauge is in our mapping dictionary
#         if gauge_id_str in usgs_to_link_id:
#             mapped_link_id = usgs_to_link_id[gauge_id_str]
#             # Check if this link ID actually exists in our model
#             if mapped_link_id in model_link_ids:
#                 new_lines.append(str(mapped_link_id))
#         else:
#             print(f"Warning: Gauge ID '{gauge_id_str}' from meas_sav not found in USGS mapping.")

#     # Write the filtered list of link IDs to a new SAV file in the temp directory
#     tmp_sav_path = os.path.join(tmp_dir, "meas.sav")
#     with open(tmp_sav_path, 'w') as f:
#         for line in new_lines:
#             f.write(f"{line}\n")

# ==============================================================================
# Input File Generation
# ==============================================================================

def _create_single_gbl(test_dict: dict, output_gbl_path: str, prm_file_path: str, 
                       input_rec_path: str, output_rec_path: str, 
                       sav_file_path: str, scratch_dir_path: str,
                       target_env: str):
    """
    A private worker function to create a single GBL file from a template.

    Args:
        ...
        target_env (str): The execution environment for which the GBL is being created.
                          Accepts 'login' or 'compute'. This determines which root path to use.
    """
    # 使用多行模板，保留参考版本中的注释和格式
    gbl_template_str = dedent("""
        %Model UID
        $MODEL_NUM

        %Begin and end date time
        $START_TIME 
        $END_TIME

        0\t%Parameters to filenames

        %Components to print
        1
        State0

        %Peakflow function
        Classic

        %Global parameters
        %9 v_0   lambda_1 lambda_2 Hu(mm)   infil(mm/hr) perc(mm/hr)  res_surf[minutes]  res_subsurf[days]  res_gw[days]
        $GLOBAL_PARAMS

        %No. steps stored at each link and
        %Max no. steps transfered between procs
        %Discontinuity buffer size
        30 10 30

        %Topology (0 = .rvr, 1 = database)
        0 $RVR_FILE

        %DEM Parameters (0 = .prm, 1 = database)
        0 $PRM_FILE

        %Initial state (0 = .ini, 1 = .uini, 2 = .rec, 3 = .dbc, 3 = .h5)
        2 $INPUT_REC_FILE

        %Forcings (0 = none, 1 = .str, 2 = binary, 3 = database, 4 = .ustr, 5 = forecasting, 6 = .gz binary, 7 = recurring)
        3

        %Rain
        5 $RAIN_DIR
        10 60 $EPOCH_START $EPOCH_END

        %Evaporation
        7 $EVAPO_FILE
        $EPOCH_START $EPOCH_END

        %Temperature 
        7 $TEMP_FILE
        $EPOCH_START $EPOCH_END

        %Dam (0 = no dam, 1 = .dam, 2 = .qvs)
        0

        %Reservoir ids (0 = no reservoirs, 1 = .rsv, 2 = .dbc file)
        0

        %Where to put write hydrographs
        %(0 = no output, 1 = .dat file, 2 = .csv file, 3 = database, 5 = .h5)
        0

        %Where to put peakflow data
        %(0 = no output, 1 = .pea file, 2 = database)
        0 

        %.sav files for hydrographs and peak file (meas.sav)
        %(0 = save no data, 1 = .sav file, 2 = .dbc file, 3 = all links)
        1 $SAV_FILE
        0

        %Snapshot information (0 = none, 1 = .rec single, 2 = .rec multiple, ...)
        1 $OUTPUT_REC_FILE

        %Filename for scratch work
        $HPC_SCRATCH_DIR

        %Numerical solver settings follow

        %facmin, facmax, fac
        .1 10.0 .9

        %Solver flag (0 = data below, 1 = .rkd)
        0
        %Numerical solver index (0-3 explicit, 4 implicit)
        2
        %Error tolerances (abs, rel, abs dense, rel dense)
        1E-2 1E-2 1E-2 1E-2 1E-2 1E-2 1E-2 1E-2 1E-2 1E-2
        1E-2 1E-2 1E-2 1E-2 1E-2 1E-2 1E-2 1E-2 1E-2 1E-2
        1E-2 1E-2 1E-2 1E-2 1E-2 1E-2 1E-2 1E-2 1E-2 1E-2
        1E-2 1E-2 1E-2 1E-2 1E-2 1E-2 1E-2 1E-2 1E-2 1E-2

        # %End of file
    """)

    start_time = test_dict["time_start"]
    end_time = test_dict["time_end"]
    epoch_time_start = int(time_to_epoch(start_time))
    epoch_time_end = int(time_to_epoch(end_time))
    
    # 根据 target_env 选择正确的根路径
    if target_env == 'compute':
        project_root = test_dict.get('compute_node_root')
        project_root = os.path.join(project_root, 'DA')
        output_gbl_path = os.path.join(test_dict.get('compute_node_root'), 'DA', output_gbl_path)
    elif target_env == 'login':
        project_root = ''#test_dict.get('login_node_root')
    else:
        raise ValueError(f"Invalid target_env: '{target_env}'. Must be 'login' or 'compute'.")

    # if not project_root:
    #     raise ValueError(f"Config dictionary is missing '{target_env}_node_root' for GBL generation.")

    template_vars = {
        "MODEL_NUM": test_dict["model_num"],
        "START_TIME": start_time,
        "END_TIME": end_time,
        "GLOBAL_PARAMS": "11 1 50 3 1 20 35 0 5 0 20 1.0",
        # Forcing data should always be absolute paths based on project_root
        "RVR_FILE": os.path.join(project_root, test_dict["rvr"]),
        "PRM_FILE": prm_file_path, # Already an absolute path
        "INPUT_REC_FILE": input_rec_path, # Already an absolute path
        "RAIN_DIR": os.path.join(project_root, test_dict["rain_dir"]),
        "EVAPO_FILE": os.path.join(project_root, test_dict["evapo"]),
        "TEMP_FILE": os.path.join(project_root, test_dict["temp"]),
        "OUTPUT_REC_FILE": output_rec_path,
        "SAV_FILE": sav_file_path,
        "EPOCH_START": str(epoch_time_start),
        "EPOCH_END": str(epoch_time_end),
        "HPC_SCRATCH_DIR": scratch_dir_path,
    }

    member_template = Template(gbl_template_str)
    member_content = member_template.safe_substitute(template_vars)
    
    with open(output_gbl_path, "w") as f:
        f.write(member_content)

def create_presim_gbl(test_dict: dict, presim_prm_path: str, presim_gbl_path: str, output_csv_path: str) -> None:
    """
    Creates the .gbl file for the presimulation run.
    """
    _create_single_gbl(
        test_dict=test_dict,
        output_gbl_path=presim_gbl_path,
        prm_file_path=presim_prm_path,
        # For presim, we still need to generate a CSV for the observation data
        # This is a special case. The GBL template needs to be different or adapted.
        # For now, we assume the DA loop is the priority.
        # A more robust solution would have a separate gbl_template for presim.
        sav_file_path=test_dict['link_sav'],
        scratch_dir_path=test_dict['scratch_dir']
    )

def create_ensemble_gbl(test_dict: dict, ens: int) -> None:
    """
    Creates a .gbl file for each member of the EKI ensemble.
    """
    tmp_dir = test_dict["tmp_dir"]

    for i in range(ens):
        _create_single_gbl(
            test_dict=test_dict,
            output_gbl_path=os.path.join(tmp_dir, f"{i}.gbl"),
            prm_file_path=os.path.join(tmp_dir, f"{i}.prm"),
            input_rec_path=os.path.join(tmp_dir, str(i), "state.rec"), # Worker creates this
            output_rec_path=os.path.join(tmp_dir, f"{i}.rec"), # HLM writes this
            sav_file_path=os.path.join(tmp_dir, "meas.sav"),
            scratch_dir_path=os.path.join(test_dict["scratch_dir"], str(i))
        )

def create_prm_from_division_params(
    test_dict: dict, 
    link_to_division_map: dict,
    physical_params_div_active: np.ndarray,
    active_param_indices: list[int],
    output_prm_path: str
) -> None:
    """
    Creates a single PRM file using division-level physical parameters for active params.
    This function reads a template, then updates parameters for each link by looking up
    its division and applying the corresponding physical parameter.

    Args:
        test_dict (dict): Configuration dictionary.
        link_to_division_map (dict): A dictionary mapping each link ID to its division index.
        physical_params_div_active (np.ndarray): A 2D array of *active* physical parameters.
                                                 Shape: (n_active_params, n_divisions).
        active_param_indices (list[int]): A list of original parameter indices (0-12)
                                          that correspond to the rows in `physical_params_div_active`.
        output_prm_path (str): The full path for the output .prm file.
    """
    # 1. Read the template PRM file to get the base parameter structure
    prm_template_path = test_dict['prm']
    with open(prm_template_path, 'r') as f:
        prm_lines = [line for line in f.readlines() if line.strip()]
    
    template_id_list = [int(line.strip().split()[0]) for line in prm_lines[1::2]]
    prm_list_template = np.array([[float(i) for i in line.strip('\n').split()] for line in prm_lines[2::2]])
    
    # Create a dictionary for quick lookup of template parameters for any link ID
    template_params_dict = {link_id: params for link_id, params in zip(template_id_list, prm_list_template)}

    # 2. Open the output file
    # The authoritative list of link IDs is now derived directly from the map
    sorted_link_ids = sorted(link_to_division_map.keys())
    n_links = len(sorted_link_ids)

    with open(output_prm_path, 'w') as f:
        # Write the header (total number of links)
        f.write(f"{n_links}\n")

        # 3. Iterate through each link required by the model
        for link_id in sorted_link_ids:
            # Start with the default parameters from the template
            n_total_params = prm_list_template.shape[1]
            final_params = template_params_dict.get(link_id, [0.0] * n_total_params).copy()
            
            # Find the division this link belongs to
            division_id = link_to_division_map.get(link_id)
            
            if division_id is not None:
                # For each *active* parameter, update its value in the final parameter list
                for active_idx, original_param_idx in enumerate(active_param_indices):
                    final_params[original_param_idx] = physical_params_div_active[active_idx, division_id]
            
            # Round all parameters to the required precision before writing
            rounded_params = [
                np.format_float_positional(p, precision=5, unique=False, fractional=False, trim='k')
                for p in final_params
            ]

            # Write the link ID and its final parameters to the file
            f.write(f"{link_id}\n")
            f.write(" ".join(map(str, rounded_params)) + "\n")


def create_presim_job_file(test_dict: dict, presim_dir: str, presim_gbl_path: str) -> str:
    """
    Create a batch job file for running the presimulation.

    Args:
        test_dict (dict): The configuration dictionary.
        presim_dir (str): The directory for presimulation files.
        presim_gbl_path (str): Path to the generated presimulation gbl file.

    Returns:
        str: The path to the created job file.
    """
    job_file_path = os.path.join(presim_dir, 'presimulate_job.sh')
    
    # Using settings from the original presimulate_Cr.sh
    num_parallel_slots = 112 
    parallel_argument = 'smp'
    queue = 'IFC'
    
    with open(job_file_path, 'w') as f:
        f.write('#!/bin/sh\n')
        f.write('#$ -N Presimulate_using_Cr_ref\n')
        f.write('#$ -j y\n')
        f.write('#$ -cwd\n')
        f.write(f'#$ -pe {parallel_argument} {num_parallel_slots}\n')
        f.write('#$ -l mf=16G\n')
        f.write(f'#$ -q {queue}\n')
        f.write('#$ -m es\n')
        f.write('#$ -M zli333@uiowa.edu\n')
        f.write('#$ -o /dev/null\n')
        f.write('#$ -e /dev/null\n')
        f.write('\n')
        f.write('/bin/echo Running on host: `hostname`.\n')
        f.write('/bin/echo In directory: `pwd`\n')
        f.write('/bin/echo Starting on: `date`\n')
        f.write('\n')
        f.write('module reset\n')
        f.write('module load openmpi\n')
        f.write('\n')
        f.write(f'mkdir -p {test_dict["scratch_dir"]}\n')
        project_root = test_dict['project_root']
        executable_path = os.path.join(project_root, 'exec/asynch/bin/asynch')
        f.write(f'mpirun -np {num_parallel_slots} {executable_path} {presim_gbl_path}\n')
        f.write(f'rm -r {test_dict["scratch_dir"]}\n')
        
    return job_file_path

def create_prm_generation_job_file(test_dict: dict, ens: int) -> str:
    """
    Creates the HPC batch job script for generating .prm files in parallel.
    """
    tmp_dir = test_dict['tmp_dir']
    job_file_path = os.path.join(tmp_dir, 'submit_prm_job.job')
    worker_script_path = os.path.join(test_dict['project_root'], 'Inverse_Problem', 'generate_prm_worker.py')

    with open(job_file_path, 'w') as f:
        f.write('#!/bin/bash\n')
        f.write('#$ -N prm_generation\n')
        f.write('#$ -j y\n')
        f.write('#$ -cwd\n')
        f.write(f'#$ -t 1-{ens}\n')
        f.write('#$ -l mf=2G\n') # Request modest memory for this simple task
        f.write('#$ -q IFC\n')
        f.write('#$ -o /dev/null\n')
        f.write('#$ -e /dev/null\n')
        f.write('\n')
        f.write('module reset\n')
        f.write('\n')
        python_executable = test_dict['hpc_python_path']
        f.write(f'{python_executable} {worker_script_path} {tmp_dir}\n')
    return job_file_path

def create_eki_run_job_file(test_dict, tmp_dir: str) -> None:
    """
    Create a batch job file for running EKI simulations.

    Args:
        tmp_dir (str): Temporary directory where the batch job file will be created.

    Returns:
        None
    """
    parallel_argument = test_dict['parallel_argument']
    num_parallel_slots = test_dict['num_parallel_slots']
    scratch_dir = test_dict['scratch_dir']
    
    with open(tmp_dir + 'submit_job.job', 'w') as f:
        f.write('#!/bin/bash\n')
        f.write('#$ -N EKI_job\n')
        # f.write('#$ -pe orte 2\n') # 每个数组作业任务都会获得 2 个 slot，而不是整个数组任务共用 2 个 slot。
        f.write(f'#$ -pe {parallel_argument} {num_parallel_slots}\n')
        # f.write('#$ -pe smp {num_parallel_slots}\n')
        # f.write('#$ -q AMCS\n')
        f.write('#$ -q IFC\n')
        f.write('#$ -cwd\n')
        # f.write(f'#$ -o "{tmp_dir}/$SGE_TASK_ID.out"\n')
        f.write('#$ -o /dev/null\n')
        # f.write(f'#$ -o {tmp_dir}/$TASK_ID.out\n')    # $TASK_ID is a placeholder that qsub can replace when parsing the script;
        f.write(f'#$ -e {tmp_dir}/$TASK_ID.err\n')      # but $SGE_TASK_ID only exists after the job has started.
        f.write('\n')
        f.write('/bin/echo Running on host: `hostname`.\n')
        f.write('/bin/echo In directory: `pwd`\n')
        f.write('/bin/echo Starting on: `date`\n')
        f.write('\n')
        f.write('module reset 2> /dev/null\n')
        f.write('module load openmpi\n')
        f.write('\n')
        f.write('ensemble_id=$(($SGE_TASK_ID - 1))\n')
        f.write(f'scratch_path="{scratch_dir}/$ensemble_id"\n')
        f.write('mkdir -p "$scratch_path"\n')
        project_root = test_dict['project_root']
        executable_path = os.path.join(project_root, 'exec/asynch/bin/asynch')
        f.write(f'mpirun -np {num_parallel_slots} {executable_path} {tmp_dir}/$ensemble_id.gbl\n')
        f.write('rm -r "$scratch_path"\n')
        # hpchome/executables/asynch/bin/asynch

def create_meas_sav(test_dict: dict, model_link_ids: list) -> None:
    """
    Create a filtered SAV file by mapping gauge IDs (from observation)
    to model link IDs and filtering by model_link_ids.

    Args:
        test_dict (dict): Test dictionary with parameters.
            Must contain keys 'meas_sav' (path to original SAV file),
            'tmp_dir' (temporary directory) and USGS mapping information.
        model_link_ids (list): List of model link IDs (from .prm) for filtering.

    Returns:
        None
    """
    # Get necessary parameters
    sav_name = test_dict['observations']['meas_sav']  # Path to original SAV file (gauge IDs as strings)
    tmp_dir = test_dict['paths']['tmp_dir']

    # Load USGS mapping (assume keys are gauge ID strings, values are model link IDs as integers)
    usgs_to_link_id, _, _ = load_usgs_mapping(test_dict['observations'])

    # Read existing SAV file and filter lines
    with open(sav_name, 'r') as f:
        sav_lines = [line.strip() for line in f if line.strip()]
    new_lines = []
    for gauge_id in sav_lines:
        if gauge_id in usgs_to_link_id: #check if in the converting dictionary keys
            mapped_link_id = usgs_to_link_id[gauge_id]  # e.g., get the integer model link id
            if mapped_link_id in model_link_ids:
                new_lines.append(str(mapped_link_id))
        else:
            # Optional: Print a warning that a gauge_id was not found in the mapping.
            print(f"Warning: Gauge ID {gauge_id} not found in USGS mapping.")

    # Write the filtered lines to a new SAV file
    tmp_sav_name = os.path.join(tmp_dir, 'meas.sav')
    os.makedirs(os.path.dirname(tmp_sav_name), exist_ok=True)
    with open(tmp_sav_name, 'w') as f:
        for line in new_lines:
            f.write("%s\n" % line)
            
# def create_meas_sav(test_dict: dict, id_list: list) -> None:
#     """
#     Create a filtered SAV file based on the given test dictionary and ID list for the test.

#     Args:
#         test_dict (dict): Test dictionary containing required parameters.
#         id_list (list): List of link IDs for filtering the SAV file.

#     Returns:
#         None
#     """
#     # Get necessary parameters
#     sav_name = test_dict['meas_sav'] # lids of gauges for observations
#     tmp_dir = test_dict['tmp_dir']

#     # loading USGS mapping
#     usgs_to_link_id, link_to_usgs_id, file_order = load_usgs_mapping(test_dict)

#     # Read existing SAV file and filter lines based on ID list
#     with open(sav_name, 'r') as f:
#         sav_lines = [line.strip() for line in f.readlines() if line.strip()] # lids of gauges for observations
#         new_lines = [line for line in sav_lines if usgs_to_link_id[int(line)] in id_list]     # lids of gauges in the sorted lids from ODE/.prm

#     # Write the filtered lines to a new SAV file
#     temp_sav_name = tmp_dir + "meas.sav"
#     os.makedirs(os.path.dirname(temp_sav_name), exist_ok=True)
#     with open(temp_sav_name, 'w') as f:
#         for line in new_lines:
#             f.write("%s\n" % line)

def create_test_initial_condition(test_dict: dict, id_list: list) -> None:
    """
    Copy the specified uini/rec file to the target temporary directory as "init.uini"/"init.rec".

    Args:
        test_dict (dict): Test dictionary containing required parameters.
        id_list (list): List of IDs (unused in this version).

    Returns:
        None
    """
    source_uini = test_dict['initial_uini']
    tmp_dir = test_dict['tmp_dir']
    dest_uini_path = tmp_dir + "init.uini"
    shutil.copyfile(source_uini, dest_uini_path)
    
    # source_rec = test_dict['initial_rec']
    # tmp_dir = test_dict['tmp_dir']
    # dest_rec_path = tmp_dir + "init.rec"
    # shutil.copyfile(source_rec, dest_rec_path)

# def create_test_rec(test_dict: dict, id_list: list) -> None:
#     """
#     Create a filtered REC file based on the given test dictionary and ID list.

#     Args:
#         test_dict (dict): Test dictionary containing required parameters.
#         id_list (list): List of IDs (integers) for filtering the REC file.

#     Returns:
#         None
#     """
#     # Get necessary parameters
#     rec_name = test_dict['rec']
#     tmp_dir = test_dict['tmp_dir']

#     # Read existing REC file and filter lines based on ID list
#     with open(rec_name, 'r') as f:
#         rec_lines = [line.strip() for line in f.readlines() if line.strip()]

#     id_num = len(id_list)
#     rec_lines[1] = str(id_num)

#     new_lines = rec_lines[:3]
#     id_lines = rec_lines[3::2]
#     state_lines = rec_lines[4::2]

#     for i, line in enumerate(id_lines):
#         if int(line) in id_list:
#             new_lines.append(line)
#             new_lines.append(state_lines[i])

#     # Write the filtered lines to a new REC file
#     rec_name = tmp_dir + "init.rec"
#     with open(rec_name, 'w') as f:
#         for item in new_lines:            
#             f.write("%s\n" % item)
#         # for item_1, item_2 in zip(new_lines[4::2], new_lines[5::2]):
#         #     f.write("%s\n" % item_1)
#         #     temp_item = ' '.join(item_2.split()[:-1])
#         #     f.write("%s\n" % temp_item)

# ==============================================================================
# Output Data Saving
# ==============================================================================

def save_statistics_csv(test_dict: dict, division_to_link_map: np.ndarray, Y_data: np.ndarray, X_mat: np.ndarray = None, name: str = "results") -> None:
    """
    Save statistical results (mean, std) of model outputs (Y) and parameters (X) to CSV files.
    This function can accept either a 3D particle array (and calculate stats) or a 2D pre-calculated mean array.

    Args:
        test_dict (dict): The main configuration dictionary.
        division_to_link_map (np.ndarray): A sparse matrix mapping divisions to links, used for parameter transformation.
                                           Shape: (n_divisions, n_links).
        Y_data (np.ndarray): The model output data. Can be the full ensemble (n_ens, t_steps, n_links)
                             or a pre-calculated mean (t_steps, n_links).
        X_mat (np.ndarray, optional): The latent parameter ensemble. If provided, its stats will also be saved.
                                      Shape: (n_latent_params, n_divisions, n_ens). Defaults to None.
        name (str, optional): A prefix for the output filenames (e.g., '0_prior').
    """
    out_dir = test_dict["out_dir"]
    tmp_dir = test_dict["tmp_dir"]
    sav_name = os.path.join(tmp_dir, "meas.sav")

    if Y_data.ndim == 3:  # Full particle set provided
        # Handle different potential particle shapes
        if Y_data.shape[0] == test_dict.get('num_ensembles', -1): # Shape: (n_ens, t_steps, n_links)
            axis_for_stats = 0
        elif Y_data.shape[2] == test_dict.get('num_ensembles', -1): # Shape: (t_steps, n_links, n_ens)
            axis_for_stats = 2
        else:
            print(f"Warning: Y_data has 3 dims but shape {Y_data.shape} doesn't match ensemble size. Cannot calc stats.")
            axis_for_stats = None

        if axis_for_stats is not None:
            Y_mean = np.mean(Y_data, axis=axis_for_stats)
            Y_std = np.std(Y_data, axis=axis_for_stats)

    else:  # Pre-calculated mean provided
        Y_mean = Y_data
        Y_std = None

    sav_val = np.genfromtxt(sav_name, delimiter=',', ndmin=1)
    title_y = sav_val.reshape(1, -1)

    Y_mean_out_content = np.concatenate((title_y, Y_mean), axis=0)
    out_name_mean = os.path.join(out_dir, f"{name}_mean.csv")
    np.savetxt(out_name_mean, Y_mean_out_content, delimiter=",", fmt="%.5e")

    if Y_std is not None:
        Y_std_out_content = np.concatenate((title_y, Y_std), axis=0)
        out_name_std = os.path.join(out_dir, f"{name}_std.csv")
        np.savetxt(out_name_std, Y_std_out_content, delimiter=",", fmt="%.5e")

    if X_mat is not None:
        # The returned parameters are already dense (active only).
        if X_mat.ndim == 3:
            prm_dist_bool = [str(val).lower() == 'true' for val in test_dict["prm_dist"]]
            active_param_indices = [i for i, is_active in enumerate(prm_dist_bool) if is_active]
            n_divisions = division_to_link_map.shape[0]
            X_physical_active = transform_latent_to_physical(test_dict, X_mat, n_divisions, active_param_indices)

            X_mean = np.mean(X_physical_active, axis=2)
            X_std = np.std(X_physical_active, axis=2)

            X_name_mean = os.path.join(out_dir, f"{name}_params_mean.csv")
            np.savetxt(X_name_mean, X_mean, delimiter=",", fmt="%.5e")

            X_name_std = os.path.join(out_dir, f"{name}_params_std.csv")
            np.savetxt(X_name_std, X_std, delimiter=",", fmt="%.5e")

def save_particles(test_dict: dict, division_to_link_map: np.ndarray, Y_particle: np.ndarray, X_particle: np.ndarray, name: str = "results") -> None:
    """
    Save the entire ensemble of parameters (X) and model outputs (Y) to .npy files.

    Args:
        test_dict (dict): The main configuration dictionary.
        division_to_link_map (np.ndarray): A sparse matrix mapping divisions to links, used for parameter transformation.
                                           Shape: (n_divisions, n_links).
        Y_particle (np.ndarray): The model output ensemble. Shape: (n_ens, t_steps, n_links).
        X_particle (np.ndarray): The latent parameter ensemble. Shape: (n_latent_params, n_divisions, n_ens).
        name (str, optional): A prefix for the output filenames (e.g., '0_prior').
    """
    out_dir = test_dict["out_dir"]

    # The returned parameters are already dense (active only).
    prm_dist_bool = [str(val).lower() == 'true' for val in test_dict["prm_dist"]]
    active_param_indices = [i for i, is_active in enumerate(prm_dist_bool) if is_active]
    n_divisions = division_to_link_map.shape[0]
    X_physical_active = transform_latent_to_physical(test_dict, X_particle, n_divisions, active_param_indices)

    X_particle_name = os.path.join(out_dir, f"{name}_params_particles.npy")
    Y_particle_name = os.path.join(out_dir, f"{name}_particles.npy")
    
    with open(Y_particle_name, 'wb') as f:
        np.save(f, Y_particle)
        
    with open(X_particle_name, 'wb') as f:
        np.save(f, X_physical_active)