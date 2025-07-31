from dateutil import parser
from scipy.sparse import coo_matrix
from typing import List, Tuple, Dict, Union
import numpy as np
import os
import yaml
import pandas as pd
from jinja2 import Environment, FileSystemLoader
from eki import subsample_data

## Utility functions

def process_yaml(yamlname: str, context: dict = None) -> dict:
    """
    读取 YAML 配置文件或 Jinja2 模板文件，并返回配置字典。
    如果文件扩展名为 .j2，则先渲染模板（使用 context 进行变量替换），
    然后加载渲染后的 YAML；否则，直接加载 YAML 文件。
    
    Args:
        yamlname (str): YAML 或 Jinja2 模板配置文件路径。
        context (dict, optional): 用于模板渲染的上下文变量。如果为 None，则使用空字典。
        
    Returns:
        dict: 配置字典。
    """
    ext = os.path.splitext(yamlname)[1].lower()
    
    if ext == ".j2":
        print(".j2")
        # 取得文件所在目录和文件名
        directory = os.path.dirname(yamlname) or "."
        filename = os.path.basename(yamlname)
        env = Environment(loader=FileSystemLoader(directory))
        template = env.get_template(filename)
        rendered = template.render()
        config = yaml.safe_load(rendered)
    else:
        with open(yamlname, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    return config

def process_json(json_name: str) -> dict:
    """
    Read and parse a JSON file.

    Args:
        json_name (str): Name of the JSON file.

    Returns:
        dict: Parsed JSON data as a dictionary.
    """
    with open(json_name) as f:
        test_dict = json.load(f)
    return test_dict

def time_to_epoch(time: str) -> float:
    """
    Convert a time string to epoch timestamp.

    Args:
        time (str): Time string in any valid format.

    Returns:
        float: Epoch timestamp representing the given time.
    """
    return parser.parse(time).timestamp()

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
    usgs_gauge_id = test_dict['meas_usgs']
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
    
    col_idx_gid = np.where(file_order == usgs_to_link_id[usgs_gauge_id])[0]
    print(f"Column index for observation gauge {usgs_gauge_id} in data file: {col_idx_gid}")
    data_use = data_tmp[:, col_idx_gid]
    
    data_plot, sav_ids = subsample_data(data_tmp, test_dict, sorted_link_ids, file_order)
    
    col_idx_in_sav = np.where(sav_ids == usgs_to_link_id[usgs_gauge_id])[0]
    print(f"Column index for observation gauge in .sav file: {col_idx_in_sav}")
    
    return data_use, data_plot, sav_ids, col_idx_in_sav
