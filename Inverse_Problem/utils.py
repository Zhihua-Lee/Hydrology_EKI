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


def get_subwatershed(test_dict, id_list_use):
    """
    Get a sparse matrix representing the subwatershed and a map from link ID to division ID.
    
    Args:
        test_dict (dict): Test dictionary containing required parameters.
        id_list_use (List[int]): List of IDs to use for subsetting the subwatershed.

    Returns:
        Tuple[coo_matrix, dict]: A tuple containing:
            - sparse_parent: Sparse matrix representing the subwatershed.
            - link_to_division_map: A dictionary mapping each link ID to its division index.
    """
    # Gets division value
    watershed_csv = test_dict["watershed_csv"]
    watershed_depth = test_dict["watershed_depth"]
    watershed_vals = np.genfromtxt(watershed_csv, delimiter=',', skip_header=True)
    id_subwatershed = watershed_vals[:, 0]
    idx_sort = np.argsort(id_subwatershed)
    id_list = id_subwatershed[idx_sort]

    """或许需要改：更换了地形和link继承关系"""
    # Selects the relevent column for index
    if watershed_depth == 4:
        idx_col = 1
    elif watershed_depth == 5:
        idx_col = 2
    elif watershed_depth == 6:
        idx_col = 3
    elif watershed_depth == 7:
        idx_col = 4
    elif watershed_depth == 8:
        idx_col = 5

    id_divs = (watershed_vals[idx_sort, idx_col] - 1).astype(int)
    # id_divs = (watershed_vals[idx_sort, idx_col] ).astype(int) # If ids in the watershed file is starting from 0 but not 1

    id_tmp = []
    id_div_tmp = []
    
    # Get only ids in id_list_use
    for i, id_val in enumerate(id_list):
        if id_val in id_list_use:
            id_tmp.append(id_val)
            id_div_tmp.append(id_divs[i])
    id_tmp = np.array(id_tmp)
    # id_div_tmp = np.array(id_div_tmp)
    id_div_tmp_orig = np.array(id_div_tmp) # Keep original division IDs for mapping
    # Create the link_id to original division_id mapping BEFORE re-indexing
    # This map uses the original division IDs from the file (minus 1)
    link_to_division_map_orig = {int(link): int(div) for link, div in zip(id_tmp, id_div_tmp_orig)}

    
    # Assigns value from 0 to max to each for divisions, used to eliminate unused indices
    # divs_new = 0
    # max_div = np.max(id_div_tmp_orig)
    id_div_tmp_new = np.copy(id_div_tmp_orig) # Use a copy for re-indexing

    unique_orig_divs = np.sort(np.unique(id_div_tmp_orig))
    orig_to_new_map = {orig_div: new_idx for new_idx, orig_div in enumerate(unique_orig_divs)}

    # for i in range(max_div + 1):
    #     count_i = np.sum(id_div_tmp_orig == i)
    #     if count_i > 0:
    #         id_div_tmp[id_div_tmp == i] = divs_new
    #         divs_new += 1
    for i in range(len(id_div_tmp_new)):
        id_div_tmp_new[i] = orig_to_new_map[id_div_tmp_orig[i]]
        
    # Create the final map from link_id to the NEW, sequential division index
    link_to_division_map_final = {int(link): int(new_div) for link, new_div in zip(id_tmp, id_div_tmp_new)}

    id_num = len(id_tmp)
    subws_num = len(np.unique(id_div_tmp_new))
    
    # Create sparse matrix to convert from full parameters to sparse representation
    # Create sparse matrix using the new sequential division indices
    val_vals = np.ones(id_num)
    col_vals = np.arange(id_num)
    row_vals = id_div_tmp_new
    sparse_parent = coo_matrix((val_vals, (row_vals, col_vals)), shape=(subws_num, id_num))


    # Return both the matrix and the final mapping
    return sparse_parent, link_to_division_map_final

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