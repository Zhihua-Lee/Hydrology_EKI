import numpy as np
import pandas as pd
from latent import transform_latent_to_physical
from typing import List, Tuple, Dict, Union
from utils import time_to_epoch, get_ids, get_subwatershed # <-- Add get_ids and get_subwatershed

from string import Template
import shutil

import os
from textwrap import dedent

from ifc_usgs_fileorder import load_usgs_mapping

import struct
import re # Import regex module for year matching

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

def create_gbl(test_dict: dict, ens: int) -> None:
    
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
        1 $INI_FILE

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
        2 60 $CSV_FILE

        %Where to put peakflow data
        %(0 = no output, 1 = .pea file, 2 = database)
        0 

        %.sav files for hydrographs and peak file (meas.sav)
        %(0 = save no data, 1 = .sav file, 2 = .dbc file, 3 = all links)
        1 $SAV_FILE
        0

        %Snapshot information (0 = none, 1 = .rec, 2 = database, 3 = .h5, 4 = recurrent .h5)
        0

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
    # 从配置中提取必要的参数
    start_time = test_dict["time_start"]
    end_time = test_dict["time_end"]
    epoch_time_start = int(time_to_epoch(start_time))
    epoch_time_end = int(time_to_epoch(end_time))
    tmp_dir = test_dict["tmp_dir"]
    scratch_dir = test_dict["scratch_dir"]


    for i in range(ens):
        # 对于每个 ensemble 成员，重新构造模板变量字典，
        template_vars = {
            "MODEL_NUM": test_dict["model_num"],
            "START_TIME": start_time,
            "END_TIME": end_time,
            "GLOBAL_PARAMS": "11 1 50 3 1 20 35 0 5 0 20 1.0",
            "RVR_FILE": test_dict["rvr"],
            "PRM_FILE": tmp_dir + str(i) + ".prm",
            "INI_FILE": tmp_dir + "init.uini",
            # "INI_FILE": tmp_dir + "init.rec",
            "RAIN_DIR": test_dict["rain_dir"],
            "EPOCH_START": str(epoch_time_start),
            "EPOCH_END": str(epoch_time_end),
            "EVAPO_FILE": test_dict["evapo"],
            "TEMP_FILE": test_dict["temp"],
            "CSV_FILE": tmp_dir + str(i) + ".csv",
            "SAV_FILE": tmp_dir + 'meas.sav' ,
            "HPC_SCRATCH_DIR": os.path.join(scratch_dir, str(i)),
        }
    
        member_template = Template(gbl_template_str)
        member_content = member_template.safe_substitute(template_vars)
        
        member_gbl_name = tmp_dir + str(i) + ".gbl"
        with open(member_gbl_name, "w") as f:
            f.write(member_content)


def create_prm_from_division_params(
    test_dict: dict, 
    link_to_division_map: dict, 
    physical_params_div: np.ndarray, 
    member_id: int
) -> None:
    """
    Creates a single PRM file for one ensemble member using division-level physical parameters.
    This function reads a template, then updates parameters for each link by looking up
    its division and applying the corresponding physical parameter.

    Args:
        test_dict (dict): Configuration dictionary.
        link_to_division_map (dict): A dictionary mapping each link ID to its division index.
        physical_params_div (np.ndarray): A 2D array of physical parameters at the division level.
                                          Shape: (n_total_physical_params, n_divisions).
                                          NaN indicates a parameter should not be updated.
        member_id (int): The ID of the ensemble member, used for the filename.
    """
    # 1. Read the template PRM file to get the base parameter structure
    prm_template_path = test_dict['prm']
    with open(prm_template_path, 'r') as f:
        prm_lines = [line for line in f.readlines() if line.strip()]
    
    template_id_list = [int(i.strip('\n')) for i in prm_lines[1::2]]
    prm_list_template = np.array([[float(i) for i in line.strip('\n').split()] for line in prm_lines[2::2]])
    
    # Create a dictionary for quick lookup of template parameters for any link ID
    template_params_dict = {link_id: params for link_id, params in zip(template_id_list, prm_list_template)}

    # 2. Open the output file
    tmp_dir = test_dict["tmp_dir"]
    prm_name = os.path.join(tmp_dir, f"{member_id}.prm")
    
    # The authoritative list of link IDs is now derived directly from the map
    sorted_link_ids = sorted(link_to_division_map.keys())
    n_links = len(sorted_link_ids)

    with open(prm_name, 'w') as f:
        # Write the header (total number of links)
        f.write(f"{n_links}\n")

        # 3. Iterate through each link required by the model
        for link_id in sorted_link_ids:
            # Start with the default parameters from the template
            final_params = template_params_dict.get(link_id, [0.0] * physical_params_div.shape[0]).copy()
            
            # Find the division this link belongs to
            division_id = link_to_division_map.get(link_id)
            
            if division_id is not None:
                # For each parameter type, check if it needs to be updated
                for param_idx in range(physical_params_div.shape[0]):
                    # Get the updated value for this parameter type and division
                    updated_val = physical_params_div[param_idx, division_id]
                    
                    # If the value is not NaN, it means it was updated by the EKI step
                    if not np.isnan(updated_val):
                        final_params[param_idx] = updated_val
            
            # Round all parameters to the required precision before writing
            rounded_params = [
                np.format_float_positional(p, precision=5, unique=False, fractional=False, trim='k')
                for p in final_params
            ]

            # Write the link ID and its final parameters to the file
            f.write(f"{link_id}\n")
            f.write(" ".join(map(str, rounded_params)) + "\n")

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

# def update_prm_add_or_overwrite_cr(prm_file_path, cr_value):
#     """
#     Update a .prm file by setting the 13th parameter (Cr) for each link.
#     If a link already has 13 or more parameters, Cr is overwritten.
#     If it has only 12, Cr is appended.

#     Args:
#         prm_file_path (str): Full path to the .prm file.
#         cr_value (float or str): The Cr value to apply.
#     """
#     with open(prm_file_path, 'r') as f:
#         lines = f.readlines()

#     updated_lines = []
#     i = 0
#     n = len(lines)

#     # Preserve leading blank lines and the "total count" line
#     while i < n and lines[i].strip() == "":
#         updated_lines.append(lines[i])
#         i += 1
#     if i < n:
#         updated_lines.append(lines[i])
#         i += 1

#     # Each link has two lines: one for ID, one for parameters
#     while i < n:
#         # Skip blank lines between blocks
#         while i < n and lines[i].strip() == "":
#             updated_lines.append(lines[i])
#             i += 1
#         # Link ID line
#         if i < n:
#             updated_lines.append(lines[i])
#             i += 1
#         # Skip blank lines before parameter line
#         while i < n and lines[i].strip() == "":
#             updated_lines.append(lines[i])
#             i += 1
#         # Parameter line
#         if i < n:
#             tokens = lines[i].strip().split()
#             if len(tokens) >= 13:
#                 tokens[12] = str(cr_value)
#             else:
#                 tokens.append(str(cr_value))
#             updated_lines.append(" ".join(tokens) + "\n")
#             i += 1

#     with open(prm_file_path, 'w') as f:
#         f.writelines(updated_lines)
#     print(f"Updated {prm_file_path}: set Cr (param #13) to {cr_value} for all links.")

def update_prm_by_division(prm_file_path: str, link_to_division_map: dict, cr_ref_vec: np.ndarray):
    """
    Updates a .prm file by assigning Cr values based on sub-watershed divisions.
    This version uses structure-aware parsing to robustly handle re-runs.

    Args:
        prm_file_path (str): Full path to the .prm file to modify.
        link_to_division_map (dict): A pre-computed dictionary mapping each link ID to its division index.
        cr_ref_vec (np.ndarray): A vector of Cr values where the index corresponds 
                                 to the sub-watershed division ID.
    """
    print(f"Executing structure-aware update for {prm_file_path}...")
    
    with open(prm_file_path, 'r') as f:
        lines = f.readlines()

    updated_lines = []
    
    # Handle the first line (total link count) separately
    if not lines:
        print("Warning: PRM file is empty.")
        return
    updated_lines.append(lines[0])
    
    # Process the rest of the file in pairs (ID line, Parameter line)
    i = 1
    while i < len(lines):
        # The current line should be the ID line.
        id_line = lines[i].strip()
        
        # Find the next non-empty line for parameters
        param_line_idx = i + 1
        while param_line_idx < len(lines) and not lines[param_line_idx].strip():
            param_line_idx += 1
            
        if not id_line: # If we encounter blank lines, just skip to the next
            i += 1
            continue
            
        if param_line_idx >= len(lines):
            # Reached end of file with a trailing ID line, just append it
            updated_lines.append(lines[i])
            break

        param_line = lines[param_line_idx].strip()

        try:
            current_link_id = int(id_line.split()[0])
            
            # Get the correct Cr value for this link
            division_id = link_to_division_map.get(current_link_id)
            
            if division_id is not None:
                cr_value_for_link = cr_ref_vec[division_id]
                
                # Modify the parameter line
                tokens = param_line.split()
                if len(tokens) >= 13:
                    tokens[12] = str(cr_value_for_link)
                else:
                    tokens.append(str(cr_value_for_link))
                
                # Add the original ID line and the MODIFIED parameter line
                updated_lines.append(lines[i])
                updated_lines.append(" ".join(tokens) + "\n")
            else:
                # If link not in map, keep original pair of lines
                updated_lines.append(lines[i])
                updated_lines.append(lines[param_line_idx])

        except (ValueError, IndexError):
            # If the "ID line" isn't a valid ID, treat both as unstructured text and preserve them
            print(f"Warning: Could not parse ID from line: '{id_line}'. Preserving original lines.")
            updated_lines.append(lines[i])
            updated_lines.append(lines[param_line_idx])

        # Jump index past the processed pair
        i = param_line_idx + 1

    # Write the updated content back to the file
    with open(prm_file_path, 'w') as f:
        f.writelines(updated_lines)
        
    print(f"Finished updating {prm_file_path}.")

def create_presim_prm_from_template(template_prm_path: str, output_prm_path: str, link_to_division_map: dict, cr_ref_vec: np.ndarray):
    """
    Creates a new .prm file from a template, assigning Cr values based on sub-watershed divisions.

    Args:
        template_prm_path (str): Full path to the template .prm file.
        output_prm_path (str): Full path for the newly created .prm file.
        link_to_division_map (dict): A pre-computed dictionary mapping each link ID to its division index.
        cr_ref_vec (np.ndarray): A vector of Cr values where the index corresponds 
                                 to the sub-watershed division ID.
    """
    print(f"Creating new PRM file '{output_prm_path}' from template '{template_prm_path}'...")
    
    with open(template_prm_path, 'r') as f:
        lines = f.readlines()

    updated_lines = []
    
    # Handle the first line (total link count) separately
    if not lines:
        print("Warning: Template PRM file is empty.")
        return
    updated_lines.append(lines[0])
    
    # Process the rest of the file in pairs (ID line, Parameter line)
    i = 1
    while i < len(lines):
        # The current line should be the ID line.
        id_line = lines[i].strip()
        
        # Find the next non-empty line for parameters
        param_line_idx = i + 1
        while param_line_idx < len(lines) and not lines[param_line_idx].strip():
            param_line_idx += 1
            
        if not id_line: # If we encounter blank lines, just skip to the next
            i += 1
            continue
            
        if param_line_idx >= len(lines):
            # Reached end of file with a trailing ID line, just append it
            updated_lines.append(lines[i])
            break

        param_line = lines[param_line_idx].strip()

        try:
            current_link_id = int(id_line.split()[0])
            
            # Get the correct Cr value for this link
            division_id = link_to_division_map.get(current_link_id)
            
            if division_id is not None:
                cr_value_for_link = cr_ref_vec[division_id]
                
                # Modify the parameter line
                tokens = param_line.split()
                if len(tokens) >= 13:
                    tokens[12] = str(cr_value_for_link)
                else:
                    tokens.append(str(cr_value_for_link))
                
                # Add the original ID line and the MODIFIED parameter line
                updated_lines.append(lines[i])
                updated_lines.append(" ".join(tokens) + "\n")
            else:
                # If link not in map, keep original pair of lines
                updated_lines.append(lines[i])
                updated_lines.append(lines[param_line_idx])

        except (ValueError, IndexError):
            # If the "ID line" isn't a valid ID, treat both as unstructured text and preserve them
            print(f"Warning: Could not parse ID from line: '{id_line}'. Preserving original lines.")
            updated_lines.append(lines[i])
            updated_lines.append(lines[param_line_idx])

        # Jump index past the processed pair
        i = param_line_idx + 1

    # Write the updated content back to the file
    with open(output_prm_path, 'w') as f:
        f.writelines(updated_lines)
        
    print(f"Finished creating {output_prm_path}.")

def create_presim_gbl(test_dict: dict, presim_prm_path: str, presim_gbl_path: str, output_csv_path: str) -> None:
    """
    Creates the .gbl file for the presimulation run.
    """
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
        1 $INI_FILE

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
        2 60 $CSV_FILE

        %Where to put peakflow data
        %(0 = no output, 1 = .pea file, 2 = database)
        0 

        %.sav files for hydrographs and peak file (meas.sav)
        %(0 = save no data, 1 = .sav file, 2 = .dbc file, 3 = all links)
        1 $SAV_FILE
        0

        %Snapshot information (0 = none, 1 = .rec, 2 = database, 3 = .h5, 4 = recurrent .h5)
        0

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

    template_vars = {
        "MODEL_NUM": test_dict["model_num"],
        "START_TIME": start_time,
        "END_TIME": end_time,
        "GLOBAL_PARAMS": "11 1 50 3 1 20 35 0 5 0 20 1.0",
        "RVR_FILE": test_dict["rvr"],
        "PRM_FILE": presim_prm_path,
        "INI_FILE": test_dict['initial_uini'],
        "RAIN_DIR": test_dict["rain_dir"],
        "EPOCH_START": str(epoch_time_start),
        "EPOCH_END": str(epoch_time_end),
        "EVAPO_FILE": test_dict["evapo"],
        "TEMP_FILE": test_dict["temp"],
        "CSV_FILE": output_csv_path,
        "SAV_FILE": test_dict['link_sav'],
        "HPC_SCRATCH_DIR": test_dict['scratch_dir'],
    }

    member_template = Template(gbl_template_str)
    member_content = member_template.safe_substitute(template_vars)
    
    with open(presim_gbl_path, "w") as f:
        f.write(member_content)

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
    sav_name = test_dict['meas_sav']  # Path to original SAV file (gauge IDs as strings)
    tmp_dir = test_dict['tmp_dir']

    # Load USGS mapping (assume keys are gauge ID strings, values are model link IDs as integers)
    usgs_to_link_id, _, _ = load_usgs_mapping(test_dict)

    # Read existing SAV file and filter lines
    with open(sav_name, 'r') as f:
        sav_lines = [line.strip() for line in f if line.strip()]
    new_lines = []
    for gauge_id in sav_lines:
        if gauge_id in usgs_to_link_id: #check if in the converting dictionary keys
            mapped_link_id = usgs_to_link_id[gauge_id]  # 例如得到整数形式的模型 link id
            if mapped_link_id in model_link_ids:
                new_lines.append(str(mapped_link_id))
        else:
            # 可选：打印警告信息，表明某个 gauge_id 没有在映射中找到
            print(f"Warning: Gauge ID {gauge_id} not found in USGS mapping.")

    # Write the filtered lines to a new SAV file
    tmp_sav_name = tmp_dir + "meas.sav"
    os.makedirs(os.path.dirname(tmp_sav_name), exist_ok=True)
    with open(tmp_sav_name, 'w') as f:
        for line in new_lines:
            f.write("%s\n" % line)

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

def save_statistics_csv(test_dict: dict, division_to_link_map: np.ndarray, Y_mean: np.ndarray, Y_std: np.ndarray = None, X_mat: np.ndarray = None, name: str = "results") -> None:
    """
    Save statistical results (mean, std) of model outputs (Y) and parameters (X) to CSV files.

    Args:
        test_dict (dict): The main configuration dictionary.
        division_to_link_map (np.ndarray): A sparse matrix mapping divisions to links, used for parameter transformation.
                                           Shape: (n_divisions, n_links).
        Y_mean (np.ndarray): The mean of the model output ensemble. Shape: (t_steps, n_links).
        Y_std (np.ndarray, optional): The standard deviation of the model output ensemble. Shape: (t_steps, n_links).
        X_mat (np.ndarray, optional): The latent parameter ensemble. Shape: (n_latent_params, n_divisions, n_ens).
        name (str, optional): A prefix for the output filenames (e.g., '0_prior').
    """
    out_dir = test_dict["out_dir"]
    tmp_dir = test_dict["tmp_dir"]
    sav_name = os.path.join(tmp_dir, "meas.sav")

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
        # Transform latent parameters to their physical representation for analysis.
        # FIX: Only save the *active* parameters to maintain compatibility with visualize.py
        # and to produce compact, NaN-free CSV files, just like the old version did.
        prm_dist_bool = [val.lower() == 'true' for val in test_dict["prm_dist"]]
        active_param_indices = [i for i, is_active in enumerate(prm_dist_bool) if is_active]

        n_divisions = division_to_link_map.shape[0]
        X_physical = transform_latent_to_physical(test_dict, X_mat, n_divisions)

        # Select only the active parameter rows before calculating statistics.
        X_physical_active = X_physical[active_param_indices, :, :]
        X_mean = np.mean(X_physical_active, axis=2) # axis=2 is the ensemble dimension
        X_std = np.std(X_physical_active, axis=2)

        X_name_mean = os.path.join(out_dir, f"{name}_params_mean.csv")
        np.savetxt(X_name_mean, X_mean, delimiter=",", fmt="%.5e") # This will save a 2D array (params, divisions)

        X_name_std = os.path.join(out_dir, f"{name}_params_std.csv")
        np.savetxt(X_name_std, X_std, delimiter=",", fmt="%.5e")

def save_particles(test_dict: dict, division_to_link_map: np.ndarray, X_particle: np.ndarray, Y_particle: np.ndarray, name: str = "results") -> None:
    """
    Save the entire ensemble of parameters (X) and model outputs (Y) to .npy files.

    Args:
        test_dict (dict): The main configuration dictionary.
        division_to_link_map (np.ndarray): A sparse matrix mapping divisions to links, used for parameter transformation.
                                           Shape: (n_divisions, n_links).
        X_particle (np.ndarray): The latent parameter ensemble. Shape: (n_latent_params, n_divisions, n_ens).
        Y_particle (np.ndarray): The model output ensemble. Shape: (n_ens, t_steps, n_links).
        name (str, optional): A prefix for the output filenames (e.g., '0_prior').
    """
    out_dir = test_dict["out_dir"]

    # Transform latent parameters to their physical representation before saving.
    # FIX: Only save the *active* parameters to NPY to maintain consistency with CSVs
    # and the expectations of visualize.py.
    prm_dist_bool = [val.lower() == 'true' for val in test_dict["prm_dist"]]
    active_param_indices = [i for i, is_active in enumerate(prm_dist_bool) if is_active]

    n_divisions = division_to_link_map.shape[0]
    X_physical_full = transform_latent_to_physical(test_dict, X_particle, n_divisions)
    X_physical_active = X_physical_full[active_param_indices, :, :]

    X_particle_name = os.path.join(out_dir, f"{name}_params_particles.npy")
    Y_particle_name = os.path.join(out_dir, f"{name}_particles.npy")
    
    with open(Y_particle_name, 'wb') as f:
        np.save(f, Y_particle)
        
    with open(X_particle_name, 'wb') as f:
        np.save(f, X_physical_active)

def create_batch_job_file(test_dict, tmp_dir: str) -> None:
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
        f.write('ensemble_id=$(($SGE_TASK_ID - 1))\n')
        f.write(f'scratch_path="{scratch_dir}/$ensemble_id"\n')
        f.write('mkdir -p "$scratch_path"\n')
        project_root = test_dict['project_root']
        executable_path = os.path.join(project_root, 'exec/asynch/bin/asynch')
        f.write(f'mpirun -np {num_parallel_slots} {executable_path} ' + tmp_dir + '$ensemble_id.gbl\n')
        f.write('rm -r "$scratch_path"\n')
        # hpchome/executables/asynch/bin/asynch