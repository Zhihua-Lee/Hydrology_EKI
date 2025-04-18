import numpy as np
from latent import transform_latent_sparse
from typing import List, Tuple, Dict, Union
from utils import time_to_epoch

from string import Template
import shutil

import os
from textwrap import dedent

from ifc_usgs_fileorder import load_usgs_mapping

def create_gbl(test_dict: dict, ens: int) -> None:
    
    # 使用多行模板，保留参考版本中的注释和格式
    gbl_template_str = dedent("""
        %Model UID
        $MODEL_NUM

        %Begin and end date time
        $START_TIME 
        $END_TIME

        0	%Parameters to filenames

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
        $TMP_DIR

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
            "TMP_DIR": tmp_dir + "_" + str(i),
        }
    
        member_template = Template(gbl_template_str)
        member_content = member_template.safe_substitute(template_vars)
        
        member_gbl_name = tmp_dir + str(i) + ".gbl"
        with open(member_gbl_name, "w") as f:
            f.write(member_content)


def create_prm(test_dict: dict, id_list: list, prm_array: np.ndarray, ens: int) -> None:
    """
    Create PRM files based on the given test dictionary, ID list, and PRM array.

    Args:
        test_dict (dict): Test dictionary containing required parameters.
        id_list (list): List of IDs.
        prm_array (np.ndarray): Array of PRM values.
        ens (int): Number of PRM files to create.

    Returns:
        None
    """
    
    # Format is:
    # Total number of IDs
    # ID 1
    # Parameters
    # ID 2
    # ...
    
    # Initialize empty list of size total number of rows
    id_num = len(id_list)
    prm_list = [[] for _ in range(2 * id_num + 1)]

    
    tmp_dir = test_dict["tmp_dir"]
    prm_num = prm_array.shape[1]
    prm_list[0] = str(id_num)
    for i in range(ens):
        prm_name = tmp_dir + str(i) + ".prm"
        with open(prm_name, 'w') as f:
            for j in range(id_num):
                prm_list[1 + 2 * j] = str(id_list[j])
                prm_list[2 + 2 * j] = " ".join([str(item) for item in prm_array[:, j, i]])
            for item in prm_list:
                f.write("%s\n" % item)

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

def update_prm_add_or_overwrite_cr(prm_file_path, cr_value):
    """
    Update a .prm file by setting the 13th parameter (Cr) for each link.
    If a link already has 13 or more parameters, Cr is overwritten.
    If it has only 12, Cr is appended.

    Args:
        prm_file_path (str): Full path to the .prm file.
        cr_value (float or str): The Cr value to apply.
    """
    with open(prm_file_path, 'r') as f:
        lines = f.readlines()

    updated_lines = []
    i = 0
    n = len(lines)

    # Preserve leading blank lines and the "total count" line
    while i < n and lines[i].strip() == "":
        updated_lines.append(lines[i])
        i += 1
    if i < n:
        updated_lines.append(lines[i])
        i += 1

    # Each link has two lines: one for ID, one for parameters
    while i < n:
        # Skip blank lines between blocks
        while i < n and lines[i].strip() == "":
            updated_lines.append(lines[i])
            i += 1
        # Link ID line
        if i < n:
            updated_lines.append(lines[i])
            i += 1
        # Skip blank lines before parameter line
        while i < n and lines[i].strip() == "":
            updated_lines.append(lines[i])
            i += 1
        # Parameter line
        if i < n:
            tokens = lines[i].strip().split()
            if len(tokens) >= 13:
                tokens[12] = str(cr_value)
            else:
                tokens.append(str(cr_value))
            updated_lines.append(" ".join(tokens) + "\n")
            i += 1

    with open(prm_file_path, 'w') as f:
        f.writelines(updated_lines)
    print(f"Updated {prm_file_path}: set Cr (param #13) to {cr_value} for all links.")


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
    temp_sav_name = tmp_dir + "meas.sav"
    os.makedirs(os.path.dirname(temp_sav_name), exist_ok=True)
    with open(temp_sav_name, 'w') as f:
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

def save_statistics_csv(test_dict, sparse_parent, Y_mean, Y_std=None, X_mat=None, name="results"):
    """
    Save statistical results to CSV files.

    Args:
        test_dict (dict): Test dictionary containing required parameters.
        sparse_parent (dict): Sparse parent information.
        Y_mean (np.ndarray): Mean results to be saved to a CSV file.
        Y_std (np.ndarray, optional): Standard deviation results to be saved to a CSV file.
        X_mat (np.ndarray, optional): Parameter data to compute mean and standard deviation.
        name (str, optional): Prefix for the output CSV file names.

    Returns:
        None
    """
    # Get necessary parameters
    out_dir = test_dict["out_dir"]
    tmp_dir = test_dict["tmp_dir"]
    sav_name = tmp_dir + "meas.sav" 

    # Load data from SAV file (gauges)
    sav_val = np.genfromtxt(sav_name, delimiter=',', ndmin=1)
    sav_num = len(sav_val)
    title_y = sav_val.reshape(1, sav_num)

    # Save mean results to CSV
    Y_mean_out_content = np.concatenate((title_y, Y_mean), axis=0)
    out_name_mean = out_dir + str(name) + "_mean.csv"
    np.savetxt(out_name_mean, Y_mean_out_content, delimiter=",", fmt="%.5e")

    # Save standard deviation results to CSV if provided
    if Y_std is not None:
        Y_std_out_content = np.concatenate((title_y, Y_std), axis=0)
        out_name_std = out_dir + str(name) + "_std.csv"
        np.savetxt(out_name_std, Y_std_out_content, delimiter=",", fmt="%.5e")

    # Save parameter mean and standard deviation to CSV if X_mat is provided
    if X_mat is not None:
        X_sparse = transform_latent_sparse(test_dict, sparse_parent, X_mat)
        X_mean = np.mean(X_sparse, axis=1, keepdims=True)
        X_std = np.std(X_sparse, axis=1, keepdims=True)

        X_name_mean = out_dir + str(name) + "_params_mean.csv"
        np.savetxt(X_name_mean, X_mean, delimiter=",", fmt="%.5e")

        X_name_std = out_dir + str(name) + "_params_std.csv"
        np.savetxt(X_name_std, X_std, delimiter=",", fmt="%.5e")

def save_particles(test_dict, sparse_parent, X_particle, Y_particle, name="results"):
    """
    Save particle data to NPY files.

    Args:
        test_dict (dict): Test dictionary containing required parameters.
        sparse_parent (dict): Sparse parent information.
        X_particle (np.ndarray): Particle data to be saved to a NPY file.
        Y_particle (np.ndarray): Particle results to be saved to a NPY file.
        name (str, optional): Prefix for the output NPY file names.

    Returns:
        None
    """
    # Get necessary parameters
    out_dir = test_dict["out_dir"]
    tmp_dir = test_dict["tmp_dir"]
    sav_name = tmp_dir + "meas.sav" 

    # Load data from SAV file
    sav_val = np.genfromtxt(sav_name, delimiter=',', ndmin=1)
    sav_num = len(sav_val)
    title_y = sav_val.reshape(1, sav_num)

    # Transform particle latent parameters
    X_sparse = transform_latent_sparse(test_dict, sparse_parent, X_particle)

    # Save particle data to NPY files
    X_particle_name = out_dir + str(name) + '_params_particles.npy'
    Y_particle_name = out_dir + str(name) + '_particles.npy'
    
    with open(Y_particle_name, 'wb') as f:
        np.save(f, Y_particle)
        
    with open(X_particle_name, 'wb') as f:
        np.save(f, X_sparse)

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
    with open(tmp_dir + 'submit_job.job', 'w') as f:
        f.write('#!/bin/bash\n')
        f.write('#$ -N EKI_job\n')
        # f.write('#$ -pe orte 2\n') # 每个数组作业任务都会获得 2 个 slot，而不是整个数组任务共用 2 个 slot。
        f.write(f'#$ -pe {parallel_argument} {num_parallel_slots}\n')
        # f.write(f'#$ -pe smp {num_parallel_slots}\n')
        f.write('#$ -q IFC\n')
        f.write('#$ -cwd\n')
        f.write('#$ -o /dev/null\n')
        f.write('#$ -e /dev/null\n')
        f.write('\n')
        f.write('module reset\n')
        f.write('module load openmpi\n')
        f.write('\n')
        f.write('filename=$(($SGE_TASK_ID - 1))\n')
        f.write(f'mpirun -np {num_parallel_slots} /Users/zli333/DA/2025_EKI/exec/asynch/bin/asynch ' + tmp_dir + '$filename.gbl\n')
        # hpchome/executables/asynch/bin/asynch