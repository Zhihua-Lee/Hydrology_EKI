#!/usr/bin/python
import sys
import shutil
import numpy as np
import os, time

from tqdm import tqdm
from utils import process_yaml, get_ids, get_subwatershed
from io_ifc import create_meas_sav, create_test_initial_condition, create_prm, create_gbl, create_batch_job_file, save_statistics_csv, save_particles, update_prm_add_or_overwrite_cr
from eki import subsample_data, pert, EnKF_step
from latent import create_latent, transform_latent
from run import run_test
from ifc_usgs_fileorder import load_usgs_mapping

import pandas as pd

import visualize

def main(yaml_name):
    # Read yaml file and get directories and number of steps
    test_dict = process_yaml(yaml_name)
    tmp_dir = test_dict['tmp_dir']
    out_dir = test_dict['out_dir']
    step_num = test_dict['steps']
    ens = test_dict['num_ensembles']

    # Presimulate data if using_simulated_data
    using_simulated_data = test_dict['using_simulated_data']
    print("using_simulated_data: ",using_simulated_data)
    print("data from: ",test_dict['meas_series'])
    
    if using_simulated_data:
        Cr_ref = test_dict['Cr_ref']
        update_prm_add_or_overwrite_cr(test_dict['prm'], Cr_ref)
        output_csv = test_dict['meas_series']
        sim_dir = os.path.join(os.path.dirname(test_dict['meas_series']), '')
        # delete old Cr_sim_data.csv if it exists
        if os.path.isfile(output_csv):
            print(f"Removing old {output_csv}")
            os.remove(output_csv)
        # Submit the presim job using the 'presimulate_Cr.sh' script.
        job_cmd = f"qsub {sim_dir}presimulate_Cr.sh"
        print(job_cmd)
        procs = os.system(job_cmd)
        # wait for Cr_sim_data.csv to be generated as our simulated observation
        while True:
            try:
                if os.path.isfile(output_csv):
                    data = open(output_csv).read()
                    if data.strip() != "":
                        # clean the first two rows
                        with open(output_csv, 'r') as f:
                            lines = f.readlines()
                        if len(lines) > 2:
                            lines_trimmed = lines[2:]
                            with open(output_csv, 'w') as f:
                                f.writelines(lines_trimmed)
                            print(f"Removed header lines from {output_csv}")
                        print(f"File {output_csv} detected and non-empty. Proceeding...")
                        break
                    else:
                        print(f"File {output_csv} exists but is empty. Waiting 10 seconds...")
                else:
                    print(f"Waiting for {output_csv} to appear...")
            except Exception as e:
                print(f"Error checking file: {e}")
            time.sleep(10)
    
    # Get data file directory, idx of locations, and standard deviation parameters
    data_file = test_dict['meas_series']
    usgs_gauge_id = test_dict['meas_usgs'] # usgs gauge id for observation
    meas_std = test_dict['abs_std_meas']
    rel_meas_std = test_dict['rel_std_meas']
    # loading USGS mapping
    usgs_to_link_id, link_to_usgs_id, file_order = load_usgs_mapping(test_dict)
    
    # Remove all temp files and copy yaml(initial, forcing, meas_csv) into out dir and tries to make output for csv and pickle outputs
    # for f in os.listdir(tmp_dir):
    #     os.remove(os.path.join(tmp_dir,f))
    for root, dirs, files in os.walk(tmp_dir, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            shutil.rmtree(os.path.join(root, name))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    shutil.copyfile(yaml_name, os.path.join(out_dir, 'test_config.j2'))
    shutil.copy(test_dict['initial_uini'], out_dir) # keep the filename
    shutil.copy(test_dict['meas_series'], out_dir) # keep the filename
    os.makedirs(out_dir + 'csv/', exist_ok=True)
    os.makedirs(out_dir + 'npy/', exist_ok=True)
    
    # Get list of link IDs from .prm(example) and create all necesary files
    model_link_ids = get_ids(test_dict)  # 来自 prm 文件
    if test_dict['watershed_csv'] is None: # if we have no watershed information, consider just using a global parameter
        sparse_parent = np.ones((1, 1))
    else:
        sparse_parent = get_subwatershed(test_dict, model_link_ids)
    latent_var = create_latent(test_dict, sparse_parent, ens) # initializing latent ensembles
    # For parameters with `dist=True`, the original values in the template `.prm` are ignored and the values are generated entirely by mapping the latent variables into the specified bounds; 
    # for parameters with `dist=False`, the original values from the template `.prm` are used unchanged.
    prm_ens, sorted_link_ids = transform_latent(test_dict, sparse_parent, latent_var)

    # Create(filtering based on lid) all necessary files for running tests
    create_meas_sav(test_dict, sorted_link_ids)
    create_test_initial_condition(test_dict, sorted_link_ids) # modified
    create_prm(test_dict, sorted_link_ids, prm_ens, ens)
    create_gbl(test_dict, ens)
    create_batch_job_file(test_dict, tmp_dir) # just need to create the submit.sh once and then modify parameters each iteration

    # filter out observation data based on observation gauge id, and specified timespan
    # Get data from csv file and seperate it into EKI / Plotting / IDs and save to file
    # data = np.genfromtxt(data_file, delimiter=',', skip_header=True, dtype=str, encoding='utf-8', filling_values='nan')
    # time_idx = np.array([np.datetime64(t.strip()) for t in data[:, 0]])
    # start_time = np.datetime64(test_dict['time_start'])
    # end_time   = np.datetime64(test_dict['time_end'])
    # filtered_data = data[(time_idx >= start_time) & (time_idx <= end_time)]
    # data_tmp = filtered_data[:, 1:].astype(float)
    if using_simulated_data:
        # 使用模拟数据时，CSV 文件没有表头和索引行
        df = pd.read_csv(data_file, header=None, dtype=str, na_values=[''], encoding='utf-8').fillna("0")
        # 检查最后一行是否全为空或者全为 "0"，如果是，则删除最后一行
        if df.iloc[-1].str.strip().eq("").all() or df.iloc[-1].eq("0").all():
            df = df.iloc[:-1, :]
        # 如果最后一列全为空或全为 "0"，删除最后一列
        if df.iloc[:, -1].str.strip().eq("").all() or df.iloc[:, -1].eq("0").all():
            df = df.iloc[:, :-1]    
        # 转换为浮点型矩阵
        data_tmp = df.astype(float).to_numpy()
    else:
        # 1. 读取 CSV，不让 pandas 自动解析日期
        df = pd.read_csv(data_file, index_col=0, dtype=str, na_values=[''], encoding='utf-8').fillna("0")
        # 2. 去掉索引字符串中的时区偏移，例如 '-06:00'
        #    假设所有行都含这个偏移，如果也有别的时区或不一致情况，需要更灵活地处理
        df.index = df.index.str.replace(r'-\d\d:\d\d$', '', regex=True)
        # 3. 将索引字符串解析为 datetime
        df.index = pd.to_datetime(df.index, errors='coerce')
        print("Index dtype after manual parse:", df.index.dtype)
        # 确保这里你看到的是 datetime64[ns]
        # 4. 若仍有无法转换的值，df.index 里会出现 NaT，你可以检查或丢弃 NaT
        if df.index.isna().any():
            print("Warning: Some indices could not be parsed to datetime!")
            df = df[~df.index.isna()]
        # 5. 现在索引已经是 tz-naive 的 datetime64[ns]，无需再 .tz_localize(None)
        #    直接进行时间范围过滤
        start_time = pd.to_datetime(test_dict['time_start'])
        end_time   = pd.to_datetime(test_dict['time_end'])
        df_filtered = df.loc[start_time:end_time]
        # 6. 提取数值部分并转换为 float
        data_tmp = df_filtered.astype(float).to_numpy()
    print("data shape:", data_tmp.shape)
    #col location in data file where the used gid is
    col_idx_gid = np.where(file_order == usgs_to_link_id[usgs_gauge_id])[0]
    data_use = data_tmp[:,col_idx_gid]
    data_plot, sav_ids = subsample_data(data_tmp, test_dict, sorted_link_ids, file_order)
    #TDOD: Change this so usgs_gauge_id can be a list of strings, to enable multiple sensors turned on simultaneously
    #col location in .sav where the used gid is
    col_idx_in_sav = np.where(sav_ids == usgs_to_link_id[usgs_gauge_id])[0] 
    save_statistics_csv(test_dict, sparse_parent, Y_mean=data_plot, Y_std=None, X_mat=None, name='csv/' + "meas")

    # EKI parameters (y = data, X = latent parameter ensemble, R = measurement uncertainty)
    y = np.reshape(data_use,(-1,1)) 
    R = (rel_meas_std * y.reshape(-1))**2 + meas_std**2
    X_post = latent_var

    # Run test
    for i in tqdm(range(step_num)):
        # Perturb previous parameters, run model, get simulation results - Prior
        X_prior = pert(X_post, test_dict, sparse_parent)   
        prm_ens_prior, _ = transform_latent(test_dict, sparse_parent, X_prior)
        create_prm(test_dict, sorted_link_ids, prm_ens_prior, ens) 
        Y_prior, Y_plot_prior, Y_plot_mean, Y_plot_std, _, _  = run_test(ens, X_prior, tmp_dir, col_idx_in_sav)    
        save_particles(test_dict, sparse_parent, X_prior, Y_plot_prior, name='npy/' + str(i) + '_prior')
        save_statistics_csv(test_dict, sparse_parent, Y_plot_mean, Y_plot_std, X_prior, name='csv/' + str(i) + "_prior")
        
        # Run EKI step, rerun model, record simulation results after assimilation - Posterior 
        X_post = EnKF_step(y, X_prior, Y_prior, R, test_dict, i)
        prm_ens_post, _ = transform_latent(test_dict, sparse_parent, X_post)
        create_prm(test_dict, sorted_link_ids, prm_ens_post, ens)    
        Y_post, Y_plot_post, Y_plot_mean, Y_plot_std, _, _ = run_test(ens, X_post, tmp_dir, col_idx_in_sav) 
        save_particles(test_dict, sparse_parent, X_post, Y_plot_post, name='npy/' + str(i) + "_post")
        save_statistics_csv(test_dict, sparse_parent, Y_plot_mean, Y_plot_std, X_post, name='csv/' + str(i) + "_post")

    # Visualization once EKI is done.
    visualize.main_visualization(test_dict)
       
    
if __name__ == "__main__": 
    yaml_name = sys.argv[1]
    # ens = int(sys.argv[2])
    # main(yaml_name, ens)
    main(yaml_name)
