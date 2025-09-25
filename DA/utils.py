from dateutil import parser
from scipy.sparse import coo_matrix
from typing import List, Tuple, Dict, Union
import numpy as np
import os
import shutil
import yaml
import pandas as pd
import logging
from jinja2 import Environment, FileSystemLoader
# +++ NEW: Import the transformation function +++
from latent import transform_latent_to_physical


## Utility functions

def process_yaml(yamlname: str) -> dict:
    """
    Reads a YAML configuration file or a Jinja2 template file and returns a config dictionary.
    If the file extension is .j2, it first renders the template (using context for variable substitution),
    then loads the rendered YAML. Otherwise, it directly loads the YAML file.
    
    Args:
        yamlname (str): The path to the YAML or Jinja2 template configuration file.
        
    Returns:
        dict: The configuration dictionary.
    """
    ext = os.path.splitext(yamlname)[1].lower()
    
    if ext == ".j2":
        print(".j2")
        # Get the directory and filename
        directory = os.path.dirname(yamlname) or "."
        filename = os.path.basename(yamlname)
        # Set up Jinja2 environment and render the template
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

def create_output_and_temp_dirs(paths_config: dict, clean_tmp: bool = True):
    """
    Creates the output and temporary directories.
    This function will ALWAYS perform a rigorous cleanup of the output directory (`out_dir`)
    to ensure a fresh start for each run.
    The temporary directory (`tmp_dir`) is cleaned only if `clean_tmp` is True.
    
    Args:
        paths_config (dict): A dictionary containing 'out_dir' and 'tmp_dir' keys.
        clean_tmp (bool): If True, also cleans the temporary directory before creation.
    """
    out_dir = paths_config.get('out_dir')
    tmp_dir = paths_config.get('tmp_dir')
    
    if out_dir:
        # Always perform a rigorous cleanup of the output directory to ensure a fresh start.
        print(f"Performing rigorous cleanup of output directory: {out_dir}")
        shutil.rmtree(out_dir, ignore_errors=True)
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(os.path.join(out_dir, 'csv'), exist_ok=True)
        os.makedirs(os.path.join(out_dir, 'npy'), exist_ok=True)
        print(f"Output directory ensured: {out_dir}")
    
    if tmp_dir:
        if clean_tmp:
            print(f"Performing rigorous cleanup of temporary directory: {tmp_dir}")
            shutil.rmtree(tmp_dir, ignore_errors=True)
        os.makedirs(tmp_dir, exist_ok=True)
        # Create structured subdirectories for better organization
        os.makedirs(os.path.join(tmp_dir, 'analysis_results'), exist_ok=True)
        os.makedirs(os.path.join(tmp_dir, 'forecast_files'), exist_ok=True)
        os.makedirs(os.path.join(tmp_dir, 'hpc_logs'), exist_ok=True)
        print(f"Temporary directory ensured: {tmp_dir}")

def save_da_step_outputs(config, t, forecast_ensemble, analysis_ensemble, hlm_runner):
    """
    Saves a comprehensive set of outputs for a single DA time step (t).
    This includes forecast values, real-time optimal analysis, and retrospective
    (smoothed) analysis results for the alpha parameter.

    Args:
        config (dict): The main configuration dictionary.
        t (int): The current time step.
        forecast_ensemble (List[StateVector]): The forecast ensemble BEFORE the update.
        analysis_ensemble (List[StateVector]): The analysis ensemble AFTER the update.
        hlm_runner (HLMRunner): The HLMRunner instance, used to get link IDs.
    """
    out_dir_csv = os.path.join(config['paths']['out_dir'], 'csv')
    
    # +++ NEW: Helper function to transform latent alphas to physical for saving +++
    def transform_alphas_for_saving(latent_alphas: np.ndarray, n_divisions: int) -> np.ndarray:
        # latent_alphas shape: (n_ens, n_history_len * n_divisions)
        # For saving, we often deal with arrays of alphas.
        prm_dist_bool = [str(val).lower() == 'true' for val in config['parameters']["prm_dist"]]
        active_param_indices = [i for i, is_active in enumerate(prm_dist_bool) if is_active]
        n_active_params = len(active_param_indices)

        # Reshape for transformer: (n_ens, n_divisions) -> (n_active_params, n_divisions, n_ens)
        latent_alphas_3d = latent_alphas.T.reshape(n_active_params, n_divisions, -1)
        physical_alphas_3d = transform_latent_to_physical(
            config['parameters'], latent_alphas_3d, n_divisions=n_divisions, active_param_indices=active_param_indices
        )
        return physical_alphas_3d.reshape(n_active_params, -1).T # Reshape back

    # --- 1. Save Mean Physical State (q_a) from the analysis ensemble ---
    all_q_a = np.array([vec.q for vec in analysis_ensemble])
    mean_q_a = np.mean(all_q_a, axis=0)
    q_a_df = pd.DataFrame(mean_q_a, index=hlm_runner.sorted_link_ids, columns=['discharge_m3s'])
    q_a_df.index.name = 'link_id'
    q_a_filename = os.path.join(out_dir_csv, f'analysis_q_mean_t{t:03d}.csv')
    q_a_df.to_csv(q_a_filename)
    logging.info(f"Saved mean physical state for t={t}.")

    # --- 2. Save Forecast Alpha (mean of alpha_r,t before update) ---
    all_latent_alpha_f = np.array([vec.get_current_parameter() for vec in forecast_ensemble])
    all_alpha_f = transform_alphas_for_saving(all_latent_alpha_f, hlm_runner.n_divisions)
    mean_alpha_f = np.mean(all_alpha_f, axis=0)
    # Save mean over all divisions
    alpha_f_df = pd.DataFrame({'forecast_alpha_mean': [np.mean(mean_alpha_f)]}, index=[t])
    alpha_f_df.index.name = 'time_step'
    alpha_f_filename = os.path.join(out_dir_csv, 'forecast_alpha_mean_timeseries.csv')
    alpha_f_df.to_csv(alpha_f_filename, mode='a', header=not os.path.exists(alpha_f_filename))

    # --- 3. Save Real-Time Optimal Alpha (mean of alpha_r,t after update) ---
    all_latent_alpha_a_current = np.array([vec.get_current_parameter() for vec in analysis_ensemble])
    all_alpha_a_current = transform_alphas_for_saving(all_latent_alpha_a_current, hlm_runner.n_divisions)
    mean_alpha_a_current = np.mean(all_alpha_a_current, axis=0)
    # Save mean over all divisions
    alpha_a_df = pd.DataFrame({'analysis_alpha_mean': [np.mean(mean_alpha_a_current)]}, index=[t])
    alpha_a_df.index.name = 'time_step'
    alpha_a_filename = os.path.join(out_dir_csv, 'analysis_alpha_mean_timeseries_realTimeOptimal.csv')
    alpha_a_df.to_csv(alpha_a_filename, mode='a', header=not os.path.exists(alpha_a_filename))

    # --- 4. Save the full Smoothed Alpha Window Snapshot ---
    # Calculate the mean of the entire parameter history window
    all_latent_alpha_histories = np.array([vec.alpha_r_history for vec in analysis_ensemble])
    n_ens, n_history, n_divs = all_latent_alpha_histories.shape

    # Transform each time step in the history for each ensemble member
    # Reshape from (n_ens, n_hist, n_divs) to (n_ens, n_hist * n_divs)
    all_latent_flat = all_latent_alpha_histories.reshape(n_ens, -1)
    all_physical_flat = transform_alphas_for_saving(all_latent_flat, n_history * n_divs)
    # Reshape back to (n_ens, n_hist, n_divs)
    all_physical_histories = all_physical_flat.reshape(n_ens, n_history, n_divs)

    # Now calculate the mean of the physical values
    mean_alpha_history = np.mean(all_physical_histories, axis=0)
    # For saving, take the mean across all divisions for each time lag
    mean_alpha_history_across_divs = np.mean(mean_alpha_history, axis=1)
    
    # Create a DataFrame for the window snapshot
    history_lags = np.arange(len(mean_alpha_history))
    window_df = pd.DataFrame({
        'history_lag': history_lags,
        'smoothed_alpha_mean': mean_alpha_history_across_divs
    })
    window_filename = os.path.join(out_dir_csv, f'analysis_alpha_window_t{t:03d}.csv')
    window_df.to_csv(window_filename, index=False)

    # --- 5. Update and save the Retrospective Optimal Alpha timeseries ---
    retro_filename = os.path.join(out_dir_csv, 'analysis_alpha_mean_timeseries_retrospectiveOptimal.csv')
    try:
        # Try to load the existing retrospective file
        retro_df = pd.read_csv(retro_filename, index_col='time_step')
    except FileNotFoundError:
        # If it doesn't exist, create an empty DataFrame
        retro_df = pd.DataFrame()
        retro_df.index.name = 'time_step'

    # The historical time steps corresponding to the mean_alpha_history vector
    # e.g., for t=10 and a history of 3, the times are [10, 9, 8]
    history_timesteps = t - history_lags

    # Create a new DataFrame for the current update step's data
    new_column_name = f'corrected_at_t{t}'
    new_data_df = pd.DataFrame({new_column_name: mean_alpha_history_across_divs}, index=history_timesteps)
    new_data_df.index.name = 'time_step'
    
    # --- BUG FIX: Make the join robust against reruns ---
    # If a column for the current timestep already exists (from a previous failed run), drop it first.
    if new_column_name in retro_df.columns:
        retro_df = retro_df.drop(columns=[new_column_name])

    # Use a robust outer join to merge the new column, ensuring new rows are added.
    retro_df = retro_df.join(new_data_df, how='outer')

    # Sort index and columns for better readability
    retro_df = retro_df.sort_index()
    retro_df = retro_df.reindex(sorted(retro_df.columns, key=lambda col: int(col.split('t')[-1])), axis=1)

    retro_df.to_csv(retro_filename)

    logging.info(f"Saved comprehensive analysis outputs for t={t} to {out_dir_csv}")

# def archive_input_data(config: dict, usgs_to_link_id: dict, file_order: np.ndarray, data_file_to_archive: str):
#     """
#     Extracts the observation time series for all specified plot gauges from the
#     original data source (real or synthetic) and saves them to a clean CSV file.
# 
#     Args:
#         config (dict): The main configuration dictionary.
#         usgs_to_link_id (Dict): Mapping from USGS gauge IDs to model link IDs.
#         file_order (np.ndarray): The order of link IDs in the columns of the raw observation file.
#         data_file_to_archive (str): The actual path of the data file used in the run.
#     """
#     try:
#         obs_config = config['observations']
#         vis_config = config['visualization']
#         da_window_config = config['da_settings']['assimilation_window']
#         out_dir_csv = os.path.join(config['paths']['out_dir'], 'csv')
# 
#         plot_gauges = vis_config.get('plot_usgs', [])
#         if not plot_gauges:
#             logging.warning("No 'plot_usgs' found in config; skipping input data archiving.")
#             return
# 
#         # --- Robustly load the data file, whether it has a header or not ---
#         try: # Try loading as if it has a header (real data)
#             df_raw = pd.read_csv(data_file_to_archive, index_col=0, dtype=str, na_values=[''], encoding='utf-8').fillna("0.0")
#         except (IndexError, ValueError): # Fallback for headerless (synthetic data)
#             df_raw = pd.read_csv(data_file_to_archive, header=None, dtype=str, na_values=[''], encoding='utf-8').fillna("0.0")
# 
#         # --- Unify and filter by time window ---
#         if not pd.api.types.is_numeric_dtype(df_raw.index):
#             df_raw.index = df_raw.index.str.replace(r'-\d\d:\d\d', '', regex=True)
#             df_raw.index = pd.to_datetime(df_raw.index, errors='coerce')
#             df_raw = df_raw[~df_raw.index.isna()]
#             start_time = pd.to_datetime(da_window_config['start'])
#             end_time = pd.to_datetime(da_window_config['end'])
#             df_raw = df_raw.loc[start_time:end_time]
# 
#         # Since synthetic data is generated for the exact window, no time filtering is needed for it.
#         df_archive = pd.DataFrame(index=df_raw.index)
#         for gauge_id in plot_gauges:
#             link_id = usgs_to_link_id.get(str(gauge_id))
#             if link_id:
#                 col_idx = np.where(file_order == link_id)[0]
#                 if col_idx.size > 0:
#                     df_archive[f"gauge_{gauge_id}"] = pd.to_numeric(df_raw.iloc[:, col_idx[0]], errors='coerce')
#         
#         df_archive.index.name = "datetime"
#         archive_path = os.path.join(out_dir_csv, 'archived_input_observations.csv')
#         df_archive.to_csv(archive_path)
#         logging.info(f"Archived input observation data for plot gauges to {archive_path}")
# 
#     except Exception as e:
#         logging.error(f"Failed to archive input observation data. Error: {e}", exc_info=True)