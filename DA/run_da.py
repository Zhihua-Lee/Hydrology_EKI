import argparse
import logging
import os
import numpy as np
import pandas as pd
import shutil
from state_vector import StateVector
from scipy.io import FortranFile
from forecast_operator import ForecastOperator
from analysis_operator import AnalysisOperator
from kalman_update import KalmanUpdate
from data_handler import DataHandler
from utils import process_yaml as load_config, create_output_and_temp_dirs, save_da_step_outputs
from io_ifc import get_ids, load_usgs_mapping, load_and_process_observations, create_meas_sav
from hlm_runner import HLMRunner
from hpc_da_tasks import run_hpc_presimulation

def main(config_path):
    """
    Main runner for the sequential Data Assimilation framework.
    """
    # 1. Load Configuration and Set Up Environment
    try:
        config = load_config(config_path)
    except Exception as e:
        print(f"Error loading configuration from {config_path}: {e}")
        return

    # --- Path Management ---
    paths_config = config.get('paths', {})
    login_root = config.get('login_node_root')
    compute_root = config.get('compute_node_root')
    if not login_root or not compute_root:
        raise ValueError("'login_node_root' and 'compute_node_root' must be defined in config paths.")
    
    # # 【重要】为了向后兼容，我们将 project_root 设置为当前环境的根目录
    # # 在 run_da.py 中，当前环境是 login node
    # config['project_root'] = login_root
    # logging.info(f"Dynamically determined project root: {project_root}")
    
    config['logging'] = config.get('logging', {'debug_mode': False}) # Ensure logging section exists

    # setup_logging(config.get('logging', {})) # Logging can be properly set up later
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    paths_config = config.get('paths', {})
    # Create fresh output and temporary directories for the run.
    create_output_and_temp_dirs(paths_config)

    # --- Archive the configuration file for reproducibility ---
    out_dir = paths_config.get('out_dir')
    shutil.copy(config_path, os.path.join(out_dir, 'config_archive.j2'))
    logging.info(f"Archived config file to {out_dir}")

    np.random.seed(config.get('random_seed', 42))
    logging.info("Configuration loaded and directories created.")

    # 2. Initialization of DA Components
    logging.info("Initializing components...")
    
    
    # A. Determine dimensions and mappings from the model structure
    # This is needed to initialize the DataHandler and the state vectors.
    hlm_config = config.get('hlm_model', {})
    link_ids = get_ids(hlm_config)
    n_links = len(link_ids)
    logging.info(f"Model has {n_links} physical state variables (links).")
    
    # B. Create static model files, following the EKI project's workflow.
    # This order is critical for dependencies.
    usgs_to_link_id, _, output_file_link_id_order = load_usgs_mapping(config['observations'])
    # Create the filtered 'meas.sav' file in the temp directory.
    create_meas_sav(config, link_ids)
    # Initialize the HLMRunner. It will now find and use the pre-created 'meas.sav'.
    hlm_runner = HLMRunner(config)
    
    # B.2 (Optional) Generate synthetic data if in twin-experiment mode
    da_settings = config.get('da_settings', {})
    obs_file_path = config['observations']['meas_series'] # Default to real data

    if da_settings.get('using_simulated_data', False):
        logging.info("--- Twin experiment mode enabled: Generating synthetic observations ---")
        run_hpc_presimulation(config, hlm_runner)
        obs_file_path = da_settings['synthetic_obs_path'] # Override with synthetic data path
        logging.info("--- Synthetic data generation complete ---")

    # C. Load all observation data for the entire window.
    # The load function returns multiple items; we only need the time series for the handler.
    obs_config = config.get('observations', {})
    assimilation_usgs_ids = obs_config.get('real_time_usgs_gauges', [])
    # Ensure n_gauges is calculated correctly whether config provides a single string or a list
    if isinstance(assimilation_usgs_ids, str):
        assimilation_usgs_ids = [assimilation_usgs_ids]
    n_gauges = len(assimilation_usgs_ids)

    observation_timeseries, _, _, _ = load_and_process_observations(
        data_file_path=obs_file_path,
        obs_config=obs_config,
        da_window_config=config['da_settings']['assimilation_window'],
        file_order=output_file_link_id_order,
        usgs_to_link_id=usgs_to_link_id,
        sorted_link_ids=link_ids,
        using_simulated_data=da_settings.get('using_simulated_data', False)
    )

    # --- Archive the specific input observation data used in this run ---
    # archive_input_data(config, usgs_to_link_id, output_file_link_id_order, data_file_to_archive=obs_file_path)

    # D. Initialize remaining DA components, injecting the pre-loaded observation data.
    data_handler = DataHandler(config, hlm_runner.n_links, n_gauges, observation_timeseries)
    forecast_op = ForecastOperator(config, hlm_runner)
    analysis_op = AnalysisOperator(config, data_handler, hlm_runner)
    kalman_updater = KalmanUpdate(config)
    
    logging.info("All DA components initialized.")

    # 3. Initial Ensemble Generation
    logging.info("Generating initial ensemble...")
    num_ensembles = da_settings.get('num_ensembles', 100)
    
    # +++ NEW: Import the latent space creation function +++
    from latent import create_latent

    # B. Create a diverse ensemble for the parameter history (alpha_r)
    # At t=0, the parameter history has length 1. It will grow dynamically.
    # +++ NEW: Generate initial alpha ensemble in LATENT space +++
    # The parameter is applied at the division level. We use the hlm_runner's
    # division_to_link_map to get the correct number of divisions.
    # create_latent returns (n_active_params, n_divisions, n_ens), 3d means 3 dims
    initial_state_path = config['hlm_model']['initial_uini']
    logging.info(f"Loading warm-start initial discharge state from: {initial_state_path}")
    with open(initial_state_path, 'r') as f:
        lines = [line.strip() for line in f.readlines()]

    # --- Correctly parse the .uini global format ---
    # The .uini file specifies a single state line to be applied to ALL links.
    try:
        # The state is on the third line (index 2)
        state_values = np.array([float(v) for v in lines[2].split()])
        # The first value is the discharge
        initial_discharge_value = state_values[0]
        logging.info(f"Parsed global initial discharge value from .uini: {initial_discharge_value}")
    except (IndexError, ValueError) as e:
        raise ValueError(f"Could not parse global state from {initial_state_path}. Error: {e}")

    # Broadcast this single initial discharge value to all links in the model.
    base_discharge_state = np.full(n_links, initial_discharge_value)

    # --- V3 Algorithm: Stage 0 & 1 ---
    logging.info("--- V3 Stage 0: Initializing state at t=0 ---")
    # 1. Generate q_0 prior from .uini file by adding perturbations.
    # The StateVector will only store this discharge vector, not the full state matrix.
    perturbation_scale = 0.05
    perturbations = np.random.normal(loc=1.0, scale=0.1, size=(num_ensembles, n_links))
    initial_q_ensemble = np.maximum(0, base_discharge_state * perturbations)
    logging.info(f"Created initial physical state (discharge) ensemble with shape {initial_q_ensemble.shape}.")

    # 2. Simplified analysis for q_0 (prior is treated as posterior) and store it.
    analysis_q0_ensemble = initial_q_ensemble
    mean_q0_for_storage = np.mean(analysis_q0_ensemble, axis=0)
    # The history part is None because we only store the physical state.
    data_handler.store_analysis_state(0, StateVector(mean_q0_for_storage, None))
    logging.info("Stored mean of initial physical state for t=0 in DataHandler.")

    logging.info("--- V3 Stage 1: First forecast from t=0 to t=1 ---")
    # 1. Generate the first parameter ensemble to be estimated: alpha_1
    initial_latent_alpha_3d = create_latent(config['parameters'], hlm_runner.division_to_link_map, num_ensembles)
    initial_alpha1_ensemble = initial_latent_alpha_3d[0].T # Shape: (n_ens, n_divisions)
    logging.info(f"Created initial parameter ensemble for alpha_1 with shape {initial_alpha1_ensemble.shape}.")

    # 2. Assemble a temporary analysis ensemble at t=0 to kick off the first forecast.
    # The parameter history for each member contains only alpha_1.
    analysis_ensemble_t0 = [
        StateVector(physical_state=analysis_q0_ensemble[i], param_history=initial_alpha1_ensemble[i].reshape(1, -1))
        for i in range(num_ensembles)
    ]

    # 3. Run the first forecast to get X_{1|0}, which will be the input for the main loop at t=1.
    forecast_ensemble = forecast_op.run_forecast(analysis_ensemble_t0, t=0)
    logging.info("Completed first forecast step to generate state for t=1.")

    # 4. Main Sequential Loop (Time-stepping)
    da_window = da_settings.get('assimilation_window', {})
    start_time = da_window.get('start')
    end_time = da_window.get('end')
    time_index = pd.date_range(start=start_time, end=end_time, freq='H') # Assuming hourly steps
    num_time_steps = len(time_index)
    logging.info(f"Starting main loop from t=1 to t={num_time_steps-1}.")
    
    max_param_history = da_settings.get('max_param_history', 5)
    # --- V3 Algorithm: Stage 2 ---
    # The loop now starts at t=1, as t=0 is handled during initialization.
    for t in range(1, num_time_steps):
        # a. Analysis (Update) Step
        # The forecast_ensemble (X_{t|t-1}) is ready from the previous step (or initialization).
        logging.info(f"--- Analysis Step at t={t} ({time_index[t]}) ---")
        y_obs_window = data_handler.get_observations_for_window(t)
        logging.info(f"Augmented state vector length: {forecast_ensemble[0].full_vector.shape[0]}, Observation window vector length: {y_obs_window.shape[0]}")

        # The analysis operator H(X_t) is called within the update step.
        # This returns the analysis matrix X_a = X_{t|t}.
        analysis_matrix_X_a = kalman_updater.run_update_step(
            forecast_ensemble=forecast_ensemble,
            y_obs_window=y_obs_window,
            analysis_operator=analysis_op,
            t_current=t,
        )
        
        # Convert the raw analysis_matrix back to a list of StateVector objects (X_{t|t}).
        # At time t, the forecast from t-1 had a history of length min(t, max+1).
        # The analysis result will have the same history length.
        current_param_history_len = min(t, max_param_history + 1)
        analysis_ensemble = StateVector.reconstruct_ensemble_from_matrix(
            analysis_matrix_X_a,
            n_physical_states=n_links,
            n_param_history=current_param_history_len
        )
        
        # b. Store the new analysis state for future re-runs
        # We store the mean of the ensemble as the "best guess" historical state.
        mean_analysis_q = np.mean([vec.q for vec in analysis_ensemble], axis=0)
        mean_analysis_state = StateVector(mean_analysis_q, None)  # History part not needed for storage
        data_handler.store_analysis_state(t, mean_analysis_state)
        logging.info(f"Stored mean analysis state for t={t}.")

        # c. Save comprehensive DA step outputs
        logging.info(f"Saving DA outputs for step t={t}...")
        save_da_step_outputs(
            config=config, t=t,
            forecast_ensemble=forecast_ensemble,
            analysis_ensemble=analysis_ensemble,
            hlm_runner=hlm_runner
        )

        # d. Forecast Step to prepare for the *next* time step (t+1)
        forecast_ensemble = forecast_op.run_forecast(analysis_ensemble, t)

        logging.info(f"--- Completed main loop for t={t} ---")

    logging.info("Sequential Data Assimilation run finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Sequential Data Assimilation for HLM.")
    parser.add_argument(
        "--config", 
        default="config.j2",
        help="Path to the configuration file relative to the DA directory (e.g., config.j2)."
    )
    args = parser.parse_args()
    
    # Assumes the runner is executed from the project root (e.g., 2025_EKI)
    config_file_path = os.path.join('DA', args.config)
    main(config_file_path)