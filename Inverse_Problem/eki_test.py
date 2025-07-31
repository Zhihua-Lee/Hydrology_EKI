#!/usr/bin/python
import sys
import shutil
import numpy as np
import os, time
import argparse

from tqdm import tqdm
from utils import process_yaml, get_ids, get_subwatershed, load_and_process_observations
from io_ifc import create_meas_sav, create_test_initial_condition, create_prm_from_division_params, create_ensemble_gbl, create_batch_job_file, save_statistics_csv, save_particles
from eki import subsample_data, pert, EnKF_step
from latent import create_latent, transform_latent_to_physical
from run import run_test, generate_synthetic_data, generate_prm_files_for_ensemble
from ifc_usgs_fileorder import load_usgs_mapping

import pandas as pd

import visualize

def main(yaml_name, visualize_only=False):
    # --- 1. Configuration and Setup ---
    # Load experiment settings from the specified YAML file.
    test_dict = process_yaml(yaml_name)

    if not visualize_only:
        tmp_dir = test_dict['tmp_dir']
        out_dir = test_dict['out_dir']
        step_num = test_dict['steps']
        ens = test_dict['num_ensembles']

        # --- 2. Environment and File Setup ---
        print("\n--- Setting up experiment environment ---")
        # Clean the temporary directory and set up the output directory structure.
        print(f"Cleaning temporary directory: {tmp_dir}")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        os.makedirs(tmp_dir)
        os.makedirs(out_dir, exist_ok=True)
        
        # Copy essential configuration and data files for reproducibility.
        print(f"Copying essential files to output directory: {out_dir}")
        shutil.copyfile(yaml_name, os.path.join(out_dir, 'test_config.j2'))
        shutil.copy(test_dict['initial_uini'], out_dir)
        os.makedirs(os.path.join(out_dir, 'csv/'), exist_ok=True)
        os.makedirs(os.path.join(out_dir, 'npy/'), exist_ok=True)

        # --- 3. Model Structure Initialization and Static File Generation ---
        print("\n--- Initializing Model Structure and Static Files ---")
        # This stage identifies the model's physical structure (link IDs, watersheds)
        # and creates all necessary static input files for the simulation runs.

        # A. Identify model structure (link IDs and watershed divisions).
        usgs_to_link_id, link_to_usgs_id, output_file_link_id_order = load_usgs_mapping(test_dict)
        sorted_link_ids = get_ids(test_dict) # The authoritative source for sorted link IDs.
        
        if test_dict['watershed_csv'] is None:
            print("No watershed division file provided. Assuming a single global parameter set.")
            # The map should represent one division containing all links.
            # Its shape must be (n_divisions, n_links), which is (1, n_links).
            n_links = len(sorted_link_ids)
            division_to_link_map = np.ones((1, n_links)) # (n_divisions, n_links)
            link_to_division_map = {link_id: 0 for link_id in sorted_link_ids}

        else:
            print(f"Loading watershed divisions from: {test_dict['watershed_csv']}")
            # Pass sorted_link_ids to ensure consistency.
            division_to_link_map, link_to_division_map = get_subwatershed(test_dict, sorted_link_ids) # (n_divisions, n_links)
        # ANNOTATION: The 'division_to_link_map' matrix is a transformation operator that maps parameters
        # defined at the watershed division level to the individual link level.
        # - Shape: (number_of_divisions, number_of_links)
        # - Value: A '1' at matrix[i, j] signifies that link 'j' belongs to division 'i'.
        # - Usage: It's used via its transpose to broadcast coarse-grained division parameters
        #          to all their fine-grained child links.
        
        # B. Create all static model files immediately.
        print("Creating static model input files (.sav, .uini, .gbl, submit script)...")
        create_meas_sav(test_dict, sorted_link_ids)       # Sensor location file.
        create_test_initial_condition(test_dict, sorted_link_ids) # Initial conditions file.
        create_ensemble_gbl(test_dict, ens)               # Global configuration files for all ensemble members.
        create_batch_job_file(test_dict, tmp_dir)         # Job submission script.

        # --- 4. EKI Core Variable Preparation ---
        print("\n--- Preparing EKI Core Variables (X, y, R) ---")
        # This stage prepares all dynamic variables required to start the EKI main loop.

        # A. Prepare State Vector (X).
        print("Initializing X_post as the initial latent variables....")
        X_post = create_latent(test_dict, division_to_link_map, ens) # (n_latent_params, n_divisions, n_ens)

        # B. Prepare Observation Vector (y) and Error Covariance Diagonal (R).
        print("Preparing observation vector y and error covariance diagonal R...")
        # Handle observation data source.
        using_simulated_data = test_dict['using_simulated_data']
        print(f"\n--- Data Source Configuration ---")
        print(f"Using simulated data: {using_simulated_data}")
        print(f"Measurement data path: {test_dict['meas_series']}")
        if using_simulated_data:
            generate_synthetic_data(test_dict)
        else:
            print("Using real observation data. Skipping pre-simulation step.")
        
        # Now that the measurement data is confirmed to exist, copy it for reproducibility.
        shutil.copy(test_dict['meas_series'], out_dir)

        # Load and process observation data now that .sav file exists.
        print("Loading and processing observation data...")
        assimilation_data, plotting_data, _, col_idx_in_sav = load_and_process_observations(
            test_dict, output_file_link_id_order, usgs_to_link_id, sorted_link_ids
        )  # assimilation_data(flattened): (n_gauges * t_steps,);   plotting_data: (t_steps, n_gauges)
        print("Saving initial measurement statistics...")
        save_statistics_csv(test_dict, division_to_link_map, Y_mean=plotting_data, Y_std=None, X_mat=None, name='csv/' + "meas")

        # Finalize y and R.
        y = np.reshape(assimilation_data,(-1,1)) # Reshape observation vector as: (n_gauges * t_steps, 1)
        abs_meas_std = test_dict['abs_std_meas']
        rel_meas_std = test_dict['rel_std_meas']
        R = (rel_meas_std * y.reshape(-1))**2 + abs_meas_std**2  # (n_gauges * t_steps,)

        # --- 5. EKI Main Loop ---
        print("\n" + "="*60 + "\n" + "🚀  STARTING EKI PROCESS" + "\n" + "="*60 + "\n")

        for i in tqdm(range(step_num)):
            # --- Prior Step ---
            X_prior = pert(X_post, test_dict, division_to_link_map)
            generate_prm_files_for_ensemble(
                test_dict, X_prior, ens, division_to_link_map.shape[0], link_to_division_map
            )
            Y_prior, Y_plot_prior, Y_plot_mean, Y_plot_std, _, _  = run_test(ens, X_prior, tmp_dir, col_idx_in_sav)
            save_particles(test_dict, division_to_link_map, X_prior, Y_plot_prior, name='npy/' + str(i) + '_prior')
            save_statistics_csv(test_dict, division_to_link_map, Y_plot_mean, Y_plot_std, X_prior, name='csv/' + str(i) + "_prior")
            
            # --- Posterior Step (Assimilation) ---
            # Reshape the 3D parameter array to 2D for the EnKF step
            n_params, n_div, n_ens_members = X_prior.shape
            X_prior_flat = X_prior.reshape(n_params * n_div, n_ens_members)
            # Run the assimilation step
            X_post_flat = EnKF_step(y, X_prior_flat, Y_prior, R, test_dict, i)
            # Reshape the updated 2D parameters back to 3D for the next iteration
            X_post = X_post_flat.reshape(n_params, n_div, n_ens_members)
            # Generate PRM files for the updated ensemble
            generate_prm_files_for_ensemble(
                test_dict, X_post, ens, division_to_link_map.shape[0], link_to_division_map
            )
            Y_post, Y_plot_post, Y_plot_mean, Y_plot_std, _, _ = run_test(ens, X_post, tmp_dir, col_idx_in_sav)
            save_particles(test_dict, division_to_link_map, X_post, Y_plot_post, name='npy/' + str(i) + '_post')
            save_statistics_csv(test_dict, division_to_link_map, Y_plot_mean, Y_plot_std, X_post, name='csv/' + str(i) + "_post")
    else:
        # --- Precondition Check for visualize-only mode ---
        out_dir = test_dict['out_dir']
        required_dir = os.path.join(out_dir, 'npy')
        # Use a file that is reliably created at the start as a sentinel.
        sentinel_file = os.path.join(out_dir, 'csv', 'meas_mean.csv')
        
        if not os.path.isdir(required_dir) or not os.path.exists(sentinel_file):
            print("\n" + "="*60)
            print("❌ ERROR: Cannot run in --visualize-only mode.")
            print(f"Required output files not found in: {out_dir}")
            print("Please run the script without the --visualize-only or -v flag first to generate the necessary data.")
            print("="*60 + "\n")
            sys.exit(1)

        print("\n--- Skipping EKI run. Proceeding directly to visualization. ---")

    # --- 6. Visualization ---
    # After all EKI steps are completed, generate visualizations of the results.
    print("\n--- Generating Visualizations ---")
    visualize.main_visualization(test_dict)
    print("\n--- EKI Workflow Complete ---")
       
    
if __name__ == "__main__":
    # The script is executed by passing the path to the YAML configuration file.
    parser = argparse.ArgumentParser(description="Run Ensemble Kalman Inversion (EKI) test.")
    parser.add_argument("yaml_name", type=str, help="Path to the YAML configuration file.")
    parser.add_argument("-v", "--visualize-only", action="store_true", help="If set, skips the EKI run and only generates visualizations from existing output.")
    
    args = parser.parse_args()
    main(args.yaml_name, args.visualize_only)