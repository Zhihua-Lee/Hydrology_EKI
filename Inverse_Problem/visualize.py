#!/usr/bin/env python
# coding: utf-8
"""
visualize.py

This module generates visualizations for the EKI pipeline. It creates:
  1. Hydrograph animations (both frames and final GIF) for each gauge,
  2. Parameter evolution plots (ensemble trajectories and mean-std plots), and
  3. Event statistics plots (e.g., peak, mean, and standard deviation).

The output is organized under the "visualization" folder, with subfolders for "prior" and "post"
assimilation results. The gauge IDs (from USGS) are used as station names in the plots.

This file can be imported and its main_visualization() function called at the end of your eki_test.py
main function, so that visualizations are automatically generated after the EKI algorithm completes.
"""

import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import hydroeval as he
import copy
import struct # <-- Added import

import geopandas as gpd
from utils import get_subwatershed, get_ids

from ifc_usgs_fileorder import load_usgs_mapping_from_path
from io_ifc import get_rainfall_for_lid_from_config
from eki import find_events, find_metric_values

# Global matplotlib settings
plt.rcParams['font.size'] = 12
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.family'] = 'STIXGeneral'

# ------------------ Utility: Clear and Create Directory ------------------
def clear_and_create_dir(dir_path):
    """If the directory exists, remove it entirely and then create a new one."""
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)

# ===================== Evaluation Metric Functions =====================
def kge_metric(obs, sim):
    """Calculate the NSE metric (used here as a proxy for KGE) between observed and simulated data."""
    return he.evaluator(he.nse, sim[obs > 0], obs[obs > 0])

def peak_relative_diff(obs, sim):
    """Calculate the relative difference between the peak values of simulated and observed data."""
    return (np.max(sim) - np.max(obs)) / np.max(obs)

def peak_timing_diff(obs, sim):
    """Calculate the difference in index (timing) where simulated and observed data reach their peaks."""
    return np.argmax(sim) - np.argmax(obs)

# ===================== Animation Frame Drawing Function =====================
def draw_animation_frame(iter_idx, ensemble_sim, station_idx, time_axis,
                         measured_data, station_label, rain_df,
                         using_simulated_data, cr_ref_for_scaling): # <-- Added flags/values
    """
    Draw a single animation frame, scaling rainfall if using simulated data.

    Parameters:
      iter_idx: Current assimilation iteration (formatted as two-digit number).
      ensemble_sim: Simulation ensemble data of shape (ensemble_size, time_steps, num_stations).
      station_idx: Index of the station (column in measured_data) to plot.
      time_axis: DatetimeIndex for the x-axis.
      measured_data: Observed data array of shape (time_steps, num_stations).
      station_label: Station name (gauge ID) to display in the title.
      rain_df (pd.DataFrame): Rainfall data for the station with DatetimeIndex.
      using_simulated_data (bool): Flag indicating if simulated data scenario is active.
      Cr_ref (float or None): The reference Cr value used in simulated experiments.
    
    """
    fig = plt.gcf() # Get current figure
    ax = plt.gca() # Get current axes

    # --- Hydrograph plotting (Existing logic) ---
    station_ensemble = ensemble_sim[:, :, station_idx]
    median_sim = np.median(station_ensemble, axis=0)
    obs_series = measured_data[:, station_idx]

    ax.plot(time_axis, median_sim, 'b-', label='Particle median')
    ax.fill_between(time_axis,
                     np.percentile(station_ensemble, 5, axis=0),
                     np.percentile(station_ensemble, 95, axis=0),
                     color='blue', alpha=0.3)
    # Use a conditional label based on the data source
    legend_text = 'Simulated Observation' if using_simulated_data else 'Observed'
    ax.plot(time_axis, obs_series, 'k--', label=legend_text)

    # --- Metric calculation (Existing logic) ---
    try:
        # Ensure obs_series and median_sim are 1D arrays for metrics
        obs_1d = obs_series.flatten()
        sim_1d = median_sim.flatten()
        # Add basic checks for valid data before calculating metrics
        valid_indices = ~np.isnan(obs_1d) & ~np.isnan(sim_1d) & (obs_1d > 0)
        if np.any(valid_indices):
             kge_val = he.evaluator(he.nse, sim_1d[valid_indices], obs_1d[valid_indices])[0] # NSE used as proxy
             # Ensure peaks are calculated only on valid data
             obs_peak_val = np.max(obs_1d[valid_indices]) if np.any(obs_1d[valid_indices]) else np.nan
             sim_peak_val = np.max(sim_1d[valid_indices]) if np.any(sim_1d[valid_indices]) else np.nan

             if not (np.isnan(obs_peak_val) or np.isnan(sim_peak_val) or obs_peak_val == 0):
                 pr_diff = (sim_peak_val - obs_peak_val) / obs_peak_val
             else:
                 pr_diff = np.nan

             # Ensure argmax is applied correctly if needed, or adjust peak timing logic
             # Note: peak_timing_diff might need adjustment based on how time_axis relates to indices
             # This simple argmax assumes time_axis corresponds directly to array index
             pt_diff = np.argmax(sim_1d) - np.argmax(obs_1d) # Be cautious with this if NaNs exist or indices != time steps

             print(f"Iteration {iter_idx:02d}, Gauge {station_label}: KGE={kge_val:.3f}, PeakRelDiff={pr_diff:.3f}, PeakTimeDiff={pt_diff}")
        else:
            print(f"Iteration {iter_idx:02d}, Gauge {station_label}: Insufficient valid data for metrics.")

    except Exception as e:
        print(f"Iteration {iter_idx:02d}, Gauge {station_label}: Metric calculation failed. Error: {e}")


    # --- Plot setup (Existing logic) ---
    ax.set_title(f'EKI iteration {iter_idx:02d} - Gauge {station_label}')
    ax.set_xlabel('Time')
    ax.set_ylabel('Discharge (m$^3$/s)') # Use LaTeX for m^3/s
    ax.legend(loc='upper left') # Adjust legend location if needed
    ax.grid(True)
    # ax.tick_params(axis='x', rotation=45) # Keep or remove rotation as desired

    # --- ADDED: Rainfall Plotting with conditional scaling ---
    if not rain_df.empty:
        ax2 = ax.twinx()

        # Determine rainfall scale factor
        scale_factor = 1.0
        if using_simulated_data and cr_ref_for_scaling  is not None:
             scale_factor = cr_ref_for_scaling 
             # print(f"Info: Scaling rainfall by cr_ref_for_scaling  = {cr_ref_for_scaling } for plotting.") # Optional info message

        # Apply scaling
        scaled_rainfall = rain_df["Rainfall"] * scale_factor
        flipped_rainfall = -scaled_rainfall # Flip after scaling

        # Calculate dynamic ylim based on the potentially scaled rainfall
        min_flipped_rainfall = flipped_rainfall.min()
        if pd.notna(min_flipped_rainfall):
            ylim_bottom = min(min_flipped_rainfall * 1.2, min_flipped_rainfall - 0.5) # Buffer below lowest bar
        else:
            ylim_bottom = -1 # Default if no rainfall data or all zero after scaling

        # Plot the potentially scaled and flipped rainfall
        # --- OPTIMIZED Bar Plotting ---
        ax2.bar(rain_df.index,           # X values (timestamps)
                flipped_rainfall,        # Y values (flipped, scaled rainfall)
                width=1/24,              # Width representing 1 hour
                alpha=0.7,               # <--- Optional: Adjusted alpha for brighter color
                color='deepskyblue',     # <--- CHANGED: High-contrast sky blue
                label='Rainfall (Flipped)',# Label
                zorder=3                 # Ensure bars are drawn on top
               )
        ax2.set_ylabel('Rainfall (mm/h, flipped)')
        ax2.set_ylim(ylim_bottom, 0)

    else:
        print(f"Iteration {iter_idx:02d}, Gauge {station_label}: No rainfall data to plot.")

    # --- Final Layout (Existing logic) ---
    # plt.tight_layout() # Apply tight layout after all plotting is done
    fig.autofmt_xdate() # Better date formatting/rotation

# ===================== Data Loading Helper Function =====================
def load_ensemble(assimilation_phase, iter_idx, out_dir):
    """
    Load simulation ensemble data based on the assimilation phase and iteration index.

    Parameters:
      assimilation_phase: 'post' or 'prior'
      iter_idx: Current iteration index (for post assimilation, iter_idx==0 uses prior file)

    Returns:
      Numpy array of simulation ensemble data.
    """
    if assimilation_phase == 'post':
        if iter_idx == 0:
            file_path = out_dir + 'npy/0_prior_particles.npy'
        else:
            file_path = out_dir + f'npy/{iter_idx - 1}_post_particles.npy'
    elif assimilation_phase == 'prior':
        file_path = out_dir + f'npy/{iter_idx}_prior_particles.npy'
    else:
        raise ValueError("assimilation_phase must be 'post' or 'prior'")
    with open(file_path, 'rb') as f:
        return np.load(f)

# ===================== Hydrograph Animation Generation Function =====================
def generate_hydrograph_animation(num_iters, station_indices, station_names, plot_link_ids,
                                  measured_data, time_axis, start_time_str, end_time_str, rain_dir,
                                  using_simulated_data, cr_ref_vec, # <-- is a vector or None
                                  link_to_division_map, # <-- new map parameter
                                  assimilation_phase, visual_output_dir, out_dir):
    """
    Generate and save hydrograph animation GIFs. Includes rainfall plotting.

    Parameters:
        num_iters: Number of assimilation iterations (post: num_iters+1; prior: num_iters)
        station_indices: List of column indices in the observed data for the stations to plot.
        station_names: List of corresponding station names (gauge IDs).
        measured_data: Observed data array of shape (time_steps, num_stations).
        time_axis: DatetimeIndex for the x-axis.
        assimilation_phase: 'post' or 'prior'
        visual_output_dir: Top-level output directory (e.g., 'visualization')
        plot_link_ids (list): List of Link IDs corresponding to station_indices/station_names.
        start_time_str (str): Start time for fetching rainfall.
        end_time_str (str): End time for fetching rainfall.
        rain_dir (str): Base directory for rainfall data.
        using_simulated_data (bool): Flag indicating if simulated data scenario is active.
        Cr_ref (float or None): The reference Cr value used in simulated experiments.
    """
    hydrograph_frames_dir = os.path.join(visual_output_dir, assimilation_phase, "hydrograph", "frames")
    hydrograph_anim_dir = os.path.join(visual_output_dir, assimilation_phase, "hydrograph", "animation")
    clear_and_create_dir(hydrograph_frames_dir)
    clear_and_create_dir(hydrograph_anim_dir)

    iter_range = range(num_iters + 1) if assimilation_phase == 'post' else range(num_iters)

    # Loop over the given station indices, names, and link IDs
    for i, station_idx in enumerate(station_indices):
        station_label = station_names[i]
        target_lid = plot_link_ids[i] # Get the corresponding Link ID

        print(f"\nProcessing Gauge: {station_label} (LID: {target_lid}) for {assimilation_phase} phase...")

        # --- ADDED: Load rainfall data ONCE per station ---
        print(f"Loading rainfall data for LID {target_lid} from {start_time_str} to {end_time_str}...")
        rain_df = get_rainfall_for_lid_from_config(target_lid, start_time_str, end_time_str, rain_dir)
        if rain_df.empty:
            print(f"Warning: No rainfall data found for LID {target_lid} in the specified period.")
        else:
            print(f"Rainfall data loaded: {len(rain_df)} records.")

        cr_ref_for_station = None # Default to None
        if using_simulated_data and cr_ref_vec is not None and link_to_division_map is not None:
            # --- New Logic: Find the specific Cr_ref for this station ---
            division_id = link_to_division_map.get(target_lid)
            if division_id is not None:
                cr_ref_for_station = cr_ref_vec[division_id] # This is now a scalar
            else:
                print(f"Warning: Could not find division for link_id {target_lid}. Using first Cr_ref value.")
                cr_ref_for_station = cr_ref_vec[0]

        frame_imgs = []
        for iter_idx in iter_range:
            # Create a new figure for each frame to avoid state issues
            fig, ax = plt.subplots(figsize=(12, 6)) # Adjust figsize as needed

            ensemble_sim = load_ensemble(assimilation_phase, iter_idx, out_dir)

            # Dynamic Y-limit based on observed data peak for this station
            max_obs_val = np.nanmax(measured_data[:, station_idx])
            y_limit_top = max_obs_val * 1.5 if pd.notna(max_obs_val) and max_obs_val > 0 else 1.0 # Set upper limit with buffer, minimum of 1.0
            y_limits = [0, y_limit_top]

            # Draw the frame (hydrograph + potentially scaled rainfall)
            # Note: draw_animation_frame now uses gcf/gca, so we don't pass ax explicitly
            draw_animation_frame(iter_idx, ensemble_sim, station_idx, time_axis, measured_data,
                                 station_label=station_label, rain_df=rain_df,
                                 using_simulated_data=using_simulated_data, 
                                 cr_ref_for_scaling=cr_ref_for_station) # <-- Pass specific scalar

            ax.set_ylim(*y_limits) # Apply hydrograph y-limits

            plt.tight_layout() # Apply layout adjustments

            # File naming and saving (Existing logic)
            frame_filepath = os.path.join(hydrograph_frames_dir, f"iter_{iter_idx:02d}_gauge_{station_label}_hydrograph.png")
            plt.savefig(frame_filepath)
            frame_imgs.append(Image.open(frame_filepath))
            plt.close(fig) # Close the figure to free memory

        # Create GIF (Existing logic)
        if frame_imgs:
            gif_filepath = os.path.join(hydrograph_anim_dir, f"gauge_{station_label}_hydrograph_animation.gif")
            frame_imgs[0].save(gif_filepath, save_all=True, append_images=frame_imgs[1:], duration=1000, loop=0)
            print(f"Animation saved to {gif_filepath}")
        else:
             print(f"No frames generated for Gauge {station_label}, GIF not created.")


# ===================== Parameter Evolution Plot Function =====================
def plot_parameter_evolution(param_array, active_param_indices, param_labels, param_ranges, 
                             plot_station_indices, plot_link_ids, # <-- new parameters
                             assimilation_phase, visual_output_dir, iter_range, 
                             cr_ref_vec=None, link_to_division_map=None): # <-- new parameters
    """
    Plot parameter evolution graphs and save ensemble and mean-std plots.

    Parameters:
      param_array: Numpy array of parameter ensemble data with shape (num_iters, num_active_params, num_stations, particle_dim)
      active_param_indices: List of indices for active parameters (used to index param_labels and param_ranges)
      param_labels: List of parameter names
      param_ranges: List of parameter value ranges (each as [min, max])
      assimilation_phase: 'post' or 'prior'
      visual_output_dir: Top-level output directory (e.g., 'visualization')
      iter_range: Array of iteration indices
      cr_ref: optional reference value to plot as a horizontal line.
    """
    param_ensemble_dir = os.path.join(visual_output_dir, assimilation_phase, "parameter", "ensemble")
    param_mean_std_dir = os.path.join(visual_output_dir, assimilation_phase, "parameter", "mean_std")
    clear_and_create_dir(param_ensemble_dir)
    clear_and_create_dir(param_mean_std_dir)
    
    num_active_params = param_array.shape[1]

    # Iterate through each ACTIVE parameter (e.g., just Cr in this case)
    for idx_active, orig_idx in enumerate(active_param_indices):
        # Iterate through the STATIONS that are selected for plotting
        for i, station_idx in enumerate(plot_station_indices):
            target_lid = plot_link_ids[i] # Get the link ID for the current station

            # --- CORRECT IMPLEMENTATION ---
            # 1. Find the division_id for the current link_id from the map
            if link_to_division_map is None:
                print(f"Warning: link_to_division_map is not available. Cannot plot parameter evolution.")
                continue
            
            division_id = link_to_division_map.get(target_lid)
            if division_id is None:
                print(f"Warning: Cannot find division for LID {target_lid}. Skipping parameter plot for this station.")
                continue

            # 2. Use this division_id to index the parameter array
            param_data_for_division = param_array[:, idx_active, division_id, :]

            # 3. Find the specific Cr_ref for this station's division
            cr_ref_for_plot = None
            if cr_ref_vec is not None:
                cr_ref_for_plot = cr_ref_vec[division_id]
            # --- END OF CORRECTION ---

            # --- Ensemble Plot ---
            plt.figure(figsize=(10, 6))
            plt.plot(iter_range, param_data_for_division) # Use the correctly indexed data
            plt.title(f'Parameter Ensemble Trajectories - {param_labels[orig_idx]} (LID: {target_lid})')
            plt.ylabel(f"Value of {param_labels[orig_idx]}")
            plt.xlabel('EKI Iterations')
            plt.ylim(*param_ranges[orig_idx])
            if cr_ref_for_plot is not None:
                plt.axhline(cr_ref_for_plot, color='red', linestyle='--', label=f'Cr_ref = {cr_ref_for_plot:.2f}')
                plt.legend()
            out_path = os.path.join(param_ensemble_dir, f"parameter_{orig_idx}_LID_{target_lid}_ensemble.png")
            plt.savefig(out_path)
            plt.close()
            
            # --- Mean-Std Plot ---
            plt.figure(figsize=(10, 6))
            param_mean = np.mean(param_data_for_division, axis=1)
            param_std = np.std(param_data_for_division, axis=1)
            plt.plot(iter_range, param_mean, 'k-', lw=2, label='Mean')
            plt.fill_between(iter_range, param_mean - param_std, param_mean + param_std,
                             color='gray', alpha=0.3, label='Mean ± Std')
            if cr_ref_for_plot is not None:
                plt.axhline(cr_ref_for_plot, color='red', linestyle='--', label=f'Cr_ref = {cr_ref_for_plot:.2f}')
            plt.title(f'Parameter Mean Evolution - {param_labels[orig_idx]} (LID: {target_lid})')
            plt.ylabel(f"Value of {param_labels[orig_idx]}")
            plt.xlabel('EKI Iterations')
            plt.ylim(*param_ranges[orig_idx])
            plt.legend()
            out_path = os.path.join(param_mean_std_dir, f"parameter_{orig_idx}_LID_{target_lid}_mean_std.png")
            plt.savefig(out_path)
            plt.close()
    
    print("Finished plotting parameter evolution.")


# ===================== Event Statistics Plot Function =====================
def plot_event_statistics(assimilation_phase, visual_output_dir, out_dir, test_dict, 
                          plot_station_indices, plot_station_names, plot_link_ids):
    """
    Tracks the evolution of simulated event metrics over EKI iterations and plots
    their convergence towards the observed metrics in a single consolidated figure per station.
    """
    if assimilation_phase != 'post':
        return

    print("\n--- Generating Consolidated Event Statistics EVOLUTION Plots ---")
    event_stats_dir = os.path.join(visual_output_dir, assimilation_phase, "event_statistics")
    clear_and_create_dir(event_stats_dir)

    # --- Steps 1-4: Load data and find events (same as before) ---
    observed_data = np.genfromtxt(os.path.join(out_dir, "csv", "meas_mean.csv"), delimiter=',', skip_header=1)
    num_steps = test_dict['steps']
    iter_range = np.arange(0, num_steps)
    event_params = test_dict.get('event_finding', {})
    min_dist = event_params.get('min_dist', 24)
    min_thresh_pct = event_params.get('min_thresh_pct', 25)
    min_length = event_params.get('min_length', 72)

    # --- Loop through each station to generate ONE figure per station ---
    for i, station_idx in enumerate(plot_station_indices):
        station_name = plot_station_names[i]
        target_lid = plot_link_ids[i]
        
        obs_series = observed_data[:, station_idx]
        min_thresh_val = np.percentile(obs_series[obs_series > 0], min_thresh_pct) if np.any(obs_series > 0) else 0
        event_indices_list, _ = find_events(obs_series, min_dist, min_thresh_val, min_length)

        if not event_indices_list:
            print(f"No events found for station {station_name}. Skipping.")
            continue
            
        print(f"Processing event evolution for Station: {station_name} (LID: {target_lid})")

        # --- Calculate observed metrics (target lines) ---
        observed_metrics = {}
        for event_num, event_indices in enumerate(event_indices_list):
            _, y_max, y_mean, _, _, _, y_std, y_mean_time, y_std_time = find_metric_values([event_indices], [obs_series[event_indices]])
            observed_metrics[event_num] = {
                'Peak': y_max[0][0], 'Mean': y_mean[0][0], 'Std_Dev': y_std[0][0],
                'Timing_Mean': y_mean_time[0][0], 'Timing_Std_Dev': y_std_time[0][0]
            }

        # --- Step 5: Gather simulated metrics evolution from all iterations (same as before) ---
        metric_names = ['Peak', 'Mean', 'Std_Dev', 'Timing_Mean', 'Timing_Std_Dev']
        evolution_data = {
            metric: {event_num: [] for event_num in range(len(event_indices_list))} 
            for metric in metric_names
        }

        for iter_idx in iter_range:
            sim_particles_file = os.path.join(out_dir, "npy", f"{iter_idx}_post_particles.npy")
            if not os.path.exists(sim_particles_file):
                print(f"Warning: Particles file not found for iter {iter_idx}. Stopping.")
                break
            
            simulated_particles = np.load(sim_particles_file)
            station_particles = simulated_particles[:, :, station_idx]
            
            for event_num, event_indices in enumerate(event_indices_list):
                event_ensemble_data = station_particles[:, event_indices]
                evolution_data['Peak'][event_num].append(np.max(event_ensemble_data, axis=1))
                evolution_data['Mean'][event_num].append(np.mean(event_ensemble_data, axis=1))
                evolution_data['Std_Dev'][event_num].append(np.std(event_ensemble_data, axis=1))
                
                ens_size = station_particles.shape[0]
                timing_mean_ensemble, timing_std_ensemble = (np.zeros(ens_size), np.zeros(ens_size))
                for particle_idx in range(ens_size):
                    vals = event_ensemble_data[particle_idx, :]
                    if np.sum(vals) > 0:
                        mean_t = np.sum(event_indices * vals) / np.sum(vals)
                        var_t = np.sum(vals * (event_indices - mean_t)**2) / np.sum(vals)
                        timing_mean_ensemble[particle_idx] = mean_t
                        timing_std_ensemble[particle_idx] = np.sqrt(var_t)
                    else:
                        timing_mean_ensemble[particle_idx], timing_std_ensemble[particle_idx] = (np.nan, np.nan)
                
                evolution_data['Timing_Mean'][event_num].append(timing_mean_ensemble)
                evolution_data['Timing_Std_Dev'][event_num].append(timing_std_ensemble)

        # --- Step 6: Create ONE figure with 5 subplots for the station ---
        fig, axes = plt.subplots(len(metric_names), 1, figsize=(12, 18), sharex=True)
        fig.suptitle(f'Event Metrics Evolution for Station {station_name} (LID: {target_lid})', fontsize=16)
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(event_indices_list)))

        for i, metric_name in enumerate(metric_names):
            ax = axes[i]
            for event_num in range(len(event_indices_list)):
                # Stack the list of ensemble arrays into a single 2D array
                metric_evolution_array = np.vstack(evolution_data[metric_name][event_num])
                
                mean_evolution = np.nanmean(metric_evolution_array, axis=1)
                std_evolution = np.nanstd(metric_evolution_array, axis=1)
                
                # Plot mean evolution for this event
                ax.plot(iter_range, mean_evolution, color=colors[event_num], label=f'Event {event_num+1} Sim Mean')
                # Plot uncertainty band for this event
                ax.fill_between(iter_range, mean_evolution - std_evolution, mean_evolution + std_evolution,
                                color=colors[event_num], alpha=0.15)
                
                # Plot the observed target value for this event
                target_value = observed_metrics[event_num][metric_name]
                ax.axhline(target_value, color=colors[event_num], linestyle='--', 
                           label=f'Event {event_num+1} Obs ({target_value:.2f})')

            y_label = 'Discharge (m$^3$/s)' if 'Timing' not in metric_name else 'Time (hours)'
            ax.set_ylabel(y_label)
            ax.set_title(f'Evolution of {metric_name.replace("_", " ")}')
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            ax.grid(True)

        axes[-1].set_xlabel('EKI Iteration')
        fig.tight_layout(rect=[0, 0, 0.85, 0.96]) # Adjust layout to make space for suptitle and legend
        
        output_path = os.path.join(event_stats_dir, f"station_{station_name}_all_metrics_evolution.png")
        plt.savefig(output_path)
        plt.close(fig)
        print(f"  - Saved consolidated evolution plot for Station {station_name}")


def generate_cr_map(assimilation_phase, visual_output_dir, out_dir, test_dict, cr_ref_vec=None):
    """
    Generates and saves a map of Cr values or Cr error, with styling
    and annotations inspired by the reference notebook.

    If using_simulated_data is true and cr_ref_vec is provided, it plots the 
    categorized percentage difference. Otherwise, it plots the absolute calibrated Cr values.
    """
    # --- 1. Setup paths and directories ---
    maps_dir = os.path.join(visual_output_dir, assimilation_phase, "maps")
    clear_and_create_dir(maps_dir)

    shapefile_path = test_dict.get('shapefile_path')
    if not shapefile_path or not os.path.exists(shapefile_path):
        print(f"Warning: Shapefile not found at {shapefile_path}. Skipping map generation.")
        return

    # --- 2. Load final parameter results ---
    num_steps = test_dict['steps']
    last_iter_idx = num_steps - 1
    
    if assimilation_phase != 'post' or last_iter_idx < 0:
        print(f"Info: Cr map is only generated for the final 'post' assimilation phase. Skipping for '{assimilation_phase}'.")
        return

    param_file = os.path.join(out_dir, 'csv', f"{last_iter_idx}_post_params_mean.csv")
    if not os.path.exists(param_file):
        print(f"Warning: Final parameter file not found: {param_file}. Skipping map generation.")
        return

    cr_sparse = pd.read_csv(param_file, header=None).to_numpy()
    cr_sparse_values = cr_sparse.flatten()
    
    # --- 3. Expand sparse parameters to all links and create a DataFrame ---
    model_link_ids = get_ids(test_dict)
    sparse_parent, link_to_division_map = get_subwatershed(test_dict, model_link_ids)
    
    cr_full_links = sparse_parent.T @ cr_sparse_values
    
    cr_df = pd.DataFrame({
        'link_id': model_link_ids,
        'Cr': cr_full_links,
        'division_id': [link_to_division_map.get(lid) for lid in model_link_ids]
    })

    # --- 4. Load shapefile and merge data ---
    gdf = gpd.read_file(shapefile_path)
    gdf.rename(columns={'LINKNO': 'link_id'}, inplace=True)
    gdf['link_id'] = gdf['link_id'].astype(int)
    gdf_merged = gdf.merge(cr_df, on='link_id', how='left')
    gdf_to_plot = gdf_merged[gdf_merged['division_id'].notna()].copy()

    # --- 5. Determine plotting mode and prepare data ---
    is_simulated = test_dict.get('using_simulated_data', False) and cr_ref_vec is not None
    
    # --- NEW: Configuration from notebook style ---
    PERCENTAGE_THRESHOLDS = [0.1, 1, 5, 10, 20, 50, 100]
    ANNOTATION_FONT_SIZE = 6
    
    plot_col = 'Cr'
    plot_title = f'Final Calibrated Cr Distribution ({assimilation_phase})'
    cbar_label = 'Calibrated Cr Value'

    if is_simulated:
        # --- NEW: Categorization logic from notebook ---
        # Map reference Cr values to each division
        ref_cr_map = {div_id: cr_ref_vec[div_id] for div_id in range(len(cr_ref_vec))}
        gdf_to_plot['Cr_ref'] = gdf_to_plot['division_id'].map(ref_cr_map)
        
        # Calculate percentage difference
        gdf_to_plot['percentage_diff'] = ((gdf_to_plot['Cr'] - gdf_to_plot['Cr_ref']).abs() / gdf_to_plot['Cr_ref']) * 100
        
        # Create bins and labels for categorization
        bins = [0] + PERCENTAGE_THRESHOLDS + [float('inf')]
        labels = [f'≤ {PERCENTAGE_THRESHOLDS[0]}%'] + \
                 [f'{PERCENTAGE_THRESHOLDS[i]}% - {PERCENTAGE_THRESHOLDS[i+1]}%' for i in range(len(PERCENTAGE_THRESHOLDS)-1)] + \
                 [f'> {PERCENTAGE_THRESHOLDS[-1]}%']
        
        gdf_to_plot['diff_category'] = pd.cut(
            gdf_to_plot['percentage_diff'],
            bins=bins, labels=labels, right=True, include_lowest=True, ordered=True
        )
        
        plot_col = 'diff_category'
        plot_title = f'Final Cr Parameter Error ({assimilation_phase})'
        cbar_label = f'% Difference from Reference Cr'
    
    # --- 6. Plotting ---
    # --- MODIFIED: Figure size and plot logic ---
    fig, ax = plt.subplots(1, 1, figsize=(12, 8)) # Unified figure size

    if is_simulated:
        # Categorical plot for error map
        active_categories = gdf_to_plot['diff_category'].cat.categories
        cmap = plt.get_cmap('Reds', len(active_categories))
        gdf_to_plot.plot(
            ax=ax,
            column=plot_col,
            cmap=cmap,
            legend=True,
            legend_kwds={'title': cbar_label, 'loc': 'upper left', 'bbox_to_anchor': (1, 1)},
            categorical=True,
            missing_kwds={"color": "lightgrey", "label": "No Data"}
        )
    else:
        # Continuous color plot for absolute values
        gdf_to_plot.plot(column=plot_col, ax=ax, legend=True,
                         legend_kwds={'label': cbar_label, 'orientation': "horizontal"})

    # --- NEW: Add annotations ---
    if 'division_id' in gdf_to_plot.columns:
        # Create a dissolved GeoDataFrame to find a representative point for each division
        dissolved = gdf_to_plot.dissolve(by='division_id', aggfunc={'Cr': 'first'})
        dissolved['point'] = dissolved.geometry.representative_point()
        
        for _, row in dissolved.iterrows():
            if pd.notna(row['Cr']) and row['point']:
                ax.text(
                    row['point'].x, row['point'].y, f"{row['Cr']:.2f}", # Format to 2 decimal places
                    fontsize=ANNOTATION_FONT_SIZE, ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.6, ec='none')
                )
    
    ax.set_title(plot_title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_axis_off() # Hide axes for a cleaner map look
    plt.tight_layout()

    # --- 7. Save the figure ---
    output_path = os.path.join(maps_dir, f"final_cr_map_{assimilation_phase}.png")
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved Cr map to {output_path}")


# ===================== Main Visualization Function =====================
def main_visualization(test_dict):
    """
    Main function to generate visualizations after the EKI algorithm finishes.
    This function can be embedded at the end of the eki_test.py main function.
    """
    visual_output_dir = test_dict['out_dir'] + "visualization"    
    start_time_str = test_dict["time_start"]
    end_time_str = test_dict["time_end"]
    # Ensure time_axis aligns with potential hourly rainfall data
    time_axis = pd.date_range(start=start_time_str, end=end_time_str, freq='H')
    num_assimilation_steps = test_dict["steps"]
    using_simulated_data = test_dict['using_simulated_data']
    rain_dir = test_dict['rain_dir'] # <-- Get rainfall directory path
    # Cr_ref = test_dict.get('Cr_ref', None) # Use .get for safety if key might be missing
    # --- New logic for handling Cr_ref and creating cr_ref_vec ---
    cr_ref_vec = None
    # --- New: Get the link-to-division map ---
    link_to_division_map = None

    model_link_ids_temp = get_ids(test_dict)
    if test_dict['watershed_csv']:
        sparse_parent_temp, link_to_division_map = get_subwatershed(test_dict, model_link_ids_temp)
        num_divisions = sparse_parent_temp.shape[0]
    else: # Handle no-watershed case
        num_divisions = 1 
        link_to_division_map = {link_id: 0 for link_id in model_link_ids_temp}
    
    if using_simulated_data:
        cr_ref_config = test_dict.get('Cr_ref')
        if cr_ref_config is not None:
            if isinstance(cr_ref_config, (int, float)):
                # If Cr_ref is a single value, expand it to a vector
                cr_ref_vec = np.full(num_divisions, float(cr_ref_config))
                print(f"Info: Expanded single Cr_ref {cr_ref_config} to a vector of size {num_divisions}.")
            elif isinstance(cr_ref_config, list):
                # If it's a list, check if its size matches the number of divisions
                if len(cr_ref_config) == num_divisions:
                    cr_ref_vec = np.array(cr_ref_config)
                    print(f"Info: Using provided Cr_ref vector of size {len(cr_ref_vec)}.")
                else:
                    raise ValueError(f"Error: The provided Cr_ref vector has size {len(cr_ref_config)}, but the number of subwatershed divisions is {num_divisions}.")
            else:
                 raise TypeError(f"Error: Cr_ref in config must be a number or a list, but got {type(cr_ref_config)}.")
    
    
    # max_station_count = 5  # Default value if desired_usgs_ids not found
    
    # Retrieve desired gauge IDs from configuration; ensure it's a list.
    desired_usgs_ids = test_dict["plot_usgs"]
    # if isinstance(desired_usgs_ids, str):
    #     desired_usgs_ids = [desired_usgs_ids]
        
    # Load USGS mapping using the provided CSV path (adjust relative path as needed)
    usgs_csv_path = test_dict["usgs_csv"]
    link_sav = test_dict["link_sav"]
    usgs_2_id, id_2_usgs, file_order = load_usgs_mapping_from_path(usgs_csv_path, link_sav)
    
    # Compute station indices, gauge names, AND link IDs based on desired_usgs_ids.
    plot_station_indices = []
    plot_station_names = []
    plot_link_ids = [] # <-- List to store corresponding Link IDs
    for usgs in desired_usgs_ids:
        link_id = usgs_2_id.get(usgs)
        if link_id is None:
            print(f"Warning: USGS ID {usgs} not found in mapping.")
            continue
        idx_arr = np.where(file_order == link_id)[0]
        if idx_arr.size > 0:
            plot_station_indices.append(idx_arr[0])
            plot_station_names.append(usgs)  # Use the gauge ID as station name.
            plot_link_ids.append(link_id) # <-- Store the link_id
        else:
            print(f"Warning: Link id {link_id} for USGS {usgs} not found in file_order.")
    # if not plot_station_indices:
    #     print("No desired station indices found, using default range.")
    #     plot_station_indices = list(range(max_station_count))
    #     plot_station_names = [str(i) for i in range(max_station_count)]
    
    observed_data = np.genfromtxt(test_dict['out_dir']+"csv/meas_mean.csv", delimiter=',', skip_header=1)
    observed_data_clean = observed_data.copy()
    observed_data[observed_data == 0] = np.nan

    # Parameter settings: assuming only one active parameter.
    param_labels = ["$Cr$"]
    param_ranges = [[0.00, 2.5]]
    active_param_indices = [0]

    assimilation_phases = ['prior', 'post']

    # -------------------- Post Assimilation (post) --------------------
    post_param_list = []
    for i in range(num_assimilation_steps + 1):
        if i > 0:
            file_path = test_dict['out_dir'] + f'npy/{i-1}_post_params_particles.npy'
        else:
            file_path = test_dict['out_dir'] + 'npy/0_prior_params_particles.npy'
        with open(file_path, 'rb') as f:
            post_param_list.append(np.load(f))
    post_param_array = np.stack(post_param_list, axis=0)
    post_param_array = post_param_array.reshape(num_assimilation_steps + 1,
                                                 len(active_param_indices),
                                                 -1,
                                                 post_param_array.shape[-1])
    iter_range_post = np.arange(0, num_assimilation_steps + 1)
    print("\n--- Generating Post-Assimilation Visualizations ---")
    generate_hydrograph_animation(num_assimilation_steps, plot_station_indices, plot_station_names,
                                  plot_link_ids,
                                  observed_data_clean, time_axis,
                                  start_time_str, end_time_str, rain_dir,
                                  using_simulated_data, cr_ref_vec, # <-- Pass flags/values
                                  link_to_division_map, # <-- Pass the map
                                  assimilation_phase='post', visual_output_dir=visual_output_dir, out_dir=test_dict['out_dir'])
    plot_parameter_evolution(post_param_array, active_param_indices, param_labels, param_ranges,
                             plot_station_indices, # Pass station indices
                             plot_link_ids,        # Pass corresponding link IDs
                             assimilation_phase='post', visual_output_dir=visual_output_dir, iter_range=iter_range_post, 
                             cr_ref_vec=cr_ref_vec,      # Pass the vector
                             link_to_division_map=link_to_division_map) # Pass the map
    plot_event_statistics('post', visual_output_dir, test_dict['out_dir'], test_dict,
                          plot_station_indices, plot_station_names, plot_link_ids)
    
    # --- CALL THE MAP GENERATION FUNCTION ---
    # Call the new map generation function here for the final post-assimilation result.
    generate_cr_map('post', visual_output_dir, test_dict['out_dir'], test_dict, cr_ref_vec)

    # -------------------- Prior Assimilation (prior) --------------------
    prior_param_list = []
    for i in range(num_assimilation_steps):
        file_path = test_dict['out_dir'] + f'npy/{i}_prior_params_particles.npy'
        with open(file_path, 'rb') as f:
            prior_param_list.append(np.load(f))
    prior_param_array = np.stack(prior_param_list, axis=0)
    prior_param_array = prior_param_array.reshape(num_assimilation_steps,
                                                   len(active_param_indices),
                                                   -1,
                                                   prior_param_array.shape[-1])
    iter_range_prior = np.arange(0, num_assimilation_steps)
    print("\n--- Generating Prior-Assimilation Visualizations ---")
    generate_hydrograph_animation(num_assimilation_steps, plot_station_indices, plot_station_names,
                                  plot_link_ids,
                                  observed_data_clean, time_axis,
                                  start_time_str, end_time_str, rain_dir,
                                  using_simulated_data, cr_ref_vec, # Pass the vector
                                  link_to_division_map, # <-- Pass the map
                                  assimilation_phase='prior', visual_output_dir=visual_output_dir, out_dir=test_dict['out_dir'])
    plot_parameter_evolution(prior_param_array, active_param_indices, param_labels, param_ranges,
                             plot_station_indices, # Pass station indices
                             plot_link_ids,        # Pass corresponding link IDs                             
                             assimilation_phase='prior', visual_output_dir=visual_output_dir, iter_range=iter_range_prior, 
                             cr_ref_vec=cr_ref_vec,      # Pass the vector
                             link_to_division_map=link_to_division_map) # Pass the map
    plot_event_statistics('prior', visual_output_dir, test_dict['out_dir'], test_dict,
                          plot_station_indices, plot_station_names, plot_link_ids)

    plt.close('all')
    print("Visualization complete.")

if __name__ == '__main__':
    main_visualization()
