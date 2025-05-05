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

from ifc_usgs_fileorder import load_usgs_mapping_from_path
from io_ifc import get_rainfall_for_lid_from_config

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
                         using_simulated_data, Cr_ref): # <-- Added flags/values
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
    ax.plot(time_axis, obs_series, 'k--', label='Observed')

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
        if using_simulated_data and Cr_ref is not None:
             scale_factor = Cr_ref
             # print(f"Info: Scaling rainfall by Cr_ref = {Cr_ref} for plotting.") # Optional info message

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
                                  using_simulated_data, Cr_ref, # <-- Added flags/values
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
                                 using_simulated_data=using_simulated_data, Cr_ref=Cr_ref) # <-- Pass flags/values

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
def plot_parameter_evolution(param_array, active_param_indices, param_labels, param_ranges, assimilation_phase, visual_output_dir, iter_range, cr_ref=None):
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
    
    num_iters = len(iter_range)
    num_stations = param_array.shape[2]
    
    # Ensemble plots: plot all ensemble trajectories
    for idx_active, orig_idx in enumerate(active_param_indices):
        for station_idx in range(num_stations):
            plt.figure()
            plt.plot(iter_range, param_array[:, idx_active, station_idx, :])
            plt.ylabel(param_labels[orig_idx])
            plt.xlabel('EKI Iterations')
            plt.ylim(*param_ranges[orig_idx])
            out_path = os.path.join(param_ensemble_dir, f"parameter_{orig_idx}_station_{station_idx}_ensemble.png")
            plt.savefig(out_path)
            plt.close()
            print(f"Saved parameter ensemble plot {out_path}")
    
    # Mean-Std plots: plot mean and standard deviation over iterations
    for idx_active, orig_idx in enumerate(active_param_indices):
        for station_idx in range(num_stations):
            plt.figure()
            param_mean = np.mean(param_array[:, idx_active, station_idx, :], axis=1)
            param_std = np.std(param_array[:, idx_active, station_idx, :], axis=1)
            plt.plot(iter_range, param_mean, 'k-', lw=2, label='Mean')
            plt.fill_between(iter_range, param_mean - param_std, param_mean + param_std,
                             color='gray', alpha=0.3, label='Mean ± Std')
            # **NEW**: overlay the Cr_ref line
            if cr_ref is not None:
                plt.axhline(cr_ref, color='red', linestyle='--', label='Cr_ref')
            plt.ylabel(param_labels[orig_idx])
            plt.xlabel('EKI Iterations')
            plt.ylim(*param_ranges[orig_idx])
            out_path = os.path.join(param_mean_std_dir, f"parameter_{orig_idx}_station_{station_idx}_mean_std.png")
            plt.savefig(out_path)
            plt.close()
            print(f"Saved parameter mean-std plot {out_path}")

# ===================== Event Statistics Plot Function =====================
def plot_event_statistics(assimilation_phase, visual_output_dir, out_dir):
    """
    Compute and plot event statistics (peak, mean, and standard deviation) for observed data.
    
    For simplicity, this example computes the statistics over the entire time series for each station.
    In a real application, you might implement a more sophisticated event detection algorithm.
    """
    measured_data = np.genfromtxt(out_dir+"csv/meas_mean.csv", delimiter=',', skip_header=1)
    measured_data[measured_data == 0] = np.nan

    event_peaks = np.nanmax(measured_data, axis=0)
    event_means = np.nanmean(measured_data, axis=0)
    event_stds  = np.nanstd(measured_data, axis=0)

    event_stats_dir = os.path.join(visual_output_dir, assimilation_phase, "event_statistics")
    clear_and_create_dir(event_stats_dir)

    stations = np.arange(measured_data.shape[1])
    plt.figure()
    plt.plot(stations, event_peaks, 'r-o', label="Peak")
    plt.xlabel("Station")
    plt.ylabel("Peak Value")
    plt.title(f"Event Peak Values ({assimilation_phase})")
    plt.legend()
    out_path = os.path.join(event_stats_dir, "event_peak.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved event peak plot {out_path}")

    plt.figure()
    plt.plot(stations, event_means, 'g-o', label="Mean")
    plt.xlabel("Station")
    plt.ylabel("Mean Value")
    plt.title(f"Event Mean Values ({assimilation_phase})")
    plt.legend()
    out_path = os.path.join(event_stats_dir, "event_mean.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved event mean plot {out_path}")

    plt.figure()
    plt.plot(stations, event_stds, 'b-o', label="Std")
    plt.xlabel("Station")
    plt.ylabel("Standard Deviation")
    plt.title(f"Event Standard Deviation ({assimilation_phase})")
    plt.legend()
    out_path = os.path.join(event_stats_dir, "event_std.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved event std plot {out_path}")

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
    Cr_ref = test_dict.get('Cr_ref', None) # Use .get for safety if key might be missing
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
                                  using_simulated_data, Cr_ref, # <-- Pass flags/values
                                  assimilation_phase='post', visual_output_dir=visual_output_dir, out_dir=test_dict['out_dir'])
    plot_parameter_evolution(post_param_array, active_param_indices, param_labels, param_ranges,
                             assimilation_phase='post', visual_output_dir=visual_output_dir, iter_range=iter_range_post, cr_ref=Cr_ref)
    plot_event_statistics('post', visual_output_dir, test_dict['out_dir'])

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
                                  using_simulated_data, Cr_ref, # <-- Pass flags/values
                                  assimilation_phase='prior', visual_output_dir=visual_output_dir, out_dir=test_dict['out_dir'])
    plot_parameter_evolution(prior_param_array, active_param_indices, param_labels, param_ranges,
                             assimilation_phase='prior', visual_output_dir=visual_output_dir, iter_range=iter_range_prior, cr_ref=Cr_ref)
    plot_event_statistics('prior', visual_output_dir, test_dict['out_dir'])
    
    plt.close('all')
    print("Visualization complete.")

if __name__ == '__main__':
    main_visualization()
