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
from tqdm import tqdm
import re
import requests
import time
import matplotlib.patheffects as pe
from adjustText import adjust_text


import geopandas as gpd

from data_handler import load_usgs_mapping_from_path, get_subwatershed, get_ids, get_rainfall_for_lid_from_config
from metric_operator import find_events, find_metric_values

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

# --- Helper Function to Fetch USGS Gauge Coordinates (from catchment_maps.py) ---
def get_usgs_coords(usgs_id):
    """Fetches latitude and longitude for a given USGS gauge ID from NWIS."""
    usgs_id_str = str(usgs_id).zfill(8)
    url = f"https://waterservices.usgs.gov/nwis/site/?format=rdb&sites={usgs_id_str}&siteOutput=expanded&siteStatus=all"
    # print(f"  Fetching coordinates for {usgs_id_str}...")
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        lat, lon = None, None
        lines = r.text.splitlines()
        content_lines = [line for line in lines if not line.startswith('#') and line.strip()]
        if len(content_lines) < 3:
             # print(f"  Warning: Unexpected RDB format or no data returned for {usgs_id_str}. Found {len(content_lines)} content lines.")
             return None, None
        header_line, data_line = content_lines[0], content_lines[2]
        header, values = header_line.split('\t'), data_line.split('\t')
        lat_col_name, lon_col_name = 'dec_lat_va', 'dec_long_va'
        try:
            lat_idx, lon_idx = header.index(lat_col_name), header.index(lon_col_name)
        except ValueError:
            # print(f"  Warning: Could not find '{lat_col_name}' or '{lon_col_name}' columns in RDB header for {usgs_id_str}.")
            return None, None
        if lat_idx < len(values) and lon_idx < len(values):
            lat_str, lon_str = values[lat_idx], values[lon_idx]
            try:
                if lat_str.strip() and lon_str.strip():
                    lat, lon = float(lat_str), float(lon_str)
                    return lat, lon
                # else: print(f"  Warning: Empty lat/lon value found for {usgs_id_str}.")
            except ValueError: print(f"  Warning: Could not parse lat/lon float for {usgs_id_str}. Values: '{lat_str}', '{lon_str}'")
        # else: print(f"  Warning: Lat/lon indices ({lat_idx}, {lon_idx}) out of bounds for data line len ({len(values)}) for {usgs_id_str}")
        return None, None
    except requests.exceptions.RequestException as e: print(f"  Error during API request for {usgs_id_str}: {e}")
    except Exception as e: print(f"  Unexpected error processing coords for {usgs_id_str}: {type(e).__name__} - {e}")
    return None, None

# ===================== Animation Frame Drawing Function =====================
def draw_animation_frame(iter_idx, ensemble_sim, station_idx, time_axis,
                         measured_data, station_label, rain_df,
                         using_simulated_data, cr_ref_for_scaling): # <-- Added flags/values
    """
    Draw a single animation frame, scaling rainfall if using simulated data.
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
    legend_text = 'Simulated Observation' if using_simulated_data else 'Observed'
    ax.plot(time_axis, obs_series, 'k--', label=legend_text)

    # # --- Metric calculation (Existing logic) ---
    # try:
    #     obs_1d = obs_series.flatten()
    #     sim_1d = median_sim.flatten()
    #     valid_indices = ~np.isnan(obs_1d) & ~np.isnan(sim_1d) & (obs_1d > 0)
    #     if np.any(valid_indices):
    #          kge_val = he.evaluator(he.nse, sim_1d[valid_indices], obs_1d[valid_indices])[0] # NSE used as proxy
    #          obs_peak_val = np.max(obs_1d[valid_indices]) if np.any(obs_1d[valid_indices]) else np.nan
    #          sim_peak_val = np.max(sim_1d[valid_indices]) if np.any(sim_1d[valid_indices]) else np.nan

    #          if not (np.isnan(obs_peak_val) or np.isnan(sim_peak_val) or obs_peak_val == 0):
    #              pr_diff = (sim_peak_val - obs_peak_val) / obs_peak_val
    #          else:
    #              pr_diff = np.nan
    #          pt_diff = np.argmax(sim_1d) - np.argmax(obs_1d)
    #          print(f"Iteration {iter_idx:02d}, Gauge {station_label}: KGE={kge_val:.3f}, PeakRelDiff={pr_diff:.3f}, PeakTimeDiff={pt_diff}")
    #     else:
    #         print(f"Iteration {iter_idx:02d}, Gauge {station_label}: Insufficient valid data for metrics.")
    # except Exception as e:
    #     print(f"Iteration {iter_idx:02d}, Gauge {station_label}: Metric calculation failed. Error: {e}")

    # --- Plot setup (Existing logic) ---
    ax.set_title(f'EKI iteration {iter_idx:02d} - Gauge {station_label}')
    ax.set_xlabel('Time')
    ax.set_ylabel('Discharge (m$^3$/s)')
    ax.legend(loc='upper left')
    ax.grid(True)

    # --- ADDED: Rainfall Plotting with conditional scaling ---
    if not rain_df.empty:
        ax2 = ax.twinx()
        scale_factor = cr_ref_for_scaling if using_simulated_data and cr_ref_for_scaling is not None else 1.0
        scaled_rainfall = rain_df["Rainfall"] * scale_factor
        flipped_rainfall = -scaled_rainfall
        min_flipped_rainfall = flipped_rainfall.min()
        ylim_bottom = min(min_flipped_rainfall * 1.2, min_flipped_rainfall - 0.5) if pd.notna(min_flipped_rainfall) else -1
        ax2.bar(rain_df.index, flipped_rainfall, width=1/24, alpha=0.7, color='deepskyblue', label='Rainfall (Flipped)', zorder=3)
        ax2.set_ylabel('Rainfall (mm/h, flipped)')
        ax2.set_ylim(ylim_bottom, 0)
    else:
        print(f"Iteration {iter_idx:02d}, Gauge {station_label}: No rainfall data to plot.")

    fig.autofmt_xdate()

# ===================== Data Loading Helper Function =====================
def load_ensemble(assimilation_phase, iter_idx, out_dir):
    """
    Load simulation ensemble data based on the assimilation phase and iteration index.
    """
    if assimilation_phase == 'post':
        file_path = os.path.join(out_dir, f'npy/{iter_idx - 1}_post_particles.npy') if iter_idx > 0 else os.path.join(out_dir, 'npy/0_prior_particles.npy')
    elif assimilation_phase == 'prior':
        file_path = os.path.join(out_dir, f'npy/{iter_idx}_prior_particles.npy')
    else:
        raise ValueError("assimilation_phase must be 'post' or 'prior'")
    with open(file_path, 'rb') as f:
        return np.load(f)

# ===================== Hydrograph Animation Generation Function =====================
def generate_hydrograph_animation(num_iters, station_indices, station_names, plot_link_ids,
                                  measured_data, time_axis, start_time_str, end_time_str, rain_dir,
                                  using_simulated_data, cr_ref_vec,
                                  link_to_division_map,
                                  assimilation_phase, visual_output_dir, out_dir):
    """
    Generate and save hydrograph animation GIFs. Includes rainfall plotting.
    """
    hydrograph_frames_dir = os.path.join(visual_output_dir, assimilation_phase, "hydrograph", "frames")
    hydrograph_anim_dir = os.path.join(visual_output_dir, assimilation_phase, "hydrograph", "animation")
    clear_and_create_dir(hydrograph_frames_dir)
    clear_and_create_dir(hydrograph_anim_dir)
    
    print(f"\n--- Generating {assimilation_phase.title()} Hydrograph Animations ---")
    iter_range = range(num_iters + 1) if assimilation_phase == 'post' else range(num_iters)

    # Use tqdm for a compact progress bar over stations
    for i, station_idx in enumerate(tqdm(station_indices, desc=f"Processing Stations ({assimilation_phase})")):
        station_label = station_names[i]
        target_lid = plot_link_ids[i]

        rain_df = get_rainfall_for_lid_from_config(target_lid, start_time_str, end_time_str, rain_dir)

        cr_ref_for_station = None
        if using_simulated_data and cr_ref_vec is not None and link_to_division_map is not None:
            division_id = link_to_division_map.get(target_lid)
            cr_ref_for_station = cr_ref_vec[division_id] if division_id is not None else cr_ref_vec[0]
            if division_id is None:
                tqdm.write(f"  - Warning: Could not find division for link_id {target_lid}. Using first Cr_ref value.")

        frame_imgs = []
        for iter_idx in iter_range:
            fig, ax = plt.subplots(figsize=(12, 6))
            ensemble_sim = load_ensemble(assimilation_phase, iter_idx, out_dir)
            max_obs_val = np.nanmax(measured_data[:, station_idx])
            y_limit_top = max_obs_val * 1.5 if pd.notna(max_obs_val) and max_obs_val > 0 else 1.0
            
            draw_animation_frame(iter_idx, ensemble_sim, station_idx, time_axis, measured_data,
                                 station_label=station_label, rain_df=rain_df,
                                 using_simulated_data=using_simulated_data, 
                                 cr_ref_for_scaling=cr_ref_for_station)
            ax.set_ylim(0, y_limit_top)
            plt.tight_layout()

            frame_filepath = os.path.join(hydrograph_frames_dir, f"iter_{iter_idx:02d}_gauge_{station_label}_hydrograph.png")
            plt.savefig(frame_filepath)
            frame_imgs.append(Image.open(frame_filepath))
            plt.close(fig)

        if frame_imgs:
            gif_filepath = os.path.join(hydrograph_anim_dir, f"gauge_{station_label}_hydrograph_animation.gif")
            frame_imgs[0].save(gif_filepath, save_all=True, append_images=frame_imgs[1:], duration=1000, loop=0)
            tqdm.write(f"  - Animation saved for Gauge {station_label}")
        else:
             tqdm.write(f"  - No frames generated for Gauge {station_label}, GIF not created.")

# ===================== Parameter Evolution Plot Function =====================
def plot_parameter_evolution(param_array, active_param_indices, param_labels, param_ranges,
                             assimilation_phase, visual_output_dir, iter_range,
                             cr_ref_vec=None):
    """
    Plot parameter evolution graphs for each sub-watershed division.
    """
    param_ensemble_dir = os.path.join(visual_output_dir, assimilation_phase, "parameter", "ensemble")
    param_mean_std_dir = os.path.join(visual_output_dir, assimilation_phase, "parameter", "mean_std")
    clear_and_create_dir(param_ensemble_dir)
    clear_and_create_dir(param_mean_std_dir)

    num_divisions = param_array.shape[2]

    for idx_active, orig_idx in enumerate(active_param_indices):
        # Sanitize parameter label for use in filename
        param_name_safe = re.sub(r'[^a-zA-Z0-9]', '', param_labels[orig_idx])

        for division_id in range(num_divisions):
            param_data_for_division = param_array[:, idx_active, division_id, :]

            cr_ref_for_plot = None
            if cr_ref_vec is not None and division_id < len(cr_ref_vec):
                cr_ref_for_plot = cr_ref_vec[division_id]

            # Ensemble Plot
            plt.figure(figsize=(10, 6))
            plt.plot(iter_range, param_data_for_division)
            plt.title(f'Parameter Ensemble - {param_labels[orig_idx]} (Division {division_id})')
            plt.ylabel(f"Value of {param_labels[orig_idx]}")
            plt.xlabel('EKI Iterations')
            plt.ylim(*param_ranges[orig_idx])
            if cr_ref_for_plot is not None:
                plt.axhline(cr_ref_for_plot, color='red', linestyle='--', label=f'Cr_ref = {cr_ref_for_plot:.2f}')
                plt.legend()
            out_path = os.path.join(param_ensemble_dir, f"parameter_{param_name_safe}_division_{division_id}_ensemble.png")
            plt.savefig(out_path)
            plt.close()

            # Mean-Std Plot
            plt.figure(figsize=(10, 6))
            param_mean = np.mean(param_data_for_division, axis=1)
            param_std = np.std(param_data_for_division, axis=1)
            plt.plot(iter_range, param_mean, 'k-', lw=2, label='Mean')
            plt.fill_between(iter_range, param_mean - param_std, param_mean + param_std,
                             color='gray', alpha=0.3, label='Mean ± Std')
            if cr_ref_for_plot is not None:
                plt.axhline(cr_ref_for_plot, color='red', linestyle='--', label=f'Cr_ref = {cr_ref_for_plot:.2f}')
            plt.title(f'Parameter Mean Evolution - {param_labels[orig_idx]} (Division {division_id})')
            plt.ylabel(f"Value of {param_labels[orig_idx]}")
            plt.xlabel('EKI Iterations')
            plt.ylim(*param_ranges[orig_idx])
            plt.legend()
            out_path = os.path.join(param_mean_std_dir, f"parameter_{param_name_safe}_division_{division_id}_mean_std.png")
            plt.savefig(out_path)
            plt.close()

    print("Finished plotting parameter evolution for all divisions.")

# ===================== Consolidated Parameter Evolution Plot Function =====================
def plot_parameter_evolution_consolidated(param_array, active_param_indices, param_labels, param_ranges,
                                          assimilation_phase, visual_output_dir, iter_range,
                                          cr_ref_vec=None):
    """
    Plot consolidated parameter evolution graphs for all divisions in a single figure.
    """
    param_consolidated_dir = os.path.join(visual_output_dir, assimilation_phase, "parameter", "consolidated")
    clear_and_create_dir(param_consolidated_dir)

    num_divisions = param_array.shape[2]

    for idx_active, orig_idx in enumerate(active_param_indices):
        # Sanitize parameter label for use in filename
        param_name_safe = re.sub(r'[^a-zA-Z0-9]', '', param_labels[orig_idx])

        # Determine grid size for subplots
        grid_size = int(np.ceil(np.sqrt(num_divisions)))
        fig, axes = plt.subplots(grid_size, grid_size, 
                                 figsize=(4 * grid_size, 3 * grid_size), 
                                 sharex=True, sharey=True, squeeze=False)
        axes = axes.flatten()

        for division_id in range(num_divisions):
            ax = axes[division_id]
            param_data_for_division = param_array[:, idx_active, division_id, :]
            
            param_mean = np.mean(param_data_for_division, axis=1)
            param_std = np.std(param_data_for_division, axis=1)

            ax.plot(iter_range, param_mean, 'k-', lw=2, label='Mean')
            ax.fill_between(iter_range, param_mean - param_std, param_mean + param_std,
                            color='gray', alpha=0.3, label='Mean ± Std')

            if cr_ref_vec is not None and division_id < len(cr_ref_vec):
                cr_ref_for_plot = cr_ref_vec[division_id]
                ax.axhline(cr_ref_for_plot, color='red', linestyle='--', label=f'Cr_ref = {cr_ref_for_plot:.2f}')

            ax.set_title(f'Division {division_id}')
            ax.grid(True, linestyle='--')

        # Hide unused subplots
        for i in range(num_divisions, len(axes)):
            axes[i].set_visible(False)

        # Add a single shared legend
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right')

        fig.suptitle(f'Consolidated Mean Evolution - {param_labels[orig_idx]} ({assimilation_phase})', fontsize=16)
        fig.supxlabel('EKI Iterations')
        fig.supylabel(f"Value of {param_labels[orig_idx]}")
        
        # Set shared properties
        plt.setp(axes, ylim=param_ranges[orig_idx])
        fig.tight_layout(rect=[0, 0, 0.9, 0.95]) # Adjust layout for suptitle and legend

        out_path = os.path.join(param_consolidated_dir, f"param_{param_name_safe}_all_divisions_consolidated.png")
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
    print("Finished plotting consolidated parameter evolution.")

# ===================== Event Statistics Plot Function =====================
def plot_event_statistics(assimilation_phase, visual_output_dir, out_dir, test_dict, 
                          plot_station_indices, plot_station_names, plot_link_ids):
    """
    Tracks and plots the evolution of simulated event metrics.
    """
    if assimilation_phase != 'post':
        return

    event_stats_dir = os.path.join(visual_output_dir, assimilation_phase, "event_statistics")
    clear_and_create_dir(event_stats_dir)

    # --- Step 1: Load static data and configuration ---
    print("\n--- Preparing for Event Statistics Evolution Analysis ---")
    observed_data = np.genfromtxt(os.path.join(out_dir, "csv", "meas_mean.csv"), delimiter=',', skip_header=1)
    num_steps = test_dict['steps']
    iter_range = np.arange(0, num_steps + 1) # Iterate from 0 to num_steps
    event_params = test_dict.get('event_finding', {})
    min_dist = event_params.get('min_dist', 24)
    min_thresh_pct = event_params.get('min_thresh_pct', 25)
    min_length = event_params.get('min_length', 72)
    metric_names = ['Peak', 'Mean', 'Std_Dev', 'Timing_Mean', 'Timing_Std_Dev']

    # --- Step 2: Pre-process events and observed metrics for all stations ---
    station_event_info = {}
    for i, station_idx in enumerate(plot_station_indices):
        station_name = plot_station_names[i]
        obs_series = observed_data[:, station_idx]
        min_thresh_val = np.percentile(obs_series[obs_series > 0], min_thresh_pct) if np.any(obs_series > 0) else 0
        event_indices_list, _ = find_events(obs_series, min_dist, min_thresh_val, min_length)
        print(f"  - Found {len(event_indices_list)} events for station {station_name} using threshold {min_thresh_val:.2f}.")

        if not event_indices_list:
            print(f"  - Warning: No events found for station {station_name}. It will be skipped.")
            continue

        observed_metrics = {}
        for event_num, event_indices in enumerate(event_indices_list):
            if len(event_indices) < 3:
                # Assign NaNs if event is too short for stable metrics
                obs_vals = [np.nan] * 5
            else:
                _, y_max, y_mean, _, _, _, y_std, y_mean_time, y_std_time = find_metric_values([event_indices], [obs_series[event_indices]])
                obs_vals = [y_max[0][0], y_mean[0][0], y_std[0][0], y_mean_time[0][0], y_std_time[0][0]]
            
            observed_metrics[event_num] = {
                'Peak': obs_vals[0], 'Mean': obs_vals[1], 'Std_Dev': obs_vals[2],
                'Timing_Mean': obs_vals[3], 'Timing_Std_Dev': obs_vals[4]
            }
        
        station_event_info[station_name] = {
            'station_idx': station_idx,
            'link_id': plot_link_ids[i],
            'event_indices_list': event_indices_list,
            'observed_metrics': observed_metrics
        }

    # --- Step 3: Efficiently gather simulated metrics evolution ---
    # Initialize a data structure to hold all evolution data, keyed by station name
    evolution_data = {
        s_name: {metric: {e_num: [] for e_num in range(len(s_info['event_indices_list']))} for metric in metric_names}
        for s_name, s_info in station_event_info.items()
    }

    print("\n--- Gathering simulated metrics across all iterations and stations (optimized) ---")
    for iter_idx in tqdm(iter_range, desc="Processing EKI Iterations"):
        # Load particle file ONCE per iteration
        file_path = os.path.join(out_dir, 'npy', f'{iter_idx-1}_post_particles.npy') if iter_idx > 0 else os.path.join(out_dir, 'npy', '0_prior_particles.npy')
        if not os.path.exists(file_path):
            print(f"Warning: Particles file not found for iter {iter_idx}, path: {file_path}. Stopping.")
            break
        
        simulated_particles = np.load(file_path)

        # Distribute calculations to each station
        for station_name, s_info in station_event_info.items():
            station_idx = s_info['station_idx']
            station_particles = simulated_particles[:, :, station_idx]
            
            for event_num, event_indices in enumerate(s_info['event_indices_list']):
                event_ensemble_data = station_particles[:, event_indices]
                ens_size = station_particles.shape[0]
                timing_mean_ensemble, timing_std_ensemble = (np.zeros(ens_size), np.zeros(ens_size))

                for particle_idx in range(ens_size):
                    vals = event_ensemble_data[particle_idx, :]
                    if np.any(vals) and np.sum(vals) > 0:
                        mean_t = np.sum(event_indices * vals) / np.sum(vals)
                        var_t = np.sum(vals * (event_indices - mean_t)**2) / np.sum(vals)
                        timing_mean_ensemble[particle_idx] = mean_t
                        timing_std_ensemble[particle_idx] = np.sqrt(var_t)
                    else:
                        timing_mean_ensemble[particle_idx] = np.nan
                        timing_std_ensemble[particle_idx] = np.nan
                
                # Append this iteration's ensemble metrics to the main data structure
                data_to_append = {
                    'Peak': np.max(event_ensemble_data, axis=1),
                    'Mean': np.mean(event_ensemble_data, axis=1),
                    'Std_Dev': np.std(event_ensemble_data, axis=1),
                    'Timing_Mean': timing_mean_ensemble,
                    'Timing_Std_Dev': timing_std_ensemble
                }
                for metric_name in metric_names:
                    evolution_data[station_name][metric_name][event_num].append(data_to_append[metric_name])

    # --- Step 4: Generate one plot per station using the pre-computed data ---
    print("\n--- Generating evolution plots for each station ---")
    for station_name, s_info in station_event_info.items():
        target_lid = s_info['link_id']
        num_events = len(s_info['event_indices_list'])
        observed_metrics = s_info['observed_metrics']

        fig, axes = plt.subplots(len(metric_names), 1, figsize=(12, 18), sharex=True)
        fig.suptitle(f'Event Metrics Evolution for Station {station_name} (LID: {target_lid})', fontsize=16)
        colors = plt.cm.viridis(np.linspace(0, 1, num_events)) if num_events > 1 else ['blue']

        # For each metric, create a subplot
        for i, metric_name in enumerate(metric_names):
            ax = axes[i]
            # For each event, plot its evolution line
            for event_num in range(num_events):
                metric_evolution_array = np.vstack(evolution_data[station_name][metric_name][event_num])
                mean_evolution = np.nanmean(metric_evolution_array, axis=1)
                std_evolution = np.nanstd(metric_evolution_array, axis=1)
                
                ax.plot(iter_range, mean_evolution, color=colors[event_num], label=f'Event {event_num+1} Sim Mean')
                ax.fill_between(iter_range, mean_evolution - std_evolution, mean_evolution + std_evolution, color=colors[event_num], alpha=0.15)
                
                target_value = observed_metrics[event_num][metric_name]
                if pd.notna(target_value):
                    ax.axhline(target_value, color=colors[event_num], linestyle='--', label=f'Event {event_num+1} Obs ({target_value:.2f})')
            
            y_label = 'Discharge (m$^3$/s)' if 'Timing' not in metric_name else 'Time (hours)'
            ax.set_ylabel(y_label)
            ax.set_title(f'Evolution of {metric_name.replace("_", " ")}')
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            ax.grid(True)

        axes[-1].set_xlabel('EKI Iteration')
        fig.tight_layout(rect=[0, 0, 0.85, 0.96])
        
        output_path = os.path.join(event_stats_dir, f"station_{station_name}_all_metrics_evolution.png")
        plt.savefig(output_path)
        plt.close(fig)
        print(f"  - Saved consolidated evolution plot for Station {station_name}")

# ===================== Geographic Map Generation Function =====================
def generate_cr_map(assimilation_phase, visual_output_dir, out_dir, test_dict, active_param_indices, param_labels, post_param_array=None, cr_ref_vec=None):
    """
    Generates and saves a map of Cr values or Cr error, with robust data handling and detailed annotations.
    """
    if assimilation_phase != 'post': return
    print("\n--- Generating Final Cr Parameter Map ---")
    maps_dir = os.path.join(visual_output_dir, assimilation_phase, "maps")
    clear_and_create_dir(maps_dir)

    # --- Find the index for the '$Cr$' parameter ---
    try:
        cr_orig_idx = param_labels.index('$Cr$')
    except ValueError:
        print("Warning: '$Cr$' not found in `prm_names` from config. Skipping Cr map generation.")
        return

    if cr_orig_idx not in active_param_indices:
        print("Warning: '$Cr$' is not an active parameter in this run. Skipping Cr map generation.")
        return
    
    # This is the row index in the CSV file of active parameters
    cr_active_idx = active_param_indices.index(cr_orig_idx)

    # --- Calculate convergence metrics if full parameter history is provided ---
    convergence_metrics = {}
    if assimilation_phase == 'post' and post_param_array is not None and post_param_array.ndim == 4:
        print("  Calculating parameter convergence metrics...")
        # Shape: (iterations, params, divisions, particles)
        num_iterations, _, num_divisions, _ = post_param_array.shape
        # Use the dynamically found index for the Cr parameter
        if cr_active_idx >= post_param_array.shape[1]:
            print(f"Warning: cr_active_idx ({cr_active_idx}) is out of bounds for post_param_array. Skipping convergence metrics.")
        else:
            param_data = post_param_array[:, cr_active_idx, :, :] # Shape: (iterations, divisions, particles)
        
        for div_id in range(num_divisions):
            std_over_time = np.std(param_data[:, div_id, :], axis=1)

            # 1. Iter to Absolute Convergence (std < 0.05)
            abs_conv_thresh = 0.05
            abs_conv_indices = np.where(std_over_time < abs_conv_thresh)[0]
            iter_abs_conv = abs_conv_indices[0] if len(abs_conv_indices) > 0 else "N/A"

            # 2. Iter to Stabilize (based on relative change in std)
            iter_stabilize = "N/A"
            window_size = 2
            tolerance = 0.01
            if len(std_over_time) > window_size:
                # Use np.diff to get change, handle potential division by zero
                with np.errstate(divide='ignore', invalid='ignore'):
                    rel_change = np.abs(np.diff(std_over_time) / std_over_time[:-1])
                # A large value for invalid results will cause the check to fail, which is desired.
                rel_change[np.isinf(rel_change) | np.isnan(rel_change)] = np.inf

                # Find the first window of consecutive changes below the tolerance
                for k in range(len(rel_change) - window_size + 1):
                    window = rel_change[k : k + window_size]
                    if np.all(window < tolerance):
                        # The stable plateau begins at iteration k, as the change from k -> k+1 is small.
                        iter_stabilize = k
                        break
            
            convergence_metrics[div_id] = {
                'abs_conv': iter_abs_conv,
                'stabilize': iter_stabilize
            }

    shapefile_path = test_dict.get('shapefile_path')
    if not shapefile_path or not os.path.exists(shapefile_path):
        print(f"Warning: Shapefile not found at {shapefile_path}. Skipping map generation.")
        return

    last_iter_idx = test_dict['steps'] - 1
    if last_iter_idx < 0: return
    
    param_mean_file = os.path.join(out_dir, 'csv', f"{last_iter_idx}_post_params_mean.csv")
    param_std_file = os.path.join(out_dir, 'csv', f"{last_iter_idx}_post_params_std.csv")

    if not os.path.exists(param_mean_file):
        print(f"Warning: Final parameter mean file not found: {param_mean_file}. Skipping map generation.")
        return

    cr_sparse_mean = pd.read_csv(param_mean_file, header=None).to_numpy()[cr_active_idx, :]
    if os.path.exists(param_std_file):
        cr_sparse_std = pd.read_csv(param_std_file, header=None).to_numpy()[cr_active_idx, :]
    else:
        cr_sparse_std = np.full_like(cr_sparse_mean, np.nan)

    
    model_link_ids = get_ids(test_dict)
    sparse_parent, link_to_division_map = get_subwatershed(test_dict, model_link_ids)
    
    cr_df = pd.DataFrame({
        'link_id': model_link_ids,
        'Cr_mean': sparse_parent.T @ cr_sparse_mean,
        'Cr_std': sparse_parent.T @ cr_sparse_std,
        'division_id': [link_to_division_map.get(lid) for lid in model_link_ids]
    })
    cr_df = cr_df[cr_df['division_id'].notna()]

    gdf = gpd.read_file(shapefile_path)
    if 'LINKNO' not in gdf.columns:
        print("Warning: Shapefile is missing 'LINKNO' column. Skipping map.")
        return
    gdf.rename(columns={'LINKNO': 'link_id'}, inplace=True)
    gdf['link_id'] = gdf['link_id'].astype(int)
    
    gdf_to_plot = gdf.merge(cr_df, on='link_id', how='inner')

    if gdf_to_plot.empty:
        print("Warning: No matching geometries found between shapefile and parameter data. Skipping map generation.")
        return

    is_simulated = test_dict.get('using_simulated_data', False) and cr_ref_vec is not None
    
    plot_col = 'Cr_mean'
    plot_title = f'Final Calibrated Cr Distribution ({assimilation_phase})'
    cbar_label = 'Calibrated Cr Value (Mean)'

    if is_simulated:
        ref_cr_map = {div_id: cr_ref_vec[div_id] for div_id in range(len(cr_ref_vec))}
        gdf_to_plot['Cr_ref'] = gdf_to_plot['division_id'].map(ref_cr_map)
        gdf_to_plot['percentage_diff'] = ((gdf_to_plot['Cr_mean'] - gdf_to_plot['Cr_ref']).abs() / gdf_to_plot['Cr_ref']) * 100
        
        PERCENTAGE_THRESHOLDS = [0.1, 1, 5, 10, 20, 50, 100]
        bins = [0] + PERCENTAGE_THRESHOLDS + [float('inf')]
        labels = [f'≤ {PERCENTAGE_THRESHOLDS[0]}%'] + \
                 [f'{PERCENTAGE_THRESHOLDS[i]}% - {PERCENTAGE_THRESHOLDS[i+1]}%' for i in range(len(PERCENTAGE_THRESHOLDS)-1)] + \
                 [f'> {PERCENTAGE_THRESHOLDS[-1]}%']
        
        gdf_to_plot['diff_category'] = pd.cut(gdf_to_plot['percentage_diff'], bins=bins, labels=labels, right=True, include_lowest=True, ordered=True)
        plot_col = 'diff_category'
        plot_title = f'Final Cr Parameter Error ({assimilation_phase})'
        cbar_label = f'% Difference from Reference Cr'
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    if is_simulated:
        active_categories = gdf_to_plot['diff_category'].cat.categories
        cmap = plt.get_cmap('Reds', len(active_categories))
        gdf_to_plot.plot(ax=ax, column=plot_col, cmap=cmap, legend=True,
                         legend_kwds={'title': cbar_label, 'loc': 'upper left', 'bbox_to_anchor': (1, 1)},
                         categorical=True, missing_kwds={"color": "lightgrey", "label": "No Data"},
                         zorder=1)
    else:
        gdf_to_plot.plot(column=plot_col, ax=ax, legend=True,
                         legend_kwds={'label': cbar_label, 'orientation': "horizontal"},
                         zorder=1)

    if 'division_id' in gdf_to_plot.columns:
        agg_dict = {'Cr_mean': 'first', 'Cr_std': 'first'}
        if is_simulated:
            agg_dict['Cr_ref'] = 'first'
        dissolved = gdf_to_plot.dissolve(by='division_id', aggfunc=agg_dict)
        dissolved['point'] = dissolved.geometry.representative_point()
        texts = []

        for division_id, row in dissolved.iterrows():
            point = row['point']
            cr_mean = row.get('Cr_mean')
            conv_data = convergence_metrics.get(division_id, {})
            if pd.notna(cr_mean) and point:
                annotation_text_lines = []
                annotation_text_lines.append(f"Division: {division_id}") # ADDED: Division ID
                if is_simulated:
                    cr_ref = row.get('Cr_ref')
                    cr_std = row.get('Cr_std')
                    rel_error = ((cr_mean - cr_ref) / cr_ref) * 100 if cr_ref != 0 else np.inf
                    annotation_text_lines.append(f"True Value: {cr_ref:.2f}")
                    annotation_text_lines.append(f"EKI Mean: {cr_mean:.2f}")
                    annotation_text_lines.append(f"Mean Rel. Err: {rel_error:.1f}%")
                    annotation_text_lines.append(f"EKI Std: {cr_std:.2f}" if pd.notna(cr_std) else "EKI Std: N/A")
                else:
                    cr_std = row.get('Cr_std')
                    annotation_text_lines.append(f"EKI Mean: {cr_mean:.2f}")
                    if pd.notna(cr_std):
                        annotation_text_lines.append(f"EKI Std: {cr_std:.2f}")

                # ADD convergence metrics
                iter_abs_conv = conv_data.get('abs_conv', "N/A")
                iter_stabilize = conv_data.get('stabilize', "N/A")
                annotation_text_lines.append(f"Iter to Abs. Conv.: {iter_abs_conv}")
                annotation_text_lines.append(f"Iter to Stabilize: {iter_stabilize}")
                annotation_text = "\n".join(annotation_text_lines)

                texts.append(ax.text(row.point.x, row.point.y, annotation_text, fontsize=6, ha='center', va='center',
                                    bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8, ec='none'), zorder=3))
        
        if texts:
            adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='->', color='black', lw=0.5, zorder=2))

    # Add annotation for hyperparameter definition
    hyperparam_text = "Note: 'Iter to Abs. Conv.' is the first iteration\nwhere the parameter ensemble std < 0.05."
    ax.text(0.99, 0.01, hyperparam_text, transform=ax.transAxes, fontsize=7,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', alpha=0.9, ec='grey'))
    
    ax.set_title(plot_title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_axis_off()
    plt.tight_layout()

    output_path = os.path.join(maps_dir, f"final_cr_map_{assimilation_phase}.png")
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved Cr map to {output_path}")

def generate_hydrograph_metric_map(assimilation_phase, visual_output_dir, out_dir, test_dict, observed_data,
                                   plot_station_indices, plot_station_names, model_link_ids, plot_link_ids):
    """
    Generates a map showing the flowline network, gauge locations, and key performance metrics at each gauge.
    """
    if assimilation_phase != 'post': return
    print("\n--- Generating Hydrograph & Metric Map ---")
    maps_dir = os.path.join(visual_output_dir, assimilation_phase, "maps")
    os.makedirs(maps_dir, exist_ok=True)

    shapefile_path = test_dict.get('shapefile_path')
    if not shapefile_path or not os.path.exists(shapefile_path):
        print(f"Warning: Shapefile not found at {shapefile_path}. Skipping metric map generation.")
        return

    # --- Define key gauge IDs ---
    all_gauge_ids_to_plot = list(dict.fromkeys(test_dict.get("plot_usgs", []))) # Remove duplicates
    assimilation_gauge_ids = test_dict.get("meas_usgs", [])
    if isinstance(assimilation_gauge_ids, str): # Backward compatibility
        assimilation_gauge_ids = [assimilation_gauge_ids]
    # Ensure they are all strings and stripped
    assimilation_gauge_ids = [str(g).strip() for g in assimilation_gauge_ids]

    most_downstream_gauge_id = str(test_dict['visualization_settings']['downstream_outlet_usgs_id']).strip()

    # --- Fetch gauge coordinates ---
    gauge_data_list = []
    print("Fetching gauge coordinates via USGS API for metric map...")
    for gauge_id in all_gauge_ids_to_plot:
        lat, lon = get_usgs_coords(gauge_id)
        if lat is not None and lon is not None:
            gauge_data_list.append({'gauge_id': str(gauge_id).strip(), 'lat': lat, 'lon': lon})
        else:
            print(f"  Failed to retrieve coordinates for gauge: {gauge_id}")
        time.sleep(0.1)
    
    if not gauge_data_list:
        print("Could not fetch any gauge coordinates. Skipping metric map.")
        return

    gauge_points = gpd.GeoDataFrame(
        pd.DataFrame(gauge_data_list),
        geometry=gpd.points_from_xy(pd.DataFrame(gauge_data_list).lon, pd.DataFrame(gauge_data_list).lat),
        crs="EPSG:4326"
    )
    
    # --- Load final simulation results ---
    last_iter_idx = test_dict['steps'] - 1
    if last_iter_idx < 0: return
    final_sim_file = os.path.join(out_dir, "npy", f"{last_iter_idx}_post_particles.npy")
    if not os.path.exists(final_sim_file):
        print(f"Final simulation file not found: {final_sim_file}. Skipping metric map.")
        return
    final_sim_ensemble = np.load(final_sim_file)
    final_sim_median = np.median(final_sim_ensemble, axis=0)

    # --- Calculate detailed error metrics for each gauge ---
    metrics_data = {}
    event_params = test_dict.get('event_finding', {})
    min_dist, min_thresh_pct, min_length = event_params.get('min_dist', 24), event_params.get('min_thresh_pct', 25), event_params.get('min_length', 72)

    for i, station_id_str in enumerate(plot_station_names):
        station_idx = plot_station_indices[i]
        obs_series = observed_data[:, station_idx]
        sim_median_series = final_sim_median[:, station_idx]
        sim_ensemble_series = final_sim_ensemble[:, :, station_idx]

        # 1. Hydrograph series relative error
        with np.errstate(divide='ignore', invalid='ignore'):
            valid_obs_mask = obs_series > 0.1 # Avoid division by zero or tiny numbers
            series_rel_err = np.abs((sim_median_series[valid_obs_mask] - obs_series[valid_obs_mask]) / obs_series[valid_obs_mask])
            avg_series_rel_err = np.nanmean(series_rel_err)

        # 2. Max hydrograph ensemble std
        max_ensemble_std = np.nanmax(np.std(sim_ensemble_series, axis=0))

        # 3. Metric relative errors (averaged over events)
        min_thresh_val = np.percentile(obs_series[obs_series > 0], min_thresh_pct) if np.any(obs_series > 0) else 0
        event_indices_list, _ = find_events(obs_series, min_dist, min_thresh_val, min_length)

        metric_names = ['Peak', 'Mean', 'Std_Dev', 'Timing_Mean', 'Timing_Std_Dev']
        metric_rel_errors = {name: [] for name in metric_names}
        avg_metric_rel_errors = {name: np.nan for name in metric_names}

        if event_indices_list:
            for event_indices in event_indices_list:
                # Skip events that are too short for stable polyfit, avoiding RankWarning
                if len(event_indices) < 3:
                    continue

                obs_event_data = obs_series[event_indices]
                sim_event_data = sim_median_series[event_indices]

                _, obs_y_max, obs_y_mean, _, _, _, obs_y_std, obs_y_mean_time, obs_y_std_time = find_metric_values([event_indices], [obs_event_data])
                obs_metrics = {'Peak': obs_y_max[0][0], 'Mean': obs_y_mean[0][0], 'Std_Dev': obs_y_std[0][0], 'Timing_Mean': obs_y_mean_time[0][0], 'Timing_Std_Dev': obs_y_std_time[0][0]}

                _, sim_y_max, sim_y_mean, _, _, _, sim_y_std, sim_y_mean_time, sim_y_std_time = find_metric_values([event_indices], [sim_event_data])
                sim_metrics = {'Peak': sim_y_max[0][0], 'Mean': sim_y_mean[0][0], 'Std_Dev': sim_y_std[0][0], 'Timing_Mean': sim_y_mean_time[0][0], 'Timing_Std_Dev': sim_y_std_time[0][0]}

                for name in metric_names:
                    obs_val, sim_val = obs_metrics.get(name), sim_metrics.get(name)
                    if obs_val is not None and sim_val is not None and obs_val != 0:
                        rel_err = np.abs((sim_val - obs_val) / obs_val)
                        metric_rel_errors[name].append(rel_err)
                    else:
                        metric_rel_errors[name].append(np.nan)
            
            for name in metric_names:
                if metric_rel_errors[name]:
                    avg_metric_rel_errors[name] = np.nanmean(metric_rel_errors[name])
                
        metrics_data[station_id_str] = {
            'AvgSeriesRelErr': avg_series_rel_err,
            'MaxEnsembleStd': max_ensemble_std,
            'AvgMetricRelErr': avg_metric_rel_errors
        }

    # --- Plot the map ---
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    network_gdf = gpd.read_file(shapefile_path)

    if 'LINKNO' in network_gdf.columns:
        network_gdf.rename(columns={'LINKNO': 'link_id'}, inplace=True)
        network_gdf['link_id'] = network_gdf['link_id'].astype(int)
        network_gdf = network_gdf[network_gdf['link_id'].isin(model_link_ids)]
    else:
        print("Warning: Shapefile is missing 'LINKNO' column. Cannot filter network.")

    network_gdf = network_gdf.to_crs("EPSG:4326") # Ensure CRS matches gauge points
    network_gdf.plot(ax=ax, lw=0.7, color="blue", zorder=3)

    # Set map extent
    bounds = network_gdf.total_bounds
    buffer = 0.05
    ax.set_xlim(bounds[0] - buffer, bounds[2] + buffer)
    ax.set_ylim(bounds[1] - buffer, bounds[3] + buffer)

    # Plot gauge points
    path_effects = [pe.withStroke(linewidth=3, foreground="white")]
    
    # Plot non-special gauges first
    special_gauges = set(assimilation_gauge_ids) | {most_downstream_gauge_id}
    other_gauges = gauge_points[~gauge_points['gauge_id'].isin(special_gauges)]
    if not other_gauges.empty:
        other_gauges.plot(ax=ax, marker='o', color='red', markersize=40, edgecolor='black', zorder=5, label='Verification Gauge')

    # Plot assimilation gauge
    assim_gauge_plot = gauge_points[gauge_points['gauge_id'].isin(assimilation_gauge_ids)]
    if not assim_gauge_plot.empty:
         assim_gauge_plot.plot(ax=ax, marker='o', color='cyan', markersize=60, edgecolor='black', zorder=5, label='Assimilation Gauge(s)')

    # Plot the most downstream gauge with a star on top
    downstream_gauge = gauge_points[gauge_points['gauge_id'] == most_downstream_gauge_id]
    if not downstream_gauge.empty:
        downstream_gauge.plot(ax=ax, marker='*', color='yellow', markersize=200, edgecolor='black', zorder=6, label='Downstream Outlet')

    # Add annotations
    texts = []
    for _, row in gauge_points.iterrows():
        gauge_id = row.gauge_id
        metrics = metrics_data.get(gauge_id)
        if metrics:
            # Pre-format strings to avoid ValueError with f-string conditional format specifiers
            metric_errs = metrics['AvgMetricRelErr']
            avg_series_rel_err_str = f"{metrics['AvgSeriesRelErr']:.2%}" if pd.notna(metrics['AvgSeriesRelErr']) else 'N/A'
            peak_err_str = f"{metric_errs['Peak']:.2%}" if pd.notna(metric_errs['Peak']) else 'N/A'
            mean_err_str = f"{metric_errs['Mean']:.2%}" if pd.notna(metric_errs['Mean']) else 'N/A'
            std_dev_err_str = f"{metric_errs['Std_Dev']:.2%}" if pd.notna(metric_errs['Std_Dev']) else 'N/A'
            timing_mean_err_str = f"{metric_errs['Timing_Mean']:.2%}" if pd.notna(metric_errs['Timing_Mean']) else 'N/A'
            timing_std_dev_err_str = f"{metric_errs['Timing_Std_Dev']:.2%}" if pd.notna(metric_errs['Timing_Std_Dev']) else 'N/A'

            annotation_text = (
                f"Gauge: {gauge_id}\n"
                f"--------------------------\n"
                f"Avg. Series Rel. Err: {avg_series_rel_err_str}\n"
                f"Max Ens. Std: {metrics['MaxEnsembleStd']:.2f}\n"
                f"Event Metric Avg. Rel. Err:\n"
                f"  - Peak: {peak_err_str}\n"
                f"  - Mean: {mean_err_str}\n"
                f"  - Std Dev: {std_dev_err_str}\n"
                f"  - Timing Mean: {timing_mean_err_str}\n"
                f"  - Timing Std Dev: {timing_std_dev_err_str}"
            )
            texts.append(ax.text(row.geometry.x, row.geometry.y, annotation_text, fontsize=7,
                                 bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8, ec='gray', lw=0.5),
                                 path_effects=path_effects, zorder=10))
    
    if texts:
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='->', color='black', lw=0.5, relpos=(0.5, 0.5)))
    
    ax.set_title(f"Flowline Network, Gauges & Final Hydrograph Error Analysis ({assimilation_phase})")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='best')
    plt.tight_layout()

    output_path = os.path.join(maps_dir, f"hydrograph_metric_map_{assimilation_phase}.png")
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved hydrograph metric map to {output_path}")


def load_and_aggregate_rainfall_by_division(start_time_str, end_time_str, rain_dir, link_to_division_map):
    """
    Efficiently loads all rainfall data within a time window and aggregates it by sub-watershed division.
    This function reads each binary rainfall file only once.
    """
    if not os.path.isdir(rain_dir):
        print(f"Warning: Rainfall directory not found: {rain_dir}")
        return {}

    try:
        start_time = pd.to_datetime(start_time_str)
        end_time = pd.to_datetime(end_time_str)
    except ValueError as e:
        print(f"Error: Could not parse start/end time strings: {e}")
        return {}

    duration_hours = (end_time - start_time).total_seconds() / 3600.0
    if duration_hours <= 0:
        print(f"Warning: Time duration is non-positive ({duration_hours} hours). Cannot calculate average rate.")
        # Return total accumulation instead of a meaningless rate
        duration_hours = 1 

    # --- Find all relevant rainfall files ---
    files_to_process = []
    years_in_range = {str(y) for y in range(start_time.year, end_time.year + 1)}
    
    # Check for yearly subdirectories
    potential_year_dirs = [os.path.join(rain_dir, item) for item in os.listdir(rain_dir) if os.path.isdir(os.path.join(rain_dir, item)) and re.fullmatch(r'(19|20)\d{2}', item)]
    
    dirs_to_scan = [p for p in potential_year_dirs if os.path.basename(p) in years_in_range]
    if not dirs_to_scan:
        dirs_to_scan = [rain_dir] # Fallback to flat structure

    for data_dir in dirs_to_scan:
        for filename in os.listdir(data_dir):
            if filename.isdigit():
                timestamp = pd.to_datetime(int(filename), unit='s')
                if start_time <= timestamp <= end_time:
                    files_to_process.append(os.path.join(data_dir, filename))

    if not files_to_process:
        print("Warning: No rainfall files found in the specified time range.")
        return {}

    # --- Process files and aggregate rainfall ---
    num_divisions = max(link_to_division_map.values()) + 1
    division_rainfall_totals = np.zeros(num_divisions)
    
    print("Aggregating rainfall data by division...")
    for file_path in tqdm(files_to_process):
        try:
            with open(file_path, "rb") as f:
                raw_data = f.read()
            
            if len(raw_data) < 8: continue
            raw_data = raw_data[4:] # Skip 4-byte header

            for lid, rainfall in struct.iter_unpack("if", raw_data):
                division_id = link_to_division_map.get(lid)
                if division_id is not None:
                    division_rainfall_totals[division_id] += rainfall
        except Exception as e:
            print(f"Warning: Could not process file {file_path}. Error: {e}")
            
    return {i: total / duration_hours for i, total in enumerate(division_rainfall_totals) if total > 0}


def generate_rainfall_map(visual_output_dir, test_dict, model_link_ids, link_to_division_map):
    """
    Generates and saves a map of total rainfall distribution by division using an efficient aggregation method.
    """
    print("\n--- Generating Total Rainfall Map ---")
    maps_dir = os.path.join(visual_output_dir, "rainfall_map")
    clear_and_create_dir(maps_dir)

    shapefile_path = test_dict.get('shapefile_path')
    if not shapefile_path or not os.path.exists(shapefile_path):
        print(f"Warning: Shapefile not found at {shapefile_path}. Skipping rainfall map generation.")
        return

    # --- Get time period for title ---
    start_time_str = test_dict["time_start"]
    end_time_str = test_dict["time_end"]

    # --- Efficiently aggregate rainfall data ---
    division_rainfall_totals = load_and_aggregate_rainfall_by_division(
        test_dict["time_start"],
        test_dict["time_end"],
        test_dict['rain_dir'],
        link_to_division_map
    )

    if not division_rainfall_totals:
        print("Warning: No rainfall data could be aggregated. Skipping rainfall map.")
        return

    # Convert aggregated data to DataFrame
    division_rainfall_rate = pd.DataFrame(list(division_rainfall_totals.items()), columns=['division_id', 'avg_rainfall_rate'])

    # --- Load shapefile and merge data ---
    gdf = gpd.read_file(shapefile_path)
    gdf.rename(columns={'LINKNO': 'link_id'}, inplace=True)
    gdf['link_id'] = gdf['link_id'].astype(int)

    division_map_df = pd.DataFrame(list(link_to_division_map.items()), columns=['link_id', 'division_id'])
    
    gdf_with_divisions = gdf.merge(division_map_df, on='link_id', how='inner')
    gdf_to_plot = gdf_with_divisions.merge(division_rainfall_rate, on='division_id', how='inner')

    if gdf_to_plot.empty:
        print("Warning: No matching geometries found for rainfall data. Skipping map generation.")
        return

    # --- Plotting ---
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    gdf_to_plot.plot(column='avg_rainfall_rate', ax=ax, legend=True,
                     cmap='Blues',
                     legend_kwds={'label': "Avg. Rainfall Rate by Division (mm/hr)",
                                  'orientation': "vertical",
                                  'shrink': 0.8})

    links_per_division = pd.Series(link_to_division_map).value_counts()
    dissolved = gdf_to_plot.dissolve(by='division_id', aggfunc={'avg_rainfall_rate': 'first'})
    dissolved['point'] = dissolved.geometry.representative_point()
    
    for division_id, row in dissolved.iterrows():
        point = row['point']
        rain_rate = row.get('avg_rainfall_rate')
        if pd.notna(rain_rate) and point:
            num_links = links_per_division.get(division_id, 1) # Default to 1 to avoid errors
            rate_density = rain_rate / num_links if num_links > 0 else 0
            annotation_text = f"Division: {division_id}\nAvg. Rate: {rain_rate:.2f} mm/hr\nDensity: {rate_density:.3f}"
            ax.text(point.x, point.y, annotation_text, fontsize=6, ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7, ec='none'))

    title = f"Spatially-Aggregated Average Rainfall Rate by Division\n({start_time_str} to {end_time_str})"
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_axis_off()
    plt.tight_layout()

    output_path = os.path.join(maps_dir, "total_rainfall_map.png")
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved rainfall map to {output_path}")


# ===================== Main Visualization Function =====================

def main_visualization(test_dict):
    """
    Main function to generate visualizations after the EKI algorithm finishes.
    """
    visual_output_dir = test_dict['out_dir'] + "visualization"    
    start_time_str = test_dict["time_start"]
    end_time_str = test_dict["time_end"]
    time_axis = pd.date_range(start=start_time_str, end=end_time_str, freq='H')
    num_assimilation_steps = test_dict["steps"]
    using_simulated_data = test_dict['using_simulated_data']
    rain_dir = test_dict['rain_dir']
    
    cr_ref_vec = None
    link_to_division_map = None
    model_link_ids_temp = get_ids(test_dict)
    if test_dict['watershed_csv']:
        sparse_parent_temp, link_to_division_map = get_subwatershed(test_dict, model_link_ids_temp)
        num_divisions = sparse_parent_temp.shape[0]
    else:
        num_divisions = 1 
        link_to_division_map = {link_id: 0 for link_id in model_link_ids_temp}
    
    if using_simulated_data:
        cr_ref_config = test_dict.get('Cr_ref')
        if cr_ref_config is not None:
            if isinstance(cr_ref_config, (int, float)):
                cr_ref_vec = np.full(num_divisions, float(cr_ref_config))
            elif isinstance(cr_ref_config, list) and len(cr_ref_config) == num_divisions:
                cr_ref_vec = np.array(cr_ref_config)
            else:
                 raise TypeError(f"Error: Cr_ref in config must be a number or a list, but got {type(cr_ref_config)}.")
    
    desired_usgs_ids = test_dict["plot_usgs"]
    usgs_2_id, _, file_order = load_usgs_mapping_from_path(test_dict["usgs_csv"], test_dict["link_sav"])
    
    plot_station_indices, plot_station_names, plot_link_ids = [], [], []
    for usgs in desired_usgs_ids:
        link_id = usgs_2_id.get(usgs)
        if link_id and np.where(file_order == link_id)[0].size > 0:
            plot_station_indices.append(np.where(file_order == link_id)[0][0])
            plot_station_names.append(usgs)
            plot_link_ids.append(link_id)
    
    observed_data = np.genfromtxt(os.path.join(test_dict['out_dir'], "csv/meas_mean.csv"), delimiter=',', skip_header=1)
    observed_data_clean = observed_data.copy()
    observed_data_clean[observed_data_clean == 0] = np.nan

    # Dynamically determine parameter labels, ranges, and active indices from config
    param_labels = test_dict['prm_names']
    prm_dist_bool = [str(val).lower() == 'true' for val in test_dict["prm_dist"]]
    active_param_indices = [i for i, is_active in enumerate(prm_dist_bool) if is_active]
    
    prm_lb = [float(lb) for lb in test_dict['prm_lb']]
    prm_ub = [float(ub) for ub in test_dict['prm_ub']]
    param_ranges = list(zip(prm_lb, prm_ub))


    # --- Post Assimilation ---
    post_param_list = []
    for i in range(num_assimilation_steps + 1):
        file_path = os.path.join(test_dict['out_dir'], f'npy/{i-1}_post_params_particles.npy') if i > 0 else os.path.join(test_dict['out_dir'], 'npy/0_prior_params_particles.npy')
        with open(file_path, 'rb') as f:
            post_param_list.append(np.load(f))
    post_param_array = np.stack(post_param_list, axis=0)
    iter_range_post = np.arange(0, num_assimilation_steps + 1)

    # --- Generate Maps (as they are independent of prior/post phase) ---
    generate_cr_map('post', visual_output_dir, test_dict['out_dir'], test_dict, active_param_indices, param_labels, post_param_array, cr_ref_vec)
    generate_hydrograph_metric_map('post', visual_output_dir, test_dict['out_dir'], test_dict, observed_data_clean,
                                   plot_station_indices, plot_station_names, model_link_ids_temp, plot_link_ids)
    generate_rainfall_map(visual_output_dir, test_dict, model_link_ids_temp, link_to_division_map)

    print("\n--- Generating Post-Assimilation Visualizations ---")
    generate_hydrograph_animation(num_assimilation_steps, plot_station_indices, plot_station_names, plot_link_ids,
                                  observed_data_clean, time_axis, start_time_str, end_time_str, rain_dir,
                                  using_simulated_data, cr_ref_vec, link_to_division_map,
                                  'post', visual_output_dir, test_dict['out_dir'])
    plot_parameter_evolution(post_param_array, active_param_indices, param_labels, param_ranges,
                             'post', visual_output_dir, iter_range_post, 
                             cr_ref_vec=cr_ref_vec)
    plot_parameter_evolution_consolidated(post_param_array, active_param_indices, param_labels, param_ranges,
                             'post', visual_output_dir, iter_range_post,
                             cr_ref_vec=cr_ref_vec)
    plot_event_statistics('post', visual_output_dir, test_dict['out_dir'], test_dict,
                          plot_station_indices, plot_station_names, plot_link_ids)


    # --- Prior Assimilation ---
    prior_param_list = []
    for i in range(num_assimilation_steps):
        file_path = os.path.join(test_dict['out_dir'], f'npy/{i}_prior_params_particles.npy')
        with open(file_path, 'rb') as f:
            prior_param_list.append(np.load(f))
    prior_param_array = np.stack(prior_param_list, axis=0)
    iter_range_prior = np.arange(0, num_assimilation_steps)

    print("\n--- Generating Prior-Assimilation Visualizations ---")
    generate_hydrograph_animation(num_assimilation_steps, plot_station_indices, plot_station_names, plot_link_ids,
                                  observed_data_clean, time_axis, start_time_str, end_time_str, rain_dir,
                                  using_simulated_data, cr_ref_vec, link_to_division_map,
                                  'prior', visual_output_dir, test_dict['out_dir'])
    plot_parameter_evolution(prior_param_array, active_param_indices, param_labels, param_ranges,
                             'prior', visual_output_dir, iter_range_prior, 
                             cr_ref_vec=cr_ref_vec)
    plot_parameter_evolution_consolidated(prior_param_array, active_param_indices, param_labels, param_ranges,
                             'prior', visual_output_dir, iter_range_prior,
                             cr_ref_vec=cr_ref_vec)
    plot_event_statistics('prior', visual_output_dir, test_dict['out_dir'], test_dict,
                          plot_station_indices, plot_station_names, plot_link_ids)

    plt.close('all')
    print("Visualization complete.")

if __name__ == '__main__':
    main_visualization()