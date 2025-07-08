# Hydrology_EKI

## River network Visualization
The big file `usgs-basins.geojson` is stored in the archive `sangamon-cartopy.tar.gz`.

## Preliminary files

### Understanding the Watershed Division CSV File (`watershed_division_by_filtered_joints.csv`)

This file provides spatial groupings (sub-watersheds or divisions) for the river network links. This allows hydrological model parameters (like `Cr`) to be estimated for these groups rather than for every individual link, simplifying calibration and analysis.

#### File Structure

The file is a Comma Separated Value (CSV) file with a header row.

* **Column 1 (`LINKNO`):** Contains the unique identifier for each river segment in the network.
* **Subsequent Columns (e.g., `subw_4`, `subw_5`, ..., `subw_8`):** Each of these columns represents a set of sub-watershed divisions derived using a specific filtering criterion (based on stream order). The integer value in a `subw_X` column for a given `LINKNO` indicates the unique ID of the sub-watershed that the link belongs to when the network is processed with criterion `X`.

**Example:**

```csv
LINKNO,subw_4,subw_5,subw_6,subw_7,subw_8
101,1,1,2,2,3
102,1,1,2,2,3
103,1,2,3,3,4
201,2,3,4,5,6
...
```

---

### Watershed Division Algorithm

This algorithm segments a river network into distinct sub-watersheds. The process is repeated for several stream order-based filtering criteria, yielding different sets of divisions. The core mechanism involves identifying significant confluences ("joints") within a filtered portion of the network and then delineating new sub-watersheds upstream from these points.

**For each specified stream order threshold `i` (e.g., 4 through 8):**

1.  **Network Filtering:**
    * A temporary subset of the network, termed `n_filtered`, is created. This subset includes only those river links from the original network that have a stream order (`strmOrder`) greater than or equal to the current threshold `i`.

2.  **Identifying Significant Confluences (Joints):**
    * Within this `n_filtered` network, an "upstream accumulation" count (`cum_up`) is calculated for each link. This count signifies how many other links *also within `n_filtered`* flow into that particular link.
    * "Joints" are then identified as those links within `n_filtered` where this `cum_up` value is 3 or more. Such a condition typically indicates a confluence where at least two distinct upstream branches (which themselves meet the `strmOrder >= i` criterion) merge.

3.  **Identifying Split Points:**
    * The immediate upstream segments (specifically, those found in the `us1` and `us2` columns) connected to these identified "joints" are designated as potential starting points (LIDs or Link IDs) for new, distinct sub-watersheds.

4.  **Sub-watershed Delineation:**
    * **Initialization:** For the current threshold `i`, all links in the *entire* river network are initially assigned a base sub-watershed ID (typically, ID 1). The network is often sorted by a measure of size (like `DSContArea`) before this, so the main stem tends to form this initial base.
    * **Creating New Sub-watersheds:** For each unique "split point" LID identified in step 3:
        * A new, unique sub-watershed ID is generated (incrementing a counter).
        * An external function (`netf.get_subwatershed`) is called. This function takes the full network (for the current iteration `i`) and the "split point" LID as input. It is responsible for delineating the geographic extent of the new sub-watershed, presumably by traversing upstream from the given split point LID, and returns the list of all `LINKNO`s that belong to this newly defined sub-watershed.
        * All links returned by `netf.get_subwatershed` are then updated in the network data to reflect their membership in this new sub-watershed ID.

5.  **Handling Empty Filtered Networks:**
    * If, for a particular stream order threshold `i`, the `n_filtered` set is empty (meaning no links in the network meet the `strmOrder >= i` criterion), then all `LINKNO`s will be assigned a `NaN` (Not a Number) value for their sub-watershed ID in the output column corresponding to that specific threshold `i` (e.g., in `subw_i`).

**Output Generation:**

* The entire process (steps 1-5) is repeated independently for each stream order threshold `i` (e.g., from 4 to 8).
* The final output CSV file (`watershed_division_by_filtered_joints.csv`) contains one row for every `LINKNO` in the original network.
* The columns include `LINKNO`, followed by `subw_4`, `subw_5`, ..., `subw_8`. Each `subw_i` column stores the calculated sub-watershed ID for each `LINKNO` when the division process was run using the stream order threshold `i`.

This methodology results in multiple sets of watershed divisions, where the granularity and extent of the sub-watersheds vary depending on the stream order threshold used for filtering the network "joints."

---

### Visualization Results

This directory contains all the visualization outputs generated by `visualize.py` after the EKI process is complete. The results are organized into subdirectories based on the assimilation phase (`prior` or `post`) and the type of visualization. The following descriptions are framed within the context of solving an inverse problem, where we aim to estimate parameters ($\theta$) by fitting a model ($G$) to observations ($y$), such that $y \approx G(\theta)$.

#### `parameter/`

This folder visualizes the evolution of the estimated parameters ($\theta$) throughout the EKI iterations. In this problem, the parameter $\theta$ is treated as a vector, where each element corresponds to a specific geographic sub-watershed division. Therefore, the dimension of $\theta$ is the number of divisions (`num_divisions`). The visualizations analyze each dimension of this vector independently.

* **`mean_std/`**: Contains plots showing the evolution of the ensemble mean for each parameter element, with the shaded area representing one standard deviation. This is crucial for assessing **convergence**. A successfully solved problem should show the parameter mean converging to a stable value with decreasing uncertainty (i.e., the standard deviation shrinks). In simulated data experiments, a reference line (`Cr_ref`) for the true parameter value is also plotted, allowing for direct evaluation of estimation accuracy for each division.
* **`ensemble/`**: Contains plots showing the trajectories of all individual ensemble members for each parameter element. This provides a detailed view of the ensemble's behavior and helps diagnose issues like particle collapse or non-convergence.

#### `hydrograph/`

This folder contains the animations (`.gif` files) of the hydrographs, which represent the observation data ($y$) in our inverse problem. The plots are generated on a per-station basis.

Two types of gauge locations are typically used:
1.  **Data Assimilation (DA) Gauges**: These are the stations whose hydrograph data are directly used as the observation vector $y$ to drive the EKI updates. Often, this is a single gauge at the watershed outlet.
2.  **Verification Gauges**: These stations are not used in the assimilation process. Their data is used to independently verify or validate how well the calibrated parameters perform at other locations within the watershed.

A key feature of these plots is the inclusion of rainfall data. Since the parameter divisions are not necessarily aligned with the gauge locations, the visualization script correctly maps each station's `link_id` to its corresponding sub-watershed `division_id`. In simulated data experiments, this allows the script to scale the plotted rainfall by the correct `Cr_ref` value for that specific division, ensuring an accurate visual comparison.

**Inverse Problem Interpretation**:
A critical aspect to analyze here is the potential for **ill-posedness**. If the hydrograph at the DA gauge (the observation $y$) is fit very well by the model output ($G(\theta)$), but the estimated parameter vector $\theta$ does not converge to the true reference vector `cr_ref_vec` (in a simulated experiment), it strongly suggests that the inverse problem is ill-posed. This indicates that different combinations of parameters can produce very similar hydrographs at the outlet, pointing to a non-unique solution.

#### `event_statistics/`

This directory contains plots that visualize the evolution of key statistical metrics for discrete hydrological events. Instead of evaluating the model's performance across the entire time series, this approach provides a more granular assessment by focusing on how well the model reproduces the characteristics of individual high-flow events (e.g., storm hydrographs). This aligns directly with the event-based operator used during the EKI assimilation steps.

A core concept of the event operator in `eki.py` is that it transforms the raw hydrograph time series into a feature vector for the data assimilation process. Each identified event is described by **5 key metrics**:
1.  Peak Discharge
2.  Mean Discharge
3.  Standard Deviation of Discharge
4.  Timing of Event "Center of Mass" (Mean Time)
5.  Duration/Spread of Event (Std Dev of Time)

**Dimensionality of the Observation Vector (`y_event`)**
A key aspect of this method is its dynamic nature. The total dimension of the observation vector `y_event` used in the Kalman Filter update is not fixed; it depends on the number of events identified in the observation data. If the `find_events` function identifies **N** distinct events in the time series, the resulting feature vector will have a total dimension of **5 \* N**. This is created by concatenating the metrics from all `N` events into a single, long vector.

**Alignment of Predicted vs. Reference Events**
A crucial question in this process is how the simulated ("predicted") events are aligned with the observed ("reference") events. The user might ask: "Is the number of predicted events always the same as the reference? Does the DA process always predict an event where a reference one exists?"

The methodology in `eki.py` and `visualize.py` handles this elegantly and robustly:
1.  **Event windows are defined *exclusively* from the observed data.** The `find_events` algorithm is run **only** on the observation time series (`y`) to identify the start and end times (i.e., the index windows) for all `N` reference events.
2.  **Simulated metrics are calculated over these fixed windows.** The system does **not** try to find separate events in the simulated hydrograph. Instead, it takes the time windows defined by the observations and calculates the metrics for the model's output (`G(theta)`) within those *exact same windows*.

By design, this means the number of "predicted" events is **always identical** to the number of reference events. The data assimilation process is not trying to predict *if* an event occurs, but rather it is trying to match the characteristics of the model's output to the observed characteristics during the known time periods of real-world events.

**Description of the Plots**
The plots in this folder reflect this "evolution-style" analysis. For each station, a single figure is generated containing 5 subplots (one for each metric).
* The **x-axis** represents the EKI iteration number.
* Each **subplot** focuses on one metric (e.g., "Peak Discharge").
* Within a subplot, multiple **colored lines** show the evolution of the ensemble mean for that metric for each distinct event (Event 1, Event 2, etc.). The shaded area represents the ensemble's uncertainty (±1 standard deviation).
* A **dashed horizontal line** corresponding to each event's line shows the target "true" value calculated from the observed data.

This visualization directly shows how effectively the EKI process constrains the model to reproduce the specific characteristics of each individual storm over time.

#### `maps/`

This folder addresses a limitation of the previous plots: the lack of spatial context. While the parameter and hydrograph plots are viewed "by element" or "by station," the map visualization allows us to investigate the spatial patterns of the solution. The automated map generation is based on the logic from the `visualize_cr_map.ipynb` notebook.

The map displays the final calibrated `Cr` parameter values (from the last `post` assimilation iteration) on the river network, with each link color-coded according to its `Cr` value. This helps to answer questions about the spatial influence on the inverse problem's solution.

The map has two distinct modes:
1.  **Absolute Value Mode**: In a real-data experiment, the map shows the spatial distribution of the final calibrated `Cr` values. This can reveal which parts of the watershed require higher or lower runoff coefficients to match observations.
2.  **Error Mode**: In a simulated-data experiment (`using_simulated_data: true`), the map shows the **error** in the estimated parameter (`Cr_calibrated - Cr_ref`). This is a powerful diagnostic tool for identifying spatial biases in the parameter estimation. For example, it can show if the EKI systematically over- or underestimates parameters in certain geographic areas (e.g., upstream vs. downstream, or in specific tributaries), providing clues about the problem's structural challenges.
