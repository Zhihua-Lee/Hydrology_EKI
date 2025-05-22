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
