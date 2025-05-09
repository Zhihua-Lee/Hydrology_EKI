# Hydrology_EKI

## River network Visualization
The big file `usgs-basins.geojson` is stored in the archive `sangamon-cartopy.tar.gz`.

## Preliminary files

### Understanding the Watershed Division CSV File (`watershed.csv`)

This file provides spatial groupings for the river network links used in the hydrological model. It allows parameters (like `Cr`) to be estimated for groups of links (sub-watersheds or divisions) rather than for every individual link.

#### File Structure

The file is expected to be a Comma Separated Value (CSV) file with a header row.

* **Column 1:** Contains the unique `LinkID` for each river segment.
* **Subsequent Columns (e.g., `Div_Depth4`, `Div_Depth5`, ..., `Div_Depth8`):** Each of these columns represents a different level or "depth" of watershed divisions. The integer value in a specific `Div_DepthX` column for a given `LinkID` indicates the unique ID of the sub-watershed/division that the link belongs to *at that specific level (X) of divisions*.

**Example:**

```csv
LinkID,Div_Depth4,Div_Depth5,Div_Depth6,...
101,1,3,5,...
102,1,3,5,...
103,1,4,6,...
201,2,5,7,...
...
```

### Watershed Division Algorithm

This algorithm segments a river network into distinct divisions for different stream order thresholds. The core idea is to propagate division IDs upstream from the main channel(s), creating new divisions when significant, order-changing tributaries are encountered.

**For each specified stream order threshold (e.g., 4 through 8):**

1.  **Initialization:**
    * A unique set of division IDs is generated for this threshold.
    * The process identifies **root links**: these are the links within the dataset that possess the overall **highest stream order**.

2.  **Upstream Traversal (Breadth-First Search - BFS):**
    * An upstream BFS is initiated starting from these identified highest-order root link(s).
    * Each root link initially defines and is assigned a new, unique division ID (e.g., division 1, division 2, if there are multiple highest-order roots).

3.  **Division ID Propagation and Splitting:**
    As the BFS explores upstream from a processed link `L` (which has an assigned division ID) to one of its direct upstream parent links `U`:
    * **Split Condition (New Division for U):** If the stream order of `U` (`order(U)`) is different from the stream order of `L` (`order(L)`) AND `order(U)` is greater than or equal to the current **threshold**, then link `U` starts a *new* division. A new unique division ID is assigned to `U`.
    * **Inherit Condition (U belongs to L's Division):** If `order(U)` is the same as `order(L)`, OR if `order(U)` is less than the current **threshold**, then link `U` *inherits* the division ID from its downstream link `L`.

4.  **Output Generation:**
    * This process is repeated for each stream order threshold specified (e.g., resulting in columns `subw_4`, `subw_5`, ..., `subw_8`).
    * For each threshold, every link in the original dataset is assigned a division ID.
    * Links that are not reached by the BFS for a given threshold (e.g., they are part of very small disconnected headwaters not connected to the main BFS paths, or their own order and all upstream orders are below threshold without a chance to inherit) or do not qualify for a numbered division under the rules are typically assigned a default value like `0`.

This method results in a hierarchical division of the watershed, where the number and extent of divisions change based on the stream order threshold being considered.