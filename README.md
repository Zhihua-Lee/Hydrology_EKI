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