# Data Assets

Large forcing data and auxiliary geospatial archives are intentionally kept out
of Git. Publish them as GitHub Release assets and extract them at the repository
root before running full experiments.

Expected restored paths include:

```text
Inverse_Problem/hlm_data/Simulated_data/Rainfall_Sangamon_river/
sangamon/usgs-basins.geojson
```

Small model-structure files and parameter templates remain tracked under
`Inverse_Problem/hlm_data/`.
