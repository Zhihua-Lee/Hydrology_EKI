# Hydrology EKI/DA Framework

This repository hosts a scientific computing framework for hydrological modeling, with a focus on parameter estimation and data assimilation (DA). It is designed to support both simulated and real-world experiments to assess parameter identifiability, model performance, and data assimilation techniques.

The project is evolving from an Ensemble Kalman Inversion (EKI) based parameter estimation system into a broader, more flexible Data Assimilation framework.

## Project Structure

*   `DA/`: **(Future Work)** This directory is being developed to house a more general and extensible Data Assimilation framework, intended to support various DA methods beyond the initial EKI implementation.
*   `Inverse_Problem/`: Contains the original, mature application for solving an inverse problem using the Ensemble Kalman Inversion (EKI) method. Its primary use is to calibrate hydrological model parameters (like the runoff coefficient, `Cr`) by assimilating streamflow data. For a deep dive into this module, its scientific background, and specific run instructions, please see the `Inverse_Problem/README.md` file.
*   `sangamon/`: A dedicated module for handling geospatial data related to the Sangamon River Basin. It includes raw data, processing notebooks for spatial analysis (e.g., catchment mapping), and related visualizations.
*   `asynch/`: A submodule containing code for asynchronous processing. This is leveraged by the hydrological model to run simulations for different ensemble members in parallel, significantly speeding up experiments.
*   `.gitignore`: Specifies files and directories to be ignored by Git.

## Getting Started

### Prerequisites

Before you begin, ensure you have the following installed:
*   Python 3.x
*   Git
*   Key Python libraries: `numpy`, `pandas`, `scipy`, `matplotlib`, `geopandas`, `jinja2`. 

It is highly recommended to manage dependencies within a dedicated Python virtual environment.

### Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd 2025_EKI
    ```

2.  **Initialize the `asynch` submodule:**
    This project uses a submodule for parallel processing. Initialize it with:
    ```bash
    git submodule update --init --recursive
    ```

3.  **Extract geospatial data:**
    The large `usgs-basins.geojson` file, required for map visualizations, is archived to keep the repository size manageable. Extract it using:
    ```bash
    tar -xzvf sangamon-cartopy.tar.gz
    ```

### Next Steps

Once the setup is complete, you can proceed to the specific modules for detailed instructions:
*   To run a parameter estimation experiment, see the guide in `Inverse_Problem/README.md`.
