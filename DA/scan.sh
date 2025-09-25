#!/bin/bash

# ==============================================================================
#                 Parameter Sensitivity Scan Runner
# ==============================================================================
# This script is designed to run the sensitivity_scan.py tool.
# It sets up the necessary environment and executes the Python script, which
# in turn submits a job array to the HPC cluster to evaluate the model's
# sensitivity to the alpha parameter.
#
# Usage: ./run_sensitivity_scan.sh
# ==============================================================================

# --- Configuration ---
# Define key paths as variables for easier maintenance.
# Using $HOME is more portable than ~.
VENV_ACTIVATE="$HOME/virtenvs/Hydro_py3108/bin/activate"
PROJECT_ROOT="$HOME/DA/2025_EKI/DA"
PYTHON_EXEC="$HOME/virtenvs/Hydro_py3108/bin/python"
SCRIPT_TO_RUN="sensitivity_scan.py"
CONFIG_FILE="config.j2"


# --- Environment Setup ---
echo "INFO: Setting up the environment..."
# Activate the Python virtual environment.
source "${VENV_ACTIVATE}"
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate virtual environment at ${VENV_ACTIVATE}"
    exit 1
fi

# Load necessary environment modules.
module load openmpi
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to load the openmpi module."
    exit 1
fi
echo "INFO: Environment ready."


# --- Execution ---
# Navigate to the project root. This ensures all relative paths in the config
# and Python scripts are resolved correctly.
cd "${PROJECT_ROOT}"
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to change directory to project root: ${PROJECT_ROOT}"
    exit 1
fi

# Assemble the full command to be executed.
CMD="${PYTHON_EXEC} ${SCRIPT_TO_RUN} ${CONFIG_FILE}"

echo "----------------------------------------------------"
echo "INFO: Current working directory: $(pwd)"
echo "INFO: Executing sensitivity scan with command:"
echo "$CMD"
echo "----------------------------------------------------"

# Execute the command.
eval $CMD

echo "----------------------------------------------------"
echo "INFO: Script finished. The Python script has submitted jobs to the HPC."
echo "INFO: Use 'qstat' or check the output directory for results."