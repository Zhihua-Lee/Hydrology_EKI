#!/bin/bash

# ==============================================================================
#                 Sequential Data Assimilation (DA) Runner
# ==============================================================================
# This script is designed to run the main DA framework on hpc.
# It activates the specified Python environment and executes the main run_da.py script.
#
# Note for HPC Users:
# This script runs the DA loop sequentially on a single node. It does NOT submit
# jobs to a scheduler like SGE or Slurm. Before running on an HPC login node,
# ensure this type of computation is permitted by your cluster's policy.
# ==============================================================================


# --- Environment Setup ---
# (Optional) Activate the virtual environment in Python
echo "Setting up the environment..."
clear
cd ~/virtenvs/Hydro_py3108/bin/
source ./activate

# Load necessary environment modules for local execution on the head node.
# This is CRITICAL for ensuring that commands like 'mpirun' are available to the Python script.
module load openmpi

# Navigate to the project's root directory. The script expects to be run from here.
cd ~/DA/2025_EKI/DA


# --- Execution ---
# Define paths for clarity
DA_PATH="~/DA/2025_EKI/DA/"
PYTHON_EXEC="~/virtenvs/Hydro_py3108/bin/python"
SCRIPT_PATH="run_da.py"
# The --config argument should be the filename within the DA/ directory
CONFIG_FILE="config.j2"

# Assemble the full command
CMD="$PYTHON_EXEC $DA_PATH$SCRIPT_PATH --config $DA_PATH$CONFIG_FILE"

# Execute the command
echo -e "\nExecuting DA framework with the following command:"
echo "$CMD"
echo "----------------------------------------------------"
eval $CMD

echo "----------------------------------------------------"
echo "Script execution finished."

# # Alpha scan job to plot Alpha-hydrograph curve
# ~/virtenvs/Hydro_py3108/bin/python ~/DA/2025_EKI/DA/sensitivity_scan.py ~/DA/2025_EKI/DA/config.j2