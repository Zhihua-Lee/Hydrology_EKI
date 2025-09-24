#!/bin/bash

# ==============================================================================
#                 HPC Job Submission Script for DA Orchestrator
# ==============================================================================
# This script submits the main DA framework's orchestrator script (run_da.py)
# as a job to the HPC scheduler (e.g., SGE). This single "head node" job will
# then manage the submission and coordination of the parallel simulation ensembles.
#
# USAGE: qsub submit_da.sh
# ==============================================================================

# --- SGE Job Scheduler Directives ---
#$ -N DA_Orchestrator        # Job name
#$ -j y                      # Join stdout and stderr
#$ -cwd                      # Run the job from the current working directory
#$ -l mf=8G                  # Request 8 GB of memory (adjust if needed)
#$ -q IFC                    # Submit to the IFC queue
#$ -m es                     # Send email at start and end of job
#$ -M your_email@uiowa.edu   # <<< REPLACE WITH YOUR EMAIL

# --- Environment and Execution ---
echo "========================================================"
echo "Job Started: $(date)"
echo "Running on host: $(hostname)"
echo "Job ID: $JOB_ID"
echo "========================================================"

# --- Environment Setup ---
# Activate the virtual environment
echo "Activating Python environment..."
source ~/virtenvs/Hydro_py3108/bin/activate
echo "Environment activated."

# Define the command to run the main DA script
# Using the absolute path to python executable is a robust practice
PYTHON_EXEC="~/virtenvs/Hydro_py3108/bin/python"
SCRIPT_PATH="~/DA/2025_EKI/DA/run_da.py"
CONFIG_FILE="~/DA/2025_EKI/DA/config.j2"
CMD="$PYTHON_EXEC $SCRIPT_PATH --config $CONFIG_FILE"

# Execute the orchestrator script
echo -e "\nExecuting DA Orchestrator with command:"
echo "$CMD"
echo "--------------------------------------------------------"
eval $CMD
echo "--------------------------------------------------------"


# --- Job Completion ---
echo "========================================================"
echo "Job Finished: $(date)"
echo "========================================================"