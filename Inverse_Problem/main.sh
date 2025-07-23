#!/bin/bash

# --- Configuration ---
# Set to true to skip the EKI run and only generate visualizations from existing output.
# Set to false to run the full EKI experiment.
VISUALIZE_ONLY=true
# VISUALIZE_ONLY=false

# IDAS or Argon doesn't allow user to run .sh files via ".\xxx.sh" directly;
# So just copy them to terminal in Argon and run

# (optional)activate the virtual environment in python
cd ~/virtenvs/Hydro_py3108/bin/
source ./activate
cd ~/DA/2025_EKI/Inverse_Problem/

# Check qstat -u zli333 before submitting jobs, if there are remaining, use qdel to delete them
qstat -u zli333

# Check available slots before running code, if the slots available is less than num_ensemble * num_parallel_slots, then we need to set num_parallel_slots to be smaller; you should be able to monitor the job status also using this command
qstat -f -q IFC -u zli333

# Modify the parameters in `config.j2` as desired

# --- Execution ---
CMD="~/virtenvs/Hydro_py3108/bin/python ~/DA/2025_EKI/Inverse_Problem/eki_test.py ~/DA/2025_EKI/Inverse_Problem/config.j2"

if [ "$VISUALIZE_ONLY" = true ]; then
  CMD="$CMD --visualize-only"
fi

# Submit the job to Argon to run
echo "Executing command: $CMD"
eval $CMD

# Cr scan job to plot Cr-hydrograph curve
# ~/virtenvs/Hydro_py3108/bin/python ~/DA/2025_EKI/Inverse_Problem/cr_scan.py ~/DA/2025_EKI/Inverse_Problem/config.j2
