#!/bin/bash

# This is a "black box" debugging script for the HLM execution.
# It records the environment and inputs right before calling mpirun.

# Arguments from Python will be:
# $1: The directory to run in (the 'cwd' argument)
# $2: The path to the GBL file
# $3 onwards: The full mpirun command (e.g., mpirun -np 6 ...)

RUN_DIR="$1"
GBL_FILE="$2"
# Capture the rest of the arguments as the command to run
shift 2
MPIRUN_COMMAND=("$@")

# Define a unique log file path inside the run directory
LOG_FILE="${RUN_DIR}/debug_log.txt"

# Redirect all subsequent output (stdout and stderr) to the log file
exec &> "$LOG_FILE"

echo "================================================="
echo "HLM RUNNER DEBUG SCRIPT ACTIVATED"
echo "Time: $(date)"
echo "Host: $(hostname)"
echo "================================================="
echo

echo "--- 1. CURRENT WORKING DIRECTORY ---"
pwd
echo

echo "--- 2. FULL ENVIRONMENT VARIABLES ---"
env | sort
echo

echo "--- 3. GBL FILE CONTENT ---"
echo "Path: ${GBL_FILE}"
cat "${GBL_FILE}"
echo

echo "--- 4. EXECUTING HLM COMMAND ---"
echo "Command: ${MPIRUN_COMMAND[*]}"
echo

# Execute the actual command
# We still source the login environment to be safe
bash -l -c "${MPIRUN_COMMAND[*]}"

EXIT_CODE=$?

echo
echo "--- 5. EXECUTION FINISHED ---"
echo "Exit Code: ${EXIT_CODE}"
echo "================================================="

exit $EXIT_CODE