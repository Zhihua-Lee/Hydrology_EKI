#!/usr/bin/python
import os
import sys
import numpy as np
import pickle

# Add the parent directory to the Python path to allow for module imports
# This assumes the script is run from the Inverse_Problem directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from latent import transform_latent_to_physical
from io_ifc import create_prm_from_division_params

def main():
    """
    Main worker function executed by each HPC job in the array.
    
    This script reads its task ID from an environment variable, loads shared data,
    and generates a single .prm file for its assigned ensemble member.
    """
    # 1. Get the ensemble member ID from the HPC environment variable
    # SGE/UGE uses SGE_TASK_ID, Slurm uses SLURM_ARRAY_TASK_ID. We default to SGE.
    try:
        task_id = int(os.environ.get('SGE_TASK_ID', os.environ.get('SLURM_ARRAY_TASK_ID')))
        # Job array tasks are typically 1-based, so convert to 0-based index
        member_id = task_id - 1
    except (ValueError, TypeError):
        print("Error: Could not get a valid task ID from environment variables.")
        sys.exit(1)

    # The main script should have passed the temporary directory path as an argument
    if len(sys.argv) < 2:
        print("Error: Path to temporary directory not provided.")
        sys.exit(1)
    tmp_dir = sys.argv[1]

    # 2. Load the shared data that was prepared by the main script
    try:
        with open(os.path.join(tmp_dir, 'prm_job_data.pkl'), 'rb') as f:
            shared_data = pickle.load(f)
        
        X_ensemble = np.load(os.path.join(tmp_dir, 'X_ensemble.npy'))

        test_dict = shared_data['test_dict']
        link_to_division_map = shared_data['link_to_division_map']
        n_divisions = shared_data['n_divisions']

    except FileNotFoundError as e:
        print(f"Error: Could not load required data file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during data loading: {e}")
        sys.exit(1)

    # 3. Perform the core task for this specific ensemble member
    try:
        # Transform latent params to physical params for this member
        physical_params_div = transform_latent_to_physical(
            test_dict,
            X_ensemble[:, :, member_id],
            n_divisions=n_divisions
        )
        
        # Create the final .prm file
        create_prm_from_division_params(
            test_dict,
            link_to_division_map,
            physical_params_div,
            member_id
        )
    except Exception as e:
        # It's crucial to catch and report errors from the worker
        print(f"Error during PRM generation for member {member_id}: {e}")
        # Optionally, write to a specific error file for easier debugging
        with open(os.path.join(tmp_dir, f'{member_id}.prm.error'), 'w') as f:
            f.write(str(e))
        sys.exit(1)

    # The script will exit with 0 if successful

if __name__ == "__main__":
    main()
