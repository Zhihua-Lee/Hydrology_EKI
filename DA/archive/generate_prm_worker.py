# File: DA/generate_da_input_worker.py

import os
import sys
import numpy as np
import pickle

# This worker needs access to the file creation utilities
from io_ifc import create_prm_from_division_params
from hlm_runner import HLMRunner # We'll borrow its .uini writing method

def main():
    """
    Main worker function for parallel DA input file generation.
    - Reads its member ID from the environment.
    - Loads shared data (config, maps) and member-specific data (q_initial, alpha).
    - Creates the .prm and .uini files for its assigned ensemble member.
    """
    try:
        task_id = int(os.environ.get('SGE_TASK_ID', os.environ.get('SLURM_ARRAY_TASK_ID')))
        member_id = task_id - 1
    except (ValueError, TypeError):
        print("Error: Could not get a valid task ID from environment variables.")
        sys.exit(1)

    if len(sys.argv) < 2:
        print("Error: Path to temporary directory not provided.")
        sys.exit(1)
    tmp_dir = sys.argv[1]

    # Load shared and member-specific data
    try:
        with open(os.path.join(tmp_dir, 'da_job_data.pkl'), 'rb') as f:
            shared_data = pickle.load(f)
        
        q_ensemble = np.load(os.path.join(tmp_dir, 'q_ensemble.npy'))
        alpha_ensemble = np.load(os.path.join(tmp_dir, 'alpha_ensemble.npy'))

        config = shared_data['config']
        link_to_division_map = shared_data['link_to_division_map']
        n_divisions = shared_data['n_divisions']
        cr_param_index = shared_data['cr_param_index']
        sorted_link_ids = shared_data['sorted_link_ids']
        
        # Get data for this specific member
        q_initial = q_ensemble[member_id]
        alpha_param = alpha_ensemble[member_id]

    except Exception as e:
        print(f"Error loading data for member {member_id}: {e}")
        sys.exit(1)

    # Perform the file generation task
    run_dir = os.path.join(tmp_dir, str(member_id))
    os.makedirs(run_dir, exist_ok=True)
    
    try:
        # 1. Create .prm file
        prm_path = os.path.join(run_dir, "params.prm")
        cr_params = np.full((1, n_divisions), alpha_param)
        create_prm_from_division_params(
            config['hlm_model'], link_to_division_map, cr_params, [cr_param_index], prm_path
        )
        
        # 2. Create .uini file
        # We can't instantiate a full HLMRunner, so we borrow its static method logic.
        # This is a bit of a trick to avoid re-writing the same code.
        uini_path = os.path.join(run_dir, "initial.uini")
        with open(uini_path, 'w') as f:
            f.write(f"{len(sorted_link_ids)}\n")
            for link_id, state_val in zip(sorted_link_ids, q_initial):
                f.write(f"{link_id}\n")
                f.write(f"{state_val:.6f}\n")

    except Exception as e:
        print(f"Error during file generation for member {member_id}: {e}")
        with open(os.path.join(tmp_dir, f'{member_id}.input.error'), 'w') as f:
            f.write(str(e))
        sys.exit(1)

if __name__ == "__main__":
    main()