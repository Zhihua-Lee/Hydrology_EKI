# File: DA/generate_da_input_worker.py

import os
import sys
import numpy as np
import pickle

# This worker needs access to the file creation utilities
from io_ifc import create_prm_from_division_params, write_rec_file

def main():
    """
    Main worker for parallel DA input file generation (Forecast Step).
    - Creates BOTH the .prm and .rec files for its assigned ensemble member.
    """
    try:
        task_id = int(os.environ.get('SGE_TASK_ID', os.environ.get('SLURM_ARRAY_TASK_ID')))
        member_id = task_id - 1
    except (ValueError, TypeError):
        sys.exit("Error: Could not get a valid task ID.")

    if len(sys.argv) < 2:
        sys.exit("Error: Path to temporary directory not provided.")
    tmp_dir = sys.argv[1]
    forecast_files_dir = os.path.join(tmp_dir, 'forecast_files')

    try:
        with open(os.path.join(forecast_files_dir, 'da_job_data.pkl'), 'rb') as f:
            shared_data = pickle.load(f)
        
        q_ensemble = np.load(os.path.join(forecast_files_dir, 'q_ensemble.npy'))
        alpha_ensemble = np.load(os.path.join(forecast_files_dir, 'alpha_ensemble.npy'))

        config = shared_data['config']
        link_to_division_map = shared_data['link_to_division_map']
        n_divisions = shared_data['n_divisions']
        cr_param_index = shared_data['cr_param_index']
        sorted_link_ids = shared_data['sorted_link_ids']
        constant_states = shared_data['constant_states']
        
        q_discharge = q_ensemble[member_id]
        alpha_param = alpha_ensemble[member_id]

    except Exception as e:
        sys.exit(f"Error loading data for member {member_id}: {e}")

    run_dir = os.path.join(forecast_files_dir, str(member_id))
    os.makedirs(run_dir, exist_ok=True)
    
    try:
        # 1. CREATE THE .prm FILE
        prm_path = os.path.join(run_dir, "params.prm")
        cr_params = np.full((1, n_divisions), alpha_param)
        create_prm_from_division_params(
            config['hlm_model'], link_to_division_map, cr_params, [cr_param_index], prm_path
        )
        
        # 2. CREATE THE .rec FILE
        rec_path = os.path.join(run_dir, "state.rec")
        n_links = len(sorted_link_ids)
        full_state_matrix = np.zeros((n_links, 5))
        full_state_matrix[:, 0] = q_discharge
        # constant_states is a 1D array of 4 values, tile it to (n_links, 4)
        full_state_matrix[:, 1:] = np.tile(constant_states, (n_links, 1))
        
        write_rec_file(rec_path, config['hlm_model']['model_num'], sorted_link_ids, full_state_matrix)

    except Exception as e:
        error_msg = f"Error during file generation for member {member_id}: {e}"
        with open(os.path.join(forecast_files_dir, f'{member_id}.input.error'), 'w') as f:
            f.write(error_msg)
        sys.exit(error_msg)

if __name__ == "__main__":
    main()