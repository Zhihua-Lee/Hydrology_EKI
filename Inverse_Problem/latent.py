import json
import numpy as np
from typing import List, Tuple, Dict, Union

def convert_logical(str_list: list) -> list:
    """
    Convert a list of strings representing logical values to a list of Python booleans.

    Args:
        str_list (list): A list of strings representing logical values ("true" or "false").

    Returns:
        list: A list of Python booleans (True or False) corresponding to the logical values.
    """
    # Convert each string element in the input list to lower case and parse it as a JSON value
    # This will convert strings "true" and "false" to Python booleans True and False, respectively
    return [json.loads(i.lower()) for i in str_list]
    

def create_latent(test_dict: dict, sparse_parent: np.ndarray, ens: int) -> np.ndarray:
    """
    Create a matrix of latent parameter values based on the given test dictionary and sparse parent information.

    Args:
        test_dict (dict): Test dictionary containing required parameters.
        sparse_parent (np.ndarray): Sparse parent information.
        ens (int): Number of parameter ensembles to create.

    Returns:
        np.ndarray: A matrix of latent parameter values with shape (total_parameters, ens).
    """
    
    # Extract required parameters from the test dictionary
    include_parameters = convert_logical(test_dict["prm_dist"])

    # Calculate the number of parent points and total parameters
    parent_num = sparse_parent.shape[0]
    total_active = np.sum(include_parameters) #Total number of included prm values
    tot_param_num = total_active * parent_num

    # Create an empty matrix to store the latent parameter values
    latent_mat = np.zeros((tot_param_num, ens))
    loc = 0  # Index to keep track of the current position in the latent matrix

    # Generate latent parameter values based on the distribution type
    for i, dist in enumerate(include_parameters):
        if dist:  # Check if the parameter is included
            # Generate random latent parameter values from a standard normal distribution
            lv = np.random.normal(0, 1, (parent_num, ens))
            latent_mat[loc:loc + parent_num, :] = lv
            loc = loc + parent_num

    # 检查是否出现 NaN
    if np.isnan(latent_mat).any():
        print("Warning: NaN found in create_latent's latent_mat!")

    return latent_mat

def unbounded_to_bounded(x: np.ndarray, lb: float, ub: float) -> np.ndarray:
    """
    Convert unbounded values to bounded values using a sigmoid transformation.

    Args:
        x (np.ndarray): Input array of unbounded values.
        lb (float): Lower bound of the desired bounded range.
        ub (float): Upper bound of the desired bounded range.

    Returns:
        np.ndarray: An array of bounded values mapped to the range [lb, ub].
    """
    # TODO: Include other transformed parameter distributions
    # Apply a sigmoid transformation to map the unbounded values to the range [0, 1]
    x_on_0_1 = (np.tanh(x) + 1) / 2.0

    # Scale and shift the values to the desired bounded range [lb, ub]
    res = lb + (x_on_0_1) * (ub - lb)

    # 检查是否出现 NaN
    if np.isnan(res).any():
        print(f"Warning: NaN found in unbounded_to_bounded for lb={lb}, ub={ub}!")
        # 可选：你也可以在此处对 NaN 进行填补或截断

    return res


def transform_latent(test_dict: dict, sparse_parent: np.ndarray, latent_var: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transform latent variables to parameter ensembles based on the given test dictionary and sparse parent data.

    Args:
        test_dict (dict): Test dictionary containing required parameters.
        sparse_parent (np.ndarray): Array representing the sparse parent data.
        latent_var (np.ndarray): Array of latent variables to transform.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing transformed parameter ensembles (prm_ens) and 
                                       the list of sorted IDs (id_list).
    """
    #TODO: get to work with more (or less) prm numbers for different model numbers
    TOTAL_PRM_NUM = len(test_dict["prm_dist"])
    # TOTAL_608_PRM_NUM = 18
    
    # Read template PRM file and extract data
    prm_name = test_dict['prm']
    with open(prm_name, 'r') as f:
        prm_lines = [line for line in f.readlines() if line.strip()]
    id_list_prm = [int(i.strip('\n')) for i in prm_lines[1::2]]
    prm_list = np.array([[float(i) for i in line.strip('\n').split()] for line in prm_lines[2::2]]) # shape: (link_num, TOTAL_PRM_NUM)
    
    # Sort by ascending ID number and get total IDs
    id_list_arg = np.argsort(id_list_prm)
    prm_array = prm_list[id_list_arg,:] # sorted hlm prm_list, shape: (link_num, TOTAL_PRM_NUM)
    id_list = np.sort(id_list_prm)
    id_num = len(id_list)  
    ens = latent_var.shape[1]
   
    # Create a ensemble of parameter matricies
    prm_ens = np.zeros((TOTAL_PRM_NUM,id_num,ens)) # EKI parameters, shape: (TOTAL_PRM_NUM,link_num,ens)
    out_num = sparse_parent.shape[0]

    # Get parameters for transformation
    include_parameters = convert_logical(test_dict["prm_dist"])
    lower_bounds = test_dict['prm_lb']
    upper_bounds = test_dict['prm_ub']

    loc = 0
    for i, dist in enumerate(include_parameters):
        if dist is True: # Check if using paramters
            # Define bounds
            lb = float(lower_bounds[i])
            ub = float(upper_bounds[i])
            
            # Convert from sparse to full space
            lv = (sparse_parent.T)@latent_var[loc:(loc+out_num),:] # shape: (out_num, ens).T * (id_num, out_num)
            
            #Apply transformation, and round to 5 digits (necessary for asynch, otherwise will fail)
            transform_func = lambda x: float(
                np.format_float_positional(
                    unbounded_to_bounded(x, lb, ub), # using $\tanh$ like functions to map any real num x to [lb, ub]
                    precision=5, unique=False, fractional=False, trim='k'
                )
            )
            vec_transform_func = np.vectorize(transform_func)
            var_val = vec_transform_func(lv)

            # NaN checking
            if np.isnan(var_val).any():
                print(f"Warning: NaN in transform_latent for parameter index={i}, lb={lb}, ub={ub}!")

            prm_ens[i,:,:] = var_val
            
            #Advance forward
            loc = loc + out_num
        else:   # otherwise, leave alone
            for j in range(ens):
                prm_ens[i,:,j] = prm_array[:,i]
    return prm_ens, id_list

def transform_latent_sparse(test_dict: dict, sparse_parent: np.ndarray, latent_var: np.ndarray) -> np.ndarray:
    """
    Transform sparse latent variables to bounded parameter ensemble based on the given test dictionary and sparse parent data.

    Args:
        test_dict (dict): Test dictionary containing required parameters.
        sparse_parent (np.ndarray): Array representing the sparse parent data.
        latent_var (np.ndarray): Array of sparse latent variables to transform.

    Returns:
        np.ndarray: Transformed parameter ensemble (prm_ens) with bounds applied.
    """
    out_num = sparse_parent.shape[0]
    sparse_transformed = np.zeros(latent_var.shape)

    include_parameters = convert_logical(test_dict["prm_dist"])
    lower_bounds = test_dict['prm_lb']
    upper_bounds = test_dict['prm_ub']

    loc = 0
    for i, dist in enumerate(include_parameters):
        if dist is True:
            # Define bounds
            lb = float(lower_bounds[i])
            ub = float(upper_bounds[i])
            
            # Applies transformation
            lv = latent_var[loc:(loc + out_num), :]
            tmp = unbounded_to_bounded(lv, lb, ub)
            if np.isnan(tmp).any():
                print(f"Warning: NaN in transform_latent_sparse for param i={i}, lb={lb}, ub={ub}!")
            sparse_transformed[loc:loc+out_num, :] = tmp
            loc = loc + out_num
    return sparse_transformed