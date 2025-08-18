from dateutil import parser
from scipy.sparse import coo_matrix
from typing import List, Tuple, Dict, Union
import numpy as np
import os
import yaml
import pandas as pd
from jinja2 import Environment, FileSystemLoader

## Utility functions

def process_yaml(yamlname: str) -> dict:
    """
    Reads a YAML configuration file or a Jinja2 template file and returns a config dictionary.
    If the file extension is .j2, it first renders the template (using context for variable substitution),
    then loads the rendered YAML. Otherwise, it directly loads the YAML file.
    
    Args:
        yamlname (str): The path to the YAML or Jinja2 template configuration file.
        
    Returns:
        dict: The configuration dictionary.
    """
    ext = os.path.splitext(yamlname)[1].lower()
    
    if ext == ".j2":
        print(".j2")
        # Get the directory and filename
        directory = os.path.dirname(yamlname) or "."
        filename = os.path.basename(yamlname)
        # Set up Jinja2 environment and render the template
        env = Environment(loader=FileSystemLoader(directory))
        template = env.get_template(filename)
        rendered = template.render()
        config = yaml.safe_load(rendered)
    else:
        with open(yamlname, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    return config

def process_json(json_name: str) -> dict:
    """
    Read and parse a JSON file.

    Args:
        json_name (str): Name of the JSON file.

    Returns:
        dict: Parsed JSON data as a dictionary.
    """
    with open(json_name) as f:
        test_dict = json.load(f)
    return test_dict

def time_to_epoch(time: str) -> float:
    """
    Convert a time string to epoch timestamp.

    Args:
        time (str): Time string in any valid format.

    Returns:
        float: Epoch timestamp representing the given time.
    """
    return parser.parse(time).timestamp()
