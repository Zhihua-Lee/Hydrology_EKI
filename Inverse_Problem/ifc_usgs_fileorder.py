import numpy as np
from typing import List, Tuple, Dict, Union
import os
import pandas as pd

#TODO: convert this into something that can be stored in a seperate file, so that it can be generalized

def load_usgs_mapping(test_dict: dict):
    """
    从 JSON 配置中指定的 CSV 文件中读取 USGS 测站与 link id 的映射关系，
    并生成 usgs_2_id、id_2_usgs 和 file_order 数组。

    配置文件中需要包含键 "usgs_csv"，指向 CSV 文件路径，
    CSV 文件中必须有 'STAID' 列和 'LINKNO' 列。

    Returns:
        tuple: (usgs_2_id, id_2_usgs, file_order)
    """
    # 从 test_dict 中获取 CSV 文件路径
    usgs_csv_path = test_dict.get("usgs_csv")
    gauges_lid_sav_path = test_dict.get("link_sav")
    if not usgs_csv_path or not os.path.exists(usgs_csv_path):
        raise FileNotFoundError(f"USGS mapping CSV file not found: {usgs_csv_path}")
    
    return load_usgs_mapping_from_path(usgs_csv_path, gauges_lid_sav_path)

def load_usgs_mapping_from_path(usgs_csv_path: str, gauges_lid_sav_path:str) -> Tuple[Dict[str, int], Dict[int, str], np.ndarray]:
    # 读取 CSV 数据，假设 'STAID' 为测站号，'LINKNO' 为 link id
    df = pd.read_csv(usgs_csv_path, dtype=str).set_index('STAID')
    
    # 构造 USGS 到 link id 的映射（转换为整数）
    usgs_2_id = df['LINKNO'].astype(int).to_dict()
    
    # 反向映射：link id 到 USGS
    id_2_usgs = {v: k for k, v in usgs_2_id.items()}

     # 从 .sav 文件中读取 file_order（即 link id 顺序）
    if not os.path.exists(gauges_lid_sav_path):
        raise FileNotFoundError(f"meas_sav file not found: {gauges_lid_sav_path}")
    with open(gauges_lid_sav_path, 'r') as f:
        lines = f.readlines()
    # 提取每行的第一个数字（link id），跳过空行
    file_order = np.array([int(line.strip().split()[0]) for line in lines if line.strip() != ""])

    
    # # 定义 file_order，保持 CSV 中 'LINKNO' 的顺序
    # file_order = df['LINKNO'].astype(int).values

    return usgs_2_id, id_2_usgs, file_order
