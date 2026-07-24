"""This file contains two connection functions. Both are intended to read the .parquet files created in 
genrate_dataset.py, do some basic pre-processing, and return the files as numpy arrays. These arrays
can then be fed as training data to phase 2 or phase 3 models, depending on the dataset. The phase 2
input is data/processed/combined_sample.parquet, and the phase 3 training is located at data/processed/flow_training_sample.parquet
This file should be run before making any train/test split"""
#read the wwt
#implement any common preprocessing from etta and davids notebooks
#return a numpy array
import sys
sys.path.append('..')
import numpy as np
from numpy.typing import NDArray
from pandas import read_parquet
import pandas as pd
from scripts.dataImport import settings
from pathlib import Path

def _load_parquet(path:Path, file_name) -> NDArray:
    """load .parquet file and convert to NDArray. pd.DataFrame as intermediary.
    """
    file = path / file_name
    df = read_parquet(file)
    array = df.to_numpy()
    return array
#For Mack organization
def etta_preprocess(array: NDArray, to_drop:list[str], log_features:list = ["stream_jitter_1_var"]):
    """For Mack's organization. I've also make comments in Etta's notebook for my organization. Ignore at your leisure"""
    df = pd.DataFrame(array)
    numeric_features = (
        df
        .select_dtypes(include=np.number)
        .columns
        .to_list()
    )
    numeric_features = [
        feat for feat in numeric_features
        if feat not in to_drop
    ]
    if len(numeric_features) == 0:
        raise ValueError("No numeric features found.")

    df = (
        df
        .reindex(columns=numeric_features)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .astype(np.float32) #This is different than etta's (float64). Because generate_dataset already reduces the file to float32, so increasing to float64 is unnecessary
    )
    for feat in log_features:
        if feat in df.columns:
            df[feat] = np.log1p(df[feat].clip(lower=0))

    processed_array = df.to_numpy()
    return processed_array



def david_preprocess(file_paths: dict, sample_rows: int|None = None):
    label_map = {
        "benign": "Benign", "benign_1": "Benign", "benign_2": "Benign", "benign_3": "Benign",
        "ddos": "DDoS", "dos": "DoS", "dos_1": "DoS",
        "dns_spoofing": "DNS_Spoofing", "brute_force": "BruteForce", "xss": "XSS",
    }

    id_cols = [
        "stream", "src_mac", "dst_mac", "src_ip", "dst_ip",
        "src_port", "dst_port", "device_mac", "eth_src_oui", "eth_dst_oui",
    ]

    dfs = []
    for label, path in file_paths.items():
        df = pd.read_csv(path, nrows=sample_rows)
        df["Label"] = label_map.get(label, label)
        dfs.append(df)
        print(f"Loaded {label}: {df.shape}")

    data = pd.concat(dfs, ignore_index=True)

    # Drop ID columns
    data = data.drop(columns=[c for c in id_cols if c in data.columns], errors="ignore")

    # Convert all non-Label columns to numeric, non-numeric values become 0
    feature_cols = [c for c in data.columns if c != "Label"]
    for col in feature_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data[feature_cols] = data[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    print(f"\nCombined shape: {data.shape}")
    print(data["Label"].value_counts())

    return data
def isolation_preprocess():
    pass


def _preprocess_combined(array:NDArray) -> NDArray:
    """Currently does nothing"""
    processed_array = array
    return processed_array

def _preprocess_flow_tr_sample(array:NDArray) -> NDArray:
    """Currently does nothing"""
    processed_array = array
    return processed_array
