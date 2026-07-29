"""Hand off between phase 2 and phase 3 models."""
import sys
sys.path.append('..')
import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame

def filter_drop(df:DataFrame, label_col_idx:int) -> NDArray:
    """Filter to rows classified as attack by phase 2 model based on
    column in pos label_col_idx. Expects attack=1 and benign=0. Drop this column. 
    Return reshaped dataset. 
    
    If reading from parquet you can use _load_parquet from phase2preprocess
    like: \\
    array = filter_drop(array=_load_parquet(YOUR_PATH_HERE)..)
    """
    if isinstance(df, DataFrame):
        array = df.to_numpy()
    #Filter
    attack_condition = array[:, label_col_idx] == "Attack"
    attack_array = array[attack_condition]
    #Remove
    clean_array = np.delete(attack_array, label_col_idx, axis=1)
    return clean_array

