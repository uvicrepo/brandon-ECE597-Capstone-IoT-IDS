"""Hand off between phase 2 and phase 3 models."""
import sys
sys.path.append('..')
import numpy as np
from numpy.typing import NDArray

def filter_drop(array:NDArray, label_col_idx:int) -> NDArray:
    """Filter to rows classified as attack by phase 2 model based on
    column in pos label_col_idx. Expects attack=1 and benign=0. Drop this column. 
    Return reshaped dataset. 
    
    If reading from parquet you can use _load_parquet from phase2preprocess
    like: \\
    array = filter_drop(array=_load_parquet(YOUR_PATH_HERE)..)
    """
    #Filter
    attack_condition = array[:, label_col_idx] == 1
    attack_array = attack_condition[attack_condition]
    #Remove
    attack_array = np.hstack((attack_array[:, :label_col_idx], attack_array[:, label_col_idx+1:]))
    return attack_array

