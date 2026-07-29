"""
This file contains phase 1 -> phase 2 connection functions. load_parquet() reads the .parquet files 
created in generate_dataset.py. All preprocessing is done by preprocess_general(). The preprocessing 
order should be: \\
1. load data/processed/combined_sample.parquet and data/processed/flow_training_sample.parquet into 
two seperate np.recarray (form of NDArray with column labels) with load_parquet()\\
2. split combined_sample into training and testing data. train_test_split from sklearn.model_selection
is recommended. \\
3. seperate out labels, as typical in ML processes\\
4. Run preprocess_general(split='Train) on the training split of combined_sample. This will preprocess
the data, but critically save the mean and std of numeric features to settings/training_preprocess_states.joblib, 
along with one-hot-encodings of non-numeric columns. 
5. Run preprocess_general(split='Flow') on flow_training_sample. This will apply the same mean and std 
from the combined_sample training set to normalize numeric features. The reasoning behind this is that 
the testing split, which is what we will treat as unseen data, will also be normalized with the mean and 
std learned from the combined_sample training set. Therefore, it makes sense to train the phase 3 model 
on data which has been normalized the same way.\\
6. Run preprocess_general(split='Test') on the testing split of combined_sample. Since this may happen in
a completly new notebook ketnel, preprocess_general is designed to import the mean and std from the .jooblib
saved to the settings dir.\\
"""
import sys
sys.path.append('..')
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import tomllib
import joblib

import numpy as np
from pandas import read_parquet
import pandas as pd
from scripts.dataImport import impsettings as ipsettings
from sklearn.preprocessing import OneHotEncoder

@dataclass
class PreProcessSettings:
    DROP_FEATURES:list[str]
    LABEL_FEATURES:list[str]
    FLOW_DROP_FEATURES:list[str]
    FLOW_LABEL_FEATURES:list[str]
    encoder: Optional[OneHotEncoder] = field(default=None, init=False)
    std: Optional[pd.Series] = field(default=None, init=False)
    mean: Optional[pd.Series] = field(default=None, init=False)

    @classmethod
    def fromTOML(cls, path: Path = Path(__file__).resolve().parent.parent.parent / "settings" / "preprocess.toml") -> "PreProcessSettings":
        with open(path, 'rb') as f:
            data = tomllib.load(f)
        return cls(**data)

    def fit_encoder(self, X_train_categorical:pd.DataFrame):
        """Fit the encoder strictly on training data."""
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.int8)
        self.encoder.set_output(transform="pandas")
        self.encoder.fit(X_train_categorical)

    def fit_scaler(self, df_train_numeric: pd.DataFrame):
        """Calculate and store mean and std strictly on training data."""
        self.mean = df_train_numeric.mean()
        self.std = df_train_numeric.std().replace(0, 1.0) #prevent 0 division on 0 variance features

    def save_state(self, path: Path = Path(__file__).resolve().parent.parent.parent / "settings" / "training_preprocess_states.joblib"):
        """Save the runtime state (mean, std, encoder) to a joblib file."""
        if self.mean is None or self.std is None:
            raise RuntimeError("Cannot save state: 'mean' and 'std' have not been fitted. Tip: compute training data first")
        state = {
                "mean": self.mean,
                "std": self.std,
                "encoder": self.encoder
            }
        joblib.dump(state, path)
    def load_state(self, path: Path = Path(__file__).resolve().parent.parent.parent / "settings" / "training_preprocess_states.joblib"):
        """Reload saved runtime state for preprocessing testing data."""
        state = joblib.load(path)
        self.mean = state.get("mean")
        self.std = state.get("std")
        self.encoder = state.get("encoder")

prpsettings = PreProcessSettings.fromTOML()

def load_parquet(path:Path, file_name) -> np.recarray:
    """load .parquet file and convert to np.recarray(keeps headings). pd.DataFrame as intermediary.
    """
    file = path / file_name
    df = read_parquet(file)
    array = df.to_records(index=False)
    return array

#For Mack organization
def _etta_preprocess(array: np.recarray, to_drop:list[str], log_features:list = ["stream_jitter_1_var"]):
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



def _david_preprocess(array: np.recarray, to_drop:list[str], file_paths: dict, sample_rows: int|None = None):
    data = pd.DataFrame(array)
    # label_map = {
    #     "benign": "Benign", "benign_1": "Benign", "benign_2": "Benign", "benign_3": "Benign",
    #     "ddos": "DDoS", "dos": "DoS", "dos_1": "DoS",
    #     "dns_spoofing": "DNS_Spoofing", "brute_force": "BruteForce", "xss": "XSS",
    # }

    # id_cols = [
    #     "stream", "src_mac", "dst_mac", "src_ip", "dst_ip",
    #     "src_port", "dst_port", "device_mac", "eth_src_oui", "eth_dst_oui",
    # ]
    id_cols = to_drop

    # dfs = []
    # for label, path in file_paths.items():
    #     df = pd.read_csv(path, nrows=sample_rows)
    #     df["Label"] = label_map.get(label, label)
    #     dfs.append(df)
    #     print(f"Loaded {label}: {df.shape}")

    # data = pd.concat(dfs, ignore_index=True)

    # Drop ID columns
    #Same for both
    data = data.drop(columns=[c for c in id_cols if c in data.columns], errors="ignore")

    # Convert all non-Label columns to numeric, non-numeric values become 0
    #this is an interesting difference with Etta's logic. Where Etta drops those features David removes them. I think it makes more sense to remove?
    feature_cols = [c for c in data.columns if c != "Label"]
    for col in feature_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    #this next line is in Etta's logic. One difference is that Etta's sets everything to float32 (trivial change?)
    data[feature_cols] = data[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    #Etta adds a step here of logging and clipping the features. Looks like something to run with just her data
    print(f"\nCombined shape: {data.shape}")
    print(data["Label"].value_counts())

    return data

def _isolation_preprocess():
    pass


def preprocess_general(array:np.recarray, split:str, settings: PreProcessSettings = prpsettings) -> np.recarray:
    """General Preprocessing for combined_sample. Should be done AFTER train/test split to avoid data leakage.
    Note that this function is prepared to preprocess both training, testing, and flow training data. Pick a setting with split = 'Train', 'Test', or 'Flow'\\
    Steps:\\
        1. Drop unique ID and timestamp features as these reveal identifying patterns in the data generation. \\
        2. Drop target features (actually probably not, this should be the y in test_train_split) \\
        3. Identify numeric features. Use Standard Normalization to reduce overfitting to high valued features \\
        4. convert non-numeric columns to One-Hot Encodings \\
        5. replace inf and -inf values with nan \\
        6. replace all nan with 0 to avoid crashing models.\\
        Running the same function on Flow training as we do on the WWT allows for all numeric values to be normalized by
          the same mean and standard deviation discovered on the wwt training data. Since we'll be testing on the wwt Testing
            data, I think this is an appropriate mean and std to normalize the phase 3 training by - Mack
        """
    df = pd.DataFrame(array)
    if split.lower() == 'flow':
        #Drop unique ID, timestamp, and label features as these reveal identifying patterns in the data generation. \\
        #df_no_UID = df.drop(prpsettings.FLOW_DROP_FEATURES, axis=1).drop(prpsettings.FLOW_LABEL_FEATURES, axis=1)
        df_no_UID = df.drop(prpsettings.FLOW_DROP_FEATURES, axis=1, errors="ignore").drop(prpsettings.FLOW_LABEL_FEATURES, axis=1, errors="ignore")
        df_no_UID = df_no_UID.add_prefix("flow_")
        settings.load_state()
    else:
        #Drop unique ID, timestamp, and label features as these reveal identifying patterns in the data generation. \\
        #df_no_UID = df.drop(prpsettings.DROP_FEATURES, axis=1).drop(prpsettings.LABEL_FEATURES, axis=1)
        df_no_UID = df.drop(prpsettings.DROP_FEATURES, axis=1, errors="ignore").drop(prpsettings.LABEL_FEATURES, axis=1, errors="ignore")
    numeric_features = (
        df_no_UID
        .select_dtypes(include=np.number)
        .columns
        .to_list()
    )
    non_numeric = [feat for feat in df_no_UID.columns if feat not in numeric_features]

    #if split.lower() == 'train':
       # #Identify numeric features. Use Standard Normalization to reduce overfitting to high valued features \\
       # settings.fit_scaler(df_train_numeric=df_no_UID[numeric_features])
       # df_no_UID[numeric_features] = (df_no_UID[numeric_features] - settings.mean) / settings.std #type:ignore #suboptimal settings configuration
       # #convert non-numeric columns to One-Hot Encodings \\
       # settings.fit_encoder(df_no_UID[non_numeric])
       # prpsettings.save_state()
       # encoded = settings.encoder.transform(df_no_UID[non_numeric]) #type:ignore #suboptimal settings configuration
       # df_no_UID = df_no_UID.drop(columns=non_numeric).join(encoded)#type:ignore #suboptimal settings configuration

    #elif split.lower() == 'test':
       # #Identify numeric features. Use Standard Normalization to reduce overfitting to high valued features \\
        #if settings.mean is None or settings.std is None:
            #raise RuntimeError("settings must contain mean and std. Tip: compute split='Train' to set values")
        #settings.load_state()
        #df_no_UID[numeric_features] = (df_no_UID[numeric_features] - settings.mean) / settings.std
        ##convert non-numeric columns to One-Hot Encodings \\
        #encoded = settings.encoder.transform(df_no_UID[non_numeric]) #type:ignore #suboptimal settings configuration
        #df_no_UID = df_no_UID.drop(columns=non_numeric).join(encoded)#type:ignore #suboptimal settings configuration

    if split.lower() == 'train':
        settings.fit_scaler(df_train_numeric=df_no_UID[numeric_features])
        df_no_UID[numeric_features] = (df_no_UID[numeric_features] - settings.mean) / settings.std # type: ignore
        df_no_UID = df_no_UID[numeric_features]  # drop non_numeric entirely
        prpsettings.save_state()

    elif split.lower() == 'test':
        if settings.mean is None or settings.std is None:
            raise RuntimeError("settings must contain mean and std. Tip: compute split='Train' to set values")
        settings.load_state()
        df_no_UID[numeric_features] = (df_no_UID[numeric_features] - settings.mean) / settings.std
        df_no_UID = df_no_UID[numeric_features]  # drop non_numeric entirely


    elif split.lower() == 'flow':
        if settings.mean is None or settings.std is None:
            raise RuntimeError("settings must contain mean and std. Tip: compute split='Train' to set values")
        settings.load_state()
        flow_mean = settings.mean[~settings.mean.index.str.startswith('pkt_')]
        flow_std = settings.std[~settings.std.index.str.startswith('pkt_')]
        df_no_UID[numeric_features] = (df_no_UID[numeric_features] - flow_mean) / flow_std
        ##convert non-numeric columns to One-Hot Encodings \\
        #encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.int8)
        #encoder.set_output(transform="pandas")
        #encoded = encoder.fit_transform(df_no_UID[non_numeric]) #type:ignore #suboptimal settings configuration
        #df_no_UID = df_no_UID.drop(columns=non_numeric).join(encoded)#type:ignore #suboptimal settings configuration
        # Reuse the SAME encoder fit on training data instead of fitting a new one
        print("non_numeric columns in flow data:", non_numeric)
        print("shape of non_numeric slice:", df_no_UID[non_numeric].shape)
        if non_numeric:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.int8)
            encoder.set_output(transform="pandas")
            encoded = encoder.fit_transform(df_no_UID[non_numeric])
            df_no_UID = df_no_UID.drop(columns=non_numeric).join(encoded) # type: ignore

    else:
        raise ValueError("split must be one of 'Train', 'Test', or 'Flow'(for phase 3 training).")

    df_no_UID[numeric_features] = (
        df_no_UID[numeric_features]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .astype(np.float32) #This is different than etta's (float64). Because generate_dataset already reduces the file to float32, so increasing to float64 is unnecessary
     ) 
    processed_array = df_no_UID.to_records(index=False)
    return processed_array

def special_preprocess_autoencoder(array:np.recarray, log_features:list = ["pkt_stream_jitter_1_var"]):
    """special log clipping needed for the autoencoder. Should be run on training and testing data AFTER preprocess_combined.\\
        Unsure why we don't do this for pkt_stream_jitter_60_sum and pkt_stream_jitter_60_mean. Also unsure how the standard normalization I added will affect this - Mack"""
    df = pd.DataFrame(array)
    numeric_features = (
            df
            .select_dtypes(include=np.number)
            .columns
            .to_list()
    )
    for feat in log_features:
            if feat in df[numeric_features].columns:
                df[feat] = np.log1p(df[feat].clip(lower=0))
    processed = df.to_records(index=False)
    return processed

def _preprocess_flow_tr_sample(array:np.recarray) -> np.recarray:
    """Not sure this is needed, hoping we can use the above function on both"""
    processed_array = array
    return processed_array
