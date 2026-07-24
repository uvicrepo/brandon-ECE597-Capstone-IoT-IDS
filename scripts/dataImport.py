#File: Dataset Import and Lazy Loading functions
import tarfile
import urllib.request
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
import tomllib
import shutil

@dataclass
class ImportSettings:
    DATASET_URL:str
    DATASET_DIR:str
    FLOW_DIR:str
    PACKET_DIR:str
    PROCESSED_DATA_DIR:str
    FLOW_FILES:dict[str, list[str]]
    PACKET_FILES:dict[str, list[str]]

    DATASET_PATH:Path = field(init=False)
    PROCESSED_DATA_PATH:Path = field(init=False)
    FLOW_PATH:Path = field(init=False)
    PACKET_PATH:Path = field(init=False)

    def __post_init__(self):
        repo_root = Path(__file__).resolve().parent.parent
        self.DATASET_PATH = repo_root / self.DATASET_DIR
        self.FLOW_PATH = self.DATASET_PATH / self.FLOW_DIR
        self.PACKET_PATH = self.DATASET_PATH / self.PACKET_DIR
        self.PROCESSED_DATA_PATH = repo_root / self.PROCESSED_DATA_DIR

    @classmethod
    
    def fromTOML(cls, path: Path = Path(__file__).resolve().parent.parent / "settings" / "dataImport.toml") -> "ImportSettings":
        with open(path, 'rb') as f:
            data = tomllib.load(f)
        return cls(**data)

settings = ImportSettings.fromTOML()

def _parquet_path(csv_path:Path):
    return csv_path.with_suffix(".parquet")

def _convert_to_parquet():
    all_csvs = (
        [Path(settings.FLOW_PATH, f)   for files in settings.FLOW_FILES.values()   for f in files] +
        [Path(settings.PACKET_PATH, f) for files in settings.PACKET_FILES.values() for f in files]
    )
    to_convert = [p for p in all_csvs if not _parquet_path(p).is_file()]
    if not to_convert:
        return
    print(f"Converting {len(to_convert)} CSV file(s) to Parquet...")
    for csv_path in to_convert:
        print(f"  {csv_path.name}")
        df = pd.read_csv(csv_path, low_memory=False)
        df.to_parquet(_parquet_path(csv_path), index=False)
        del df
    print("Done.")


def fetch_datasets(dataset_url=settings.DATASET_URL, dataset_path=settings.DATASET_PATH):
    """
    Fetch data from DATASET_URL configured in dataImport.toml and convert all .csv files to .parquet format.  Will save all files in a root-level directory called 'datasets'. If no such directory exists, one will be created.
    """
    if not settings.FLOW_PATH.is_dir():
        settings.DATASET_PATH.mkdir(parents=True, exist_ok=True)
        tgz_path = settings.DATASET_PATH / "datasets.tar.gz"
        # os.makedirs("datasets", exist_ok=True)
        print("Downloading dataset...")
        urllib.request.urlretrieve(dataset_url, tgz_path)
        print("Extracting...")
        dataset_tgz = tarfile.open(tgz_path)
        dataset_tgz.extractall(path=settings.DATASET_PATH.parent)
        dataset_tgz.close()
        tgz_path.unlink()
        # Tarball's top-level folder is named 'datasets'; move its contents into data/raw
        extracted = settings.DATASET_PATH.parent / "datasets"
        for child in extracted.iterdir():
            shutil.move(str(child), str(settings.DATASET_PATH / child.name))
        extracted.rmdir()
    else:
        print("Datasets already downloaded.")
    _convert_to_parquet()

def _load(files_dict, path, key):
    filenames = files_dict[key]
    frames = [pd.read_parquet(_parquet_path(Path(path, f))) for f in filenames]
    return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]

def load_flow(key):
    """
    Load a pandas DataFrame of all flow data using a lazy load
    Use the cells below to load specific datasets as needed.
    Available keys: `benign`, `ddos_http`, `dos_http`, `dns_spoofing`, `xss`, `brute_force`

    Call `load_flow(<key>)` for flow-based features or `load_packet(<key>)` for packet-based features.

    Example:
    ```python
    df_xss_flow = load_flow("xss")
    df_benign_packet = load_packet("benign")
    ```
    """
    print(f"Loading Flow Dataset: {key}...")
    df = _load(settings.FLOW_FILES, settings.FLOW_PATH, key)
    print(f"Shape: {df.shape}")
    return df

def load_packet(key):
    """
    Load a pandas DataFrame of all packet data using a lazy load.
    Use the cells below to load specific datasets as needed.
    Available keys: `benign`, `ddos_http`, `dos_http`, `dns_spoofing`, `xss`, `brute_force`

    Call `load_flow(<key>)` for flow-based features or `load_packet(<key>)` for packet-based features.

    Example:
    ```python
    df_xss_flow = load_flow("xss")
    df_benign_packet = load_packet("benign")
    ```
    """
    print(f"Loading Packet Dataset: {key}...")
    df = _load(settings.PACKET_FILES, settings.PACKET_PATH, key)
    print(f"Shape: {df.shape}")
    return df

if __name__ == '__main__':
    fetch_datasets()
    print('Datasets created. Use load_flow or load_packet to do a lazy load')
