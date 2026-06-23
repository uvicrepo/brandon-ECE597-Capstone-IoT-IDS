import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from dataImport import settings, fetch_datasets

BENIGN_COUNT = 200000
ATTACK_MIN   = 4000
ATTACK_MAX   = 6200

REPO_ROOT  = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "data" / "sampled"

# Maps dataImport keys to readable label names
ATTACK_KEYS = {
    "ddos_http":    "DDoS-HTTP Flood",
    "dos_http":     "DoS-HTTP Flood",
    "dns_spoofing": "DNS Spoofing",
    "xss":          "XSS",
    "brute_force":  "Brute Force",
}


def _load_parquet(path):
    return pd.read_parquet(path.with_suffix(".parquet"))


def _sample_benign(files_dict, data_path, rng):
    """Load each benign file individually and sample from it before loading the next.
    Distributes 200k rows proportionally based on each file's actual row count."""
    filenames  = files_dict["benign"]
    parquet_paths = [(data_path / f).with_suffix(".parquet") for f in filenames]

    # Read row counts from parquet metadata — no need to load the data
    sizes    = [pq.read_metadata(str(p)).num_rows for p in parquet_paths]
    total    = sum(sizes)
    per_file = [int(BENIGN_COUNT * s / total) for s in sizes]
    per_file[-1] += BENIGN_COUNT - sum(per_file)  # fix any rounding remainder

    frames = []
    for path, n in zip(parquet_paths, per_file):
        df = pd.read_parquet(path)
        frames.append(df.sample(n=n, random_state=int(rng.integers(0, 2**31))))
        del df

    result = pd.concat(frames, ignore_index=True)
    result["label"] = "Benign"
    return result


def _sample_attacks(files_dict, data_path, n_attack, rng):
    """Load each attack type individually and sample from it before loading the next."""
    keys   = list(ATTACK_KEYS.keys())
    labels = list(ATTACK_KEYS.values())

    proportions = rng.dirichlet(np.ones(len(keys)))
    per_type    = np.maximum(np.floor(proportions * n_attack).astype(int), 1)
    diff        = n_attack - per_type.sum()
    if diff > 0:
        per_type[rng.choice(len(keys), size=diff, replace=False)] += 1

    frames = []
    for key, label, n in zip(keys, labels, per_type):
        parts = [_load_parquet(data_path / f) for f in files_dict[key]]
        df = pd.concat(parts, ignore_index=True) if len(parts) > 1 else parts[0]
        df["label"] = label
        frames.append(df.sample(n=int(n), random_state=int(rng.integers(0, 2**31))))
        del df

    return pd.concat(frames, ignore_index=True)


def sample_dataset(mode="packet", seed=None):
    """
    Sample from the raw dataset one file at a time to keep memory usage low.
    mode: "packet" or "flow"
    """
    rng      = np.random.default_rng(seed)
    n_attack = int(rng.integers(ATTACK_MIN, ATTACK_MAX + 1))

    files_dict = settings.PACKET_FILES if mode == "packet" else settings.FLOW_FILES
    data_path  = settings.PACKET_PATH  if mode == "packet" else settings.FLOW_PATH

    benign  = _sample_benign(files_dict, data_path, rng)
    attacks = _sample_attacks(files_dict, data_path, n_attack, rng)

    result = pd.concat([benign, attacks], ignore_index=True)
    result = result.sample(frac=1, random_state=int(rng.integers(0, 2**31))).reset_index(drop=True)
    return result


def validate(df):
    n_benign = (df["label"] == "Benign").sum()
    n_attack = (df["label"] != "Benign").sum()

    assert n_benign == BENIGN_COUNT, f"Expected {BENIGN_COUNT} benign rows, got {n_benign}"
    assert ATTACK_MIN <= n_attack <= ATTACK_MAX, f"Attack count {n_attack} out of range"
    assert 97.0 <= n_benign / len(df) * 100 <= 98.5, "Benign % outside expected range"

    print(df["label"].value_counts().to_string())


if __name__ == "__main__":
    SEED = 42  # change to None for a different result each run

    fetch_datasets()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Sampling packet dataset...")
    packet_sample = sample_dataset("packet", seed=SEED)
    validate(packet_sample)
    packet_sample.to_parquet(OUTPUT_DIR / "packet_sample.parquet", index=False)
    print(f"Saved to {OUTPUT_DIR / 'packet_sample.parquet'}\n")

    print("Sampling flow dataset...")
    flow_sample = sample_dataset("flow", seed=SEED)
    validate(flow_sample)
    flow_sample.to_parquet(OUTPUT_DIR / "flow_sample.parquet", index=False)
    print(f"Saved to {OUTPUT_DIR / 'flow_sample.parquet'}")