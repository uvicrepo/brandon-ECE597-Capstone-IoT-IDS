""" Produces TWO datasets per run:

1. combined_sample.parquet (the "World's Widest Table" / WWT)
    Packet rows sampled per the required distribution, each one matched to
    its flow record (inner join - every row is guaranteed to have BOTH
    pkt_* and flow_* columns, via oversample-and-compensate so unmatched
    packets get topped up rather than shrinking your sample below target).
    Feeds Phase 2 (unsupervised, trained on pkt_* columns only) and,
    later, Phase 3's re-classification step (flow_* columns already sit
    on whatever rows Phase 2 flags, no extra lookup needed).

2. flow_training_sample.parquet
    An INDEPENDENT random sample taken directly from the flow-level files
    (same 200k/4-6.2k proportions, segments collapsed), with no packet
    involvement at all. This is Phase 3's actual supervised TRAINING set,
    per the instructions: "generate a second random dataset directly from
    the flow-level datato be used for the supervised stage."
"""
# ----------------------------------------------------------------------

import gc
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
# from pathlib import Path
from dataImport import impsettings, fetch_datasets
import flowMatch as fm
import warnings
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

impsettings.BENIGN_COUNT #200000
impsettings.ATTACK_MIN #4000
impsettings.ATTACK_MAX #6200

impsettings.PROCESSED_DATA_PATH

impsettings.ATTACK_KEYS
# {
#     "ddos_http":    "DDoS-HTTP Flood",
#     "dos_http":     "DoS-HTTP Flood",
#     "dns_spoofing": "DNS Spoofing",
#     "xss":          "XSS",
#     "brute_force":  "Brute Force",
# }

# Rough packet-row match rates measured empirically via explore_flow_matching.py
# Only used to size the FIRST oversample draw efficiently
# correctness doesn't depend on these being exact, since _sample_matched
# tops up with additional draws if the first one falls short.
impsettings.EXPECTED_MATCH_RATE
# {
#     "benign":       0.94,
#     "ddos_http":    0.99,
#     "dos_http":     0.99,
#     "dns_spoofing": 0.92,
#     "xss":          0.86,
#     "brute_force":  0.60,
# }

impsettings.MAX_SAMPLE_ROUNDS #6
# ----------------------------------------------------------------------

def _downcast(df):
    """Shrink memory footprint: float64->float32 (direct cast - float32 is
    standard ML precision anyway, and pd.to_numeric's downcast='float' only
    shrinks values with NO precision loss, which never applies to genuine
    measured floats like durations/rates, so it does nothing for most flow
    columns), int64->int32-or-smaller where the value range allows it.
    A file like dos_http's flow CSV (1.64M rows x 84 mostly-numeric columns)
    can be several GB as float64/int64 in memory this roughly halves
    that, which is often the difference between fitting in RAM and crashing."""
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype("float32")
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    return df

def _load_parquet(path):
    return _downcast(pd.read_parquet(path.with_suffix(".parquet")))

def _load_flows_for_key(key):
    filenames = impsettings.FLOW_FILES[key]
    parts = [_load_parquet(impsettings.FLOW_PATH / f) for f in filenames]
    flows = pd.concat(parts, ignore_index=True) if len(parts) > 1 else parts[0]
    del parts
    gc.collect()
    return flows

# ----------------------------------------------------------------------
# Oversample-and-compensate: draw packets, keep only ones with a matched
# flow (inner join), top up with more draws until the target count is hit.
# ----------------------------------------------------------------------

def _sample_matched(pool_df, flows_df, n_needed, rng, expected_rate):
    """Sample exactly n_needed rows from pool_df that successfully match
    a flow record in flows_df, drawing extra as needed to compensate for
    unmatched packets. Raises if pool_df runs out before reaching target."""
    collected = []
    used_idx = pd.Index([], dtype=pool_df.index.dtype)
    n_collected = 0

    for _ in range(impsettings.MAX_SAMPLE_ROUNDS):
        remaining = n_needed - n_collected
        if remaining <= 0:
            break

        available = pool_df.drop(index=used_idx, errors="ignore")
        if available.empty:
            break

        # oversample by the inverse of the expected match rate, with a
        # 20% safety margin, but never less than what's actually needed
        draw_n = int(remaining / max(expected_rate, 0.05) * 1.2)
        draw_n = max(draw_n, remaining)
        draw_n = min(draw_n, len(available))

        draw = available.sample(n=draw_n, random_state=int(rng.integers(0, 2**31)))
        used_idx = used_idx.union(draw.index)

        matched = fm.attach_flow_features(draw, flows_df, how="inner",
                                            keep_match_key=True)
        # keep_match_key=True: lets us record which flow record each row
        # used, so build_flow_only_sample() can exclude those and avoid
        # duplicate rows between the WWT and the flow-only training set.
        take = matched.iloc[: max(remaining, 0)]
        collected.append(take)
        n_collected += len(take)

    if n_collected < n_needed:
        raise RuntimeError(
            f"Only found {n_collected} / {n_needed} matched rows -- the "
            f"packet file's pool is exhausted. Check EXPECTED_MATCH_RATE "
            f"or reduce the target count."
        )

    return pd.concat(collected, ignore_index=True).iloc[:n_needed].reset_index(drop=True)

# ----------------------------------------------------------------------
# WWT: packet sample, matched to flow, per category
# ----------------------------------------------------------------------

def _wwt_benign(files_dict, data_path, rng) -> tuple[pd.DataFrame, dict]:
    filenames = files_dict["benign"]
    parquet_paths = [(data_path / f).with_suffix(".parquet") for f in filenames]

    sizes    = [pq.read_metadata(str(p)).num_rows for p in parquet_paths]
    total    = sum(sizes)
    per_file = [int(impsettings.BENIGN_COUNT * s / total) for s in sizes]
    per_file[-1] += impsettings.BENIGN_COUNT - sum(per_file)

    flows = _load_flows_for_key("benign")
    rate = impsettings.EXPECTED_MATCH_RATE["benign"]

    frames = []
    used_keys = set()
    for path, n in zip(parquet_paths, per_file):
        df = pd.read_parquet(path)
        df["label"] = "Benign"
        matched = _sample_matched(df, flows, n, rng, rate)
        used_keys.update(matched[fm.MATCH_KEY_COL].unique())
        frames.append(matched)
        del df
        gc.collect()

    return pd.concat(frames, ignore_index=True), {"benign": used_keys}

def _wwt_attacks(files_dict, data_path, n_attack, rng) -> tuple[pd.DataFrame, dict]:
    keys   = list(impsettings.ATTACK_KEYS.keys())
    labels = list(impsettings.ATTACK_KEYS.values())

    proportions = rng.dirichlet(np.ones(len(keys)))
    per_type    = np.maximum(np.floor(proportions * n_attack).astype(int), 1)
    diff        = n_attack - per_type.sum()
    if diff > 0:
        per_type[rng.choice(len(keys), size=diff, replace=False)] += 1

    frames = []
    used_keys = {}
    for key, label, n in zip(keys, labels, per_type):
        parts = [_load_parquet(data_path / f) for f in files_dict[key]]
        df = pd.concat(parts, ignore_index=True) if len(parts) > 1 else parts[0]
        df["label"] = label

        flows = _load_flows_for_key(key)
        rate = impsettings.EXPECTED_MATCH_RATE[key]
        matched = _sample_matched(df, flows, int(n), rng, rate)
        used_keys[key] = set(matched[fm.MATCH_KEY_COL].unique())
        frames.append(matched)
        del df, flows
        gc.collect()

    return pd.concat(frames, ignore_index=True), used_keys

def build_wwt(seed=None) -> tuple[pd.DataFrame, dict]:
    """The 'World's Widest Table': packet rows for Phase 2, each one
    guaranteed to already carry its matching flow_* columns (compensated
    for unmatched packets, so the final counts still hit BENIGN_COUNT /
    ATTACK_MIN-ATTACK_MAX exactly)."""
    rng = np.random.default_rng(seed)
    n_attack = int(rng.integers(impsettings.ATTACK_MIN, impsettings.ATTACK_MAX + 1))

    benign, benign_keys = _wwt_benign(impsettings.PACKET_FILES, impsettings.PACKET_PATH, rng)
    attacks, attack_keys = _wwt_attacks(impsettings.PACKET_FILES, impsettings.PACKET_PATH, n_attack, rng)
    used_keys = {**benign_keys, **attack_keys}

    result = pd.concat([benign, attacks], ignore_index=True)
    result = result.sample(frac=1, random_state=int(rng.integers(0, 2**31))).reset_index(drop=True)
    return result, used_keys

# ----------------------------------------------------------------------
# Independent flow-only sample (Phase 3 training set) no packet
# involvement, segments collapsed the same way as the WWT's flow side.
# ----------------------------------------------------------------------

def _exclude_used(collapsed, exclude_keys, key, needed):
    """Drop flow records already consumed by the WWT, so the two datasets
    share no rows. Raises with the exact shortfall if too few remain."""
    total = len(collapsed)
    if exclude_keys:
        collapsed = collapsed[~collapsed[fm.MATCH_KEY_COL].isin(exclude_keys)]
    remaining = len(collapsed)
    if remaining < needed:
        raise RuntimeError(
            f"'{key}': {total} unique flows exist, {total - remaining} were used "
            f"by the WWT, leaving {remaining}. Need {needed} for a zero-overlap "
            f"flow-only sample -- short by {needed - remaining}."
        )
    return collapsed

def _flow_only_benign(rng, exclude_keys=None):
    filenames = impsettings.FLOW_FILES["benign"]
    parts = [_load_parquet(impsettings.FLOW_PATH / f) for f in filenames]
    raw = pd.concat(parts, ignore_index=True) if len(parts) > 1 else parts[0]

    collapsed = fm.collapse_flow_segments(raw)
    collapsed["label"] = "Benign"
    collapsed = _exclude_used(collapsed, exclude_keys, "benign", impsettings.BENIGN_COUNT)
    return collapsed.sample(n=impsettings.BENIGN_COUNT, random_state=int(rng.integers(0, 2**31)))

def _flow_only_attacks(n_attack, rng, exclude_keys=None):
    keys = list(impsettings.ATTACK_KEYS.keys())
    labels = list(impsettings.ATTACK_KEYS.values())

    proportions = rng.dirichlet(np.ones(len(keys)))
    per_type = np.maximum(np.floor(proportions * n_attack).astype(int), 1)
    diff = n_attack - per_type.sum()
    if diff > 0:
        per_type[rng.choice(len(keys), size=diff, replace=False)] += 1

    frames = []
    for key, label, n in zip(keys, labels, per_type):
        filenames = impsettings.FLOW_FILES[key]
        parts = [_load_parquet(impsettings.FLOW_PATH / f) for f in filenames]
        raw = pd.concat(parts, ignore_index=True) if len(parts) > 1 else parts[0]

        collapsed = fm.collapse_flow_segments(raw)
        collapsed["label"] = label
        n = int(n)
        collapsed = _exclude_used(collapsed, (exclude_keys or {}).get(key), key, n)
        frames.append(collapsed.sample(n=n, random_state=int(rng.integers(0, 2**31))))
        del raw, collapsed
        gc.collect()

    return pd.concat(frames, ignore_index=True)

def build_flow_only_sample(seed=None, exclude_keys=None):
    """Phase 3's actual supervised training set: sampled directly from
    flow-level data, independent of the packet side, same proportions.
    
    exclude_keys: optional dict of category -> set of flow records already
    used by the WWT. Pass build_wwt()'s second return value to guarantee
    the two datasets share zero rows.
    """
    rng = np.random.default_rng(seed)
    n_attack = int(rng.integers(impsettings.ATTACK_MIN, impsettings.ATTACK_MAX + 1))

    benign = _flow_only_benign(rng, (exclude_keys or {}).get("benign"))
    attacks = _flow_only_attacks(n_attack, rng, exclude_keys)

    result = pd.concat([benign, attacks], ignore_index=True)
    result = result.sample(frac=1, random_state=int(rng.integers(0, 2**31))).reset_index(drop=True)
    return result.drop(columns=[fm.MATCH_KEY_COL], errors="ignore")

# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------

def validate(df, label_col):
    n_benign = (df[label_col] == "Benign").sum()
    n_attack = (df[label_col] != "Benign").sum()

    assert n_benign == impsettings.BENIGN_COUNT, f"Expected {impsettings.BENIGN_COUNT} benign rows, got {n_benign}"
    assert impsettings.ATTACK_MIN <= n_attack <= impsettings.ATTACK_MAX, f"Attack count {n_attack} out of range"
    assert 97.0 <= n_benign / len(df) * 100 <= 98.5, "Benign % outside expected range"

    print(df[label_col].value_counts().to_string())


def generate_dataset():
    """Callable function for using in other files"""
    SEED = 42

    fetch_datasets()
    impsettings.PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)

    print("Building WWT (packet rows matched to flow, compensated for misses)...")
    wwt, used_flow_keys = build_wwt(seed=SEED)
    validate(wwt, "pkt_label")
    print(f"WWT shape: {wwt.shape}, all rows flow-matched by construction "
            f"(flow_matched should be all True): {wwt['flow_matched'].all()}")
    for k, v in used_flow_keys.items():
        print(f"  {k}: consumed {len(v)} unique flow records")
    wwt.drop(columns=[fm.MATCH_KEY_COL], errors="ignore").to_parquet(
        impsettings.PROCESSED_DATA_PATH / "combined_sample.parquet", index=False)
    print(f"Saved to {impsettings.PROCESSED_DATA_PATH / 'combined_sample.parquet'}\n")

    print("Building flow-only sample (Phase 3 training set), excluding "
            "flow records already used by the WWT...")
    flow_only = build_flow_only_sample(seed=SEED, exclude_keys=used_flow_keys)
    validate(flow_only, "label")
    flow_only.to_parquet(impsettings.PROCESSED_DATA_PATH / "flow_training_sample.parquet", index=False)
    print(f"Saved to {impsettings.PROCESSED_DATA_PATH / 'flow_training_sample.parquet'}")

if __name__ == "__main__":
    generate_dataset()