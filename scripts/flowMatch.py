"""
flowMatch.py — attach matching flow-level features onto packet-level rows,
for the CIC IoT-DIAD 2024 dataset.

Packet-level features come from packet-per-packet analysis (no single
"Protocol" column — instead l4_tcp / l4_udp flags, plus icmp_type).
Flow-level features come from CICFlowMeter (has Src/Dst IP+Port and a
numeric Protocol column, plus a pre-built "Flow ID" string).

Matching strategy: a flow is bidirectional, so the same physical flow can
have its "source" and "destination" swapped between the packet file and
the flow file (whichever side sent the first packet becomes "source" in
CICFlowMeter's Flow ID). So instead of string-matching the raw Flow ID,
we build a DIRECTION-INDEPENDENT key from sorted (ip, port) pairs +
protocol, on both sides, and match on that.
"""

import difflib
import pandas as pd

# ----------------------------------------------------------------------
# Packet-level columns (from the DI_AD_Packet-based-features CSVs)
# ----------------------------------------------------------------------
PKT_SRC_IP   = "src_ip"
PKT_DST_IP   = "dst_ip"
PKT_SRC_PORT = "src_port"
PKT_DST_PORT = "dst_port"
PKT_TCP_FLAG = "l4_tcp"
PKT_UDP_FLAG = "l4_udp"
PKT_ICMP_COL = "icmp_type"   # non-null/non-NaN => ICMP

PROTO_TCP  = 6
PROTO_UDP  = 17
PROTO_ICMP = 1

# ----------------------------------------------------------------------
# Flow-level columns (from the AD_Flow-based-features CSVs / CICFlowMeter)
# ----------------------------------------------------------------------
FLOW_ID_COL   = "Flow ID"
FLOW_SRC_IP   = "Src IP"
FLOW_DST_IP   = "Dst IP"
FLOW_SRC_PORT = "Src Port"
FLOW_DST_PORT = "Dst Port"
FLOW_PROTOCOL = "Protocol"

MATCH_KEY_COL = "_match_key"

# ----------------------------------------------------------------------
# FULL column lists as documented on the CIC IoT-DIAD 2024 website
# (https://www.unb.ca/cic/datasets/iot-diad-2024.html), for verify_columns.py
# to diff against your real CSV/parquet headers. These are NOT all used by
# the matching/aggregation logic above -- just the complete reference lists.
# ----------------------------------------------------------------------
ALL_PACKET_FEATURES = [
    "stream", "device_mac", "src_ip", "dst_ip", "src_port", "dst_port",
    "inter_arrival_time", "time_since_previously_displayed_frame",
    "port_class_dst", "l4_tcp", "l4_udp", "ttl", "eth_size",
    "tcp_window_size", "payload_entropy", "handshake_version",
    "handshake_cipher_suites_length", "handshake_cipher_suites",
    "handshake_extensions_length", "tls_server",
    "handshake_sig_hash_alg_len", "http_request_method", "http_host",
    "http_response_code", "User_Agent", "dns_server", "dns_query_type",
    "dns_len_qry", "dns_interval", "dns_len_ans", "eth_src_oui",
    "eth_dst_oui", "payload_length", "highest_layer", "http_uri",
    "http_content_len", "http_content_type", "icmp_type",
    "icmp_checksum_status", "icmp_data_size", "ntp_interval",
    "most_freq_spot", "min_et", "q1", "min_e", "var_e", "q1_e", "sum_p",
    "min_p", "max_p", "med_p", "average_p", "var_p", "q3_p", "q1_p",
    "iqr_p", "l3_ip_dst_count", "jitter",
    "stream_1_count", "stream_1_mean", "stream_1_var",
    "src_ip_1_count", "src_ip_1_mean", "src_ip_1_var",
    "src_ip_mac_1_count", "src_ip_mac_1_mean", "src_ip_mac_1_var",
    "channel_1_count", "channel_1_mean", "channel_1_var",
    "stream_jitter_1_sum", "stream_jitter_1_mean", "stream_jitter_1_var",
    "stream_5_count", "stream_5_mean", "stream_5_var",
    "src_ip_5_count", "src_ip_5_mean", "src_ip_5_var",
    "src_ip_mac_5_count", "src_ip_mac_5_mean", "src_ip_mac_5_var",
    "channel_5_count", "channel_5_mean", "channel_5_var",
    "stream_jitter_5_sum", "stream_jitter_5_mean", "stream_jitter_5_var",
    "stream_10_count", "stream_10_mean", "stream_10_var",
    "src_ip_10_count", "src_ip_10_mean", "src_ip_10_var",
    "src_ip_mac_10_count", "src_ip_mac_10_mean", "src_ip_mac_10_var",
    "channel_10_count", "channel_10_mean", "channel_10_var",
    "stream_jitter_10_sum", "stream_jitter_10_mean", "stream_jitter_10_var",
    "stream_30_count", "stream_30_mean", "stream_30_var",
    "src_ip_30_count", "src_ip_30_mean", "src_ip_30_var",
    "src_ip_mac_30_count", "src_ip_mac_30_mean", "src_ip_mac_30_var",
    "channel_30_count", "channel_30_mean", "channel_30_var",
    "stream_jitter_30_sum", "stream_jitter_30_mean", "stream_jitter_30_var",
    "stream_60_count", "stream_60_mean", "stream_60_var",
    "src_ip_60_count", "src_ip_60_mean", "src_ip_60_var",
    "src_ip_mac_60_count", "src_ip_mac_60_mean", "src_ip_mac_60_var",
    "channel_60_count", "channel_60_mean", "channel_60_var",
    "stream_jitter_60_sum", "stream_jitter_60_mean", "stream_jitter_60_var",
    "Label",  # "Label 2 for AD" -- exact header name may differ, verify this one first
]

ALL_FLOW_FEATURES = [
    "Flow ID", "Src IP", "Src Port", "Dst IP", "Dst Port", "Protocol",
    "Timestamp", "Flow Duration", "Total Fwd Packet", "Total Bwd packets",
    "Total Length of Fwd Packet", "Total Length of Bwd Packet",
    "Fwd Packet Length Max", "Fwd Packet Length Min",
    "Fwd Packet Length Mean", "Fwd Packet Length Std",
    "Bwd Packet Length Max", "Bwd Packet Length Min",
    "Bwd Packet Length Mean", "Bwd Packet Length Std",
    "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean", "Flow IAT Std",
    "Flow IAT Max", "Flow IAT Min", "Fwd IAT Total", "Fwd IAT Mean",
    "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min", "Bwd IAT Total",
    "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
    "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags",
    "Fwd Header Length", "Bwd Header Length", "Fwd Packets/s",
    "Bwd Packets/s", "Packet Length Min", "Packet Length Max",
    "Packet Length Mean", "Packet Length Std", "Packet Length Variance",
    "FIN Flag Count", "SYN Flag Count", "RST Flag Count", "PSH Flag Count",
    "ACK Flag Count", "URG Flag Count", "CWR Flag Count", "ECE Flag Count",
    "Down/Up Ratio", "Average Packet Size", "Fwd Segment Size Avg",
    "Bwd Segment Size Avg", "Fwd Bytes/Bulk Avg", "Fwd Packet/Bulk Avg",
    "Fwd Bulk Rate Avg", "Bwd Bytes/Bulk Avg", "Bwd Packet/Bulk Avg",
    "Bwd Bulk Rate Avg", "Subflow Fwd Packets", "Subflow Fwd Bytes",
    "Subflow Bwd Packets", "Subflow Bwd Bytes", "FWD Init Win Bytes",
    "Bwd Init Win Bytes", "Fwd Act Data Pkts", "Fwd Seg Size Min",
    "Active Mean", "Active Std", "Active Max", "Active Min",
    "Idle Mean", "Idle Std", "Idle Max", "Idle Min", "Label",
]

# ----------------------------------------------------------------------
# Aggregation rules for collapsing a flow's split 2-minute segments into
# one row. Anything not listed falls back to AGG_DEFAULT.
# ----------------------------------------------------------------------
AGG_DEFAULT = "mean"

_SUM_COLS = [
    "Flow Duration", "Total Fwd Packet", "Total Bwd packets",
    "Total Length of Fwd Packet", "Total Length of Bwd Packet",
    "Fwd IAT Total", "Bwd IAT Total",
    "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags",
    "Fwd Header Length", "Bwd Header Length",
    "FIN Flag Count", "SYN Flag Count", "RST Flag Count", "PSH Flag Count",
    "ACK Flag Count", "URG Flag Count", "CWR Flag Count", "ECE Flag Count",
    "Subflow Fwd Packets", "Subflow Fwd Bytes",
    "Subflow Bwd Packets", "Subflow Bwd Bytes",
    "Fwd Act Data Pkts",
]

_MAX_COLS = [
    "Fwd Packet Length Max", "Bwd Packet Length Max",
    "Flow IAT Max", "Fwd IAT Max", "Bwd IAT Max",
    "Packet Length Max", "Active Max", "Idle Max",
]

_MIN_COLS = [
    "Fwd Packet Length Min", "Bwd Packet Length Min",
    "Flow IAT Min", "Fwd IAT Min", "Bwd IAT Min",
    "Packet Length Min", "Active Min", "Idle Min",
    "Fwd Seg Size Min",
]

_MEAN_COLS = [
    "Fwd Packet Length Mean", "Fwd Packet Length Std",
    "Bwd Packet Length Mean", "Bwd Packet Length Std",
    "Flow Bytes/s", "Flow Packets/s",
    "Flow IAT Mean", "Flow IAT Std",
    "Fwd IAT Mean", "Fwd IAT Std", "Bwd IAT Mean", "Bwd IAT Std",
    "Fwd Packets/s", "Bwd Packets/s",
    "Packet Length Mean", "Packet Length Std", "Packet Length Variance",
    "Down/Up Ratio", "Average Packet Size",
    "Fwd Segment Size Avg", "Bwd Segment Size Avg",
    "Fwd Bytes/Bulk Avg", "Fwd Packet/Bulk Avg", "Fwd Bulk Rate Avg",
    "Bwd Bytes/Bulk Avg", "Bwd Packet/Bulk Avg", "Bwd Bulk Rate Avg",
    "Active Mean", "Active Std", "Idle Mean", "Idle Std",
]

_FIRST_COLS = [
    FLOW_ID_COL, FLOW_SRC_IP, FLOW_DST_IP, FLOW_SRC_PORT, FLOW_DST_PORT,
    FLOW_PROTOCOL, "Timestamp", "Label",
    "FWD Init Win Bytes", "Bwd Init Win Bytes",
]

AGG_RULES = {}
for _c in _SUM_COLS:   AGG_RULES[_c] = "sum"
for _c in _MAX_COLS:   AGG_RULES[_c] = "max"
for _c in _MIN_COLS:   AGG_RULES[_c] = "min"
for _c in _MEAN_COLS:  AGG_RULES[_c] = "mean"
for _c in _FIRST_COLS: AGG_RULES[_c] = "first"

PKT_PREFIX = "pkt_"
FLOW_PREFIX = "flow_"

# Columns this module actually needs to exist (matching key + everything
# named in AGG_RULES). Used to fail fast with a helpful message instead
# of silently producing all-NaN columns if the real CSV headers differ
# from what's documented on the CIC website (spacing, casing, etc.).
REQUIRED_PACKET_COLS = [PKT_SRC_IP, PKT_DST_IP, PKT_SRC_PORT, PKT_DST_PORT,
                        PKT_TCP_FLAG, PKT_UDP_FLAG]
REQUIRED_FLOW_COLS = sorted(set(
    [FLOW_SRC_IP, FLOW_DST_IP, FLOW_SRC_PORT, FLOW_DST_PORT, FLOW_PROTOCOL]
    + list(AGG_RULES.keys())
))


def check_columns(packet_df: pd.DataFrame, flow_df: pd.DataFrame) -> None:
    """Raise a clear error (with 'did you mean...' suggestions) if the
    real dataframes don't have the columns this module expects, rather
    than silently matching nothing / aggregating nothing."""
    problems = []
    for label, required, actual in [
        ("packet", REQUIRED_PACKET_COLS, packet_df.columns),
        ("flow", REQUIRED_FLOW_COLS, flow_df.columns),
    ]:
        missing = [c for c in required if c not in actual]
        for col in missing:
            close = difflib.get_close_matches(col, list(actual), n=2, cutoff=0.6)
            hint = f" (close match(es) found: {close})" if close else ""
            problems.append(f"  [{label}] missing expected column '{col}'{hint}")

    if problems:
        raise ValueError(
            "flowMatch: dataframe columns don't match what was hardcoded "
            "from the CIC IoT-DIAD 2024 website. Update the CONFIG section "
            "at the top of flowMatch.py to match your real CSV headers:\n"
            + "\n".join(problems)
        )


# ----------------------------------------------------------------------
# Direction-independent matching key
# ----------------------------------------------------------------------

def _packet_protocol(df: pd.DataFrame) -> pd.Series:
    """Derive a numeric protocol per packet row from l4_tcp / l4_udp / icmp_type.
    Note: icmp_type uses -1 as its "not applicable" sentinel, NOT NaN --
    confirmed empirically (see explore_flow_matching.py). Checking .notna()
    here would incorrectly tag every row as ICMP."""
    proto = pd.Series(pd.NA, index=df.index, dtype="Int64")
    proto = proto.mask(df[PKT_TCP_FLAG].fillna(0).astype(bool), PROTO_TCP)
    proto = proto.mask(df[PKT_UDP_FLAG].fillna(0).astype(bool), PROTO_UDP)
    if PKT_ICMP_COL in df.columns:
        proto = proto.mask(df[PKT_ICMP_COL].fillna(-1).ne(-1), PROTO_ICMP)
    return proto


def _canonical_key(ip_a, port_a, ip_b, port_b, protocol) -> pd.Series:
    """Order-independent key: sort the two (ip, port) endpoints so a flow
    and its reverse-direction packets produce the SAME key."""
    end_a = ip_a.astype(str) + ":" + port_a.astype(str)
    end_b = ip_b.astype(str) + ":" + port_b.astype(str)
    lo = pd.concat([end_a, end_b], axis=1).min(axis=1)
    hi = pd.concat([end_a, end_b], axis=1).max(axis=1)
    return lo + "__" + hi + "__" + protocol.astype(str)


def _add_packet_key(packet_df: pd.DataFrame) -> pd.DataFrame:
    df = packet_df.copy()
    protocol = _packet_protocol(df)
    df[MATCH_KEY_COL] = _canonical_key(
        df[PKT_SRC_IP], df[PKT_SRC_PORT], df[PKT_DST_IP], df[PKT_DST_PORT], protocol
    )
    return df


def _add_flow_key(flow_df: pd.DataFrame) -> pd.DataFrame:
    df = flow_df.copy()
    df[MATCH_KEY_COL] = _canonical_key(
        df[FLOW_SRC_IP], df[FLOW_SRC_PORT], df[FLOW_DST_IP], df[FLOW_DST_PORT],
        df[FLOW_PROTOCOL],
    )
    return df


# ----------------------------------------------------------------------
# Collapse multi-segment flows (same match key) into one row -> "Connection"
# ----------------------------------------------------------------------

def _collapse_flow_segments(flow_df: pd.DataFrame) -> pd.DataFrame:
    df = _add_flow_key(flow_df)
    agg_map = {
        col: AGG_RULES.get(col, AGG_DEFAULT)
        for col in df.columns
        if col != MATCH_KEY_COL
    }
    return df.groupby(MATCH_KEY_COL, as_index=False).agg(agg_map)


# Public alias -- used by generate_dataset.py to build the independent
# flow-only sample (Task 3.2's "second random dataset... directly from
# the flow-level data"), which also needs segment collapsing applied.
collapse_flow_segments = _collapse_flow_segments


# ----------------------------------------------------------------------
# Attach the matching (collapsed) flow row onto each packet row
# ----------------------------------------------------------------------

def attach_flow_features(packet_df: pd.DataFrame,
                            flow_df: pd.DataFrame,
                            how: str = "left",
                            keep_match_key: bool = False) -> pd.DataFrame:
    """
    packet_df: sampled packet rows for ONE category (e.g. one attack type)
    flow_df:   the raw, un-collapsed flow rows for that SAME category
    Returns packet_df with flow_df's columns attached, prefixed:
        pkt_<original packet column>
        flow_<original flow column>
    plus a `flow_matched` bool column.
    how="left" keeps unmatched packets (flow_ cols become NaN);
    how="inner" drops packets with no matching flow.

    keep_match_key: if True, keeps the internal _match_key column in the
    output instead of dropping it. Used by generate_dataset.py to track
    which flow records got consumed, for de-duplication against the
    independent flow-only sample. Never meant to reach a saved dataset
    or a model; always stripped before parquet output.
    """
    check_columns(packet_df, flow_df)
    pkt = _add_packet_key(packet_df)
    flow_collapsed = _collapse_flow_segments(flow_df)

    pkt = pkt.rename(columns={c: PKT_PREFIX + c for c in packet_df.columns})
    flow_collapsed = flow_collapsed.rename(
        columns={c: FLOW_PREFIX + c for c in flow_df.columns}
    )

    merged = pkt.merge(
        flow_collapsed,
        left_on=MATCH_KEY_COL,
        right_on=MATCH_KEY_COL,
        how=how,
    )
    probe_col = FLOW_PREFIX + FLOW_ID_COL
    merged["flow_matched"] = merged[probe_col].notna() if how == "left" else True
    if keep_match_key:
        return merged
    return merged.drop(columns=[MATCH_KEY_COL])


def report_match_rate(packet_df: pd.DataFrame, flow_df: pd.DataFrame) -> None:
    """Sanity check: what fraction of a category's packets found a flow match?"""
    pkt = _add_packet_key(packet_df)
    flow_keys = set(_add_flow_key(flow_df)[MATCH_KEY_COL].unique())
    matched = pkt[MATCH_KEY_COL].isin(flow_keys)
    print(f"{matched.sum()} / {len(pkt)} packets matched a flow "
          f"({matched.mean() * 100:.1f}%)")


if __name__ == "__main__":
    # Example:
    # from dataImport import load_packet, load_flow
    # packets = load_packet("xss")
    # flows   = load_flow("xss")
    # report_match_rate(packets, flows)
    # wide = attach_flow_features(packets, flows)
    # print(wide.shape)
    pass