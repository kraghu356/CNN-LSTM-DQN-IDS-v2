"""
Cross-dataset alignment for CNN-LSTM-DQN-IDS-v2.

A cross-dataset generalization study only works if every dataset is mapped onto
a SHARED feature space and a SHARED label taxonomy. This module defines both,
plus per-dataset mappings. Train-on-A / test-on-B is only meaningful once the
inputs and the targets mean the same thing.

Edit the per-dataset column maps to match the exact headers in your downloaded
files; the canonical names below are the contract every preprocessor must honor.
"""

# ---------------------------------------------------------------------------
# Canonical flow-feature schema (the intersection that exists across datasets)
# ---------------------------------------------------------------------------
# Keep this deliberately small: only features that are recoverable in ALL four
# datasets. A smaller honest feature set transfers; a large one does not.
CANONICAL_FEATURES = [
    "duration",          # flow duration (seconds)
    "protocol",          # categorical: tcp/udp/icmp/other
    "src_bytes",         # bytes sent source->dest
    "dst_bytes",         # bytes sent dest->source
    "src_pkts",          # packets source->dest
    "dst_pkts",          # packets dest->source
    "flow_bytes_per_s",  # throughput
    "flow_pkts_per_s",   # packet rate
    "syn_flag",          # SYN flag present/count (binary or count)
    "rst_flag",          # RST flag present/count
]

CATEGORICAL_FEATURES = ["protocol"]

# ---------------------------------------------------------------------------
# Canonical coarse label taxonomy (shared across datasets)
# ---------------------------------------------------------------------------
# Cross-dataset multiclass is only possible at a coarse family level, because
# fine labels do not overlap. Binary (Normal vs Attack) is derived from this.
CANONICAL_LABELS = [
    "Normal",
    "DoS",        # DoS / DDoS
    "Probe",      # scanning / reconnaissance
    "Access",     # unauthorized access: R2L, U2R, exploits, web, brute force, backdoor
    "Malware",    # botnet / worms / mirai-family
    "Other",      # generic / fuzzers / anything not cleanly in the above
]


def to_binary(coarse_label: str) -> int:
    """Normal -> 0, anything else -> 1."""
    return 0 if coarse_label == "Normal" else 1


# ---------------------------------------------------------------------------
# Per-dataset label -> canonical family maps
# ---------------------------------------------------------------------------
# NSL-KDD (5 coarse classes; fine attack names also collapse here)
NSLKDD_LABEL_MAP = {
    "normal": "Normal",
    "dos": "DoS",
    "probe": "Probe",
    "r2l": "Access",
    "u2r": "Access",
}

# UNSW-NB15 attack_cat
UNSW_LABEL_MAP = {
    "Normal": "Normal",
    "DoS": "DoS",
    "Reconnaissance": "Probe",
    "Analysis": "Probe",
    "Exploits": "Access",
    "Backdoor": "Access",
    "Shellcode": "Access",
    "Worms": "Malware",
    "Fuzzers": "Other",
    "Generic": "Other",
}

# CICIDS2017 Label column
CICIDS2017_LABEL_MAP = {
    "BENIGN": "Normal",
    "DoS Hulk": "DoS",
    "DoS GoldenEye": "DoS",
    "DoS slowloris": "DoS",
    "DoS Slowhttptest": "DoS",
    "DDoS": "DoS",
    "PortScan": "Probe",
    "FTP-Patator": "Access",
    "SSH-Patator": "Access",
    "Web Attack \u2013 Brute Force": "Access",
    "Web Attack \u2013 XSS": "Access",
    "Web Attack \u2013 Sql Injection": "Access",
    "Infiltration": "Access",
    "Heartbleed": "Access",
    "Bot": "Malware",
}

# CICIoT2023 (33 fine classes -> families). Keys are matched case-insensitively
# by substring in the preprocessor, so prefixes like "DDoS-" / "Mirai-" cover
# the whole family without listing all 33.
CICIOT2023_LABEL_PREFIX_MAP = {
    "BenignTraffic": "Normal",
    "Benign": "Normal",
    "DDoS": "DoS",
    "DoS": "DoS",
    "Recon": "Probe",
    "Scanning": "Probe",
    "VulnerabilityScan": "Probe",
    "Mirai": "Malware",
    "BruteForce": "Access",
    "DictionaryBruteForce": "Access",
    "Web": "Access",
    "XSS": "Access",
    "SqlInjection": "Access",
    "CommandInjection": "Access",
    "Backdoor": "Access",
    "Uploading": "Access",
    "Spoofing": "Other",
    "MITM": "Other",
}


def map_ciciot_label(raw: str) -> str:
    """CICIoT2023 fine labels are like 'DDoS-SYN_Flood'; match by prefix."""
    r = str(raw)
    for prefix, fam in CICIOT2023_LABEL_PREFIX_MAP.items():
        if r.lower().startswith(prefix.lower()):
            return fam
    return "Other"


# ---------------------------------------------------------------------------
# Per-dataset raw-column -> canonical-feature maps
# ---------------------------------------------------------------------------
# Fill these in against the ACTUAL headers of your downloaded CSVs. The values
# on the right are the raw column names in each dataset; left are canonical.
# Where a dataset has no direct equivalent, leave None and the preprocessor will
# zero-fill that canonical feature (and you should note that in the paper).
COLUMN_MAPS = {
    "nslkdd": {
        "duration": "duration",
        "protocol": "protocol_type",
        "src_bytes": "src_bytes",
        "dst_bytes": "dst_bytes",
        "src_pkts": None,        # NSL-KDD has no raw packet counts
        "dst_pkts": None,
        "flow_bytes_per_s": None,
        "flow_pkts_per_s": None,
        "syn_flag": "flag",      # flag is categorical; preprocessor derives SYN
        "rst_flag": "flag",
    },
    "unsw": {
        "duration": "dur",
        "protocol": "proto",
        "src_bytes": "sbytes",
        "dst_bytes": "dbytes",
        "src_pkts": "spkts",
        "dst_pkts": "dpkts",
        "flow_bytes_per_s": "rate",
        "flow_pkts_per_s": None,
        "syn_flag": "synack",
        "rst_flag": None,
    },
    "cicids2017": {
        "duration": "Flow Duration",
        "protocol": "Protocol",
        "src_bytes": "Total Length of Fwd Packets",
        "dst_bytes": "Total Length of Bwd Packets",
        "src_pkts": "Total Fwd Packets",
        "dst_pkts": "Total Backward Packets",
        "flow_bytes_per_s": "Flow Bytes/s",
        "flow_pkts_per_s": "Flow Packets/s",
        "syn_flag": "SYN Flag Count",
        "rst_flag": "RST Flag Count",
    },
    "ciciot2023": {
        "duration": "flow_duration",
        "protocol": "Protocol Type",
        "src_bytes": "Tot sum",
        "dst_bytes": None,
        "src_pkts": "Number",
        "dst_pkts": None,
        "flow_bytes_per_s": "Rate",
        "flow_pkts_per_s": "Srate",
        "syn_flag": "syn_flag_number",
        "rst_flag": "rst_flag_number",
    },
}
