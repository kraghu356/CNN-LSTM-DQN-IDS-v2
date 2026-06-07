"""
Unified preprocessing for CNN-LSTM-DQN-IDS-v2.

Maps any of {nslkdd, unsw, cicids2017, ciciot2023} onto the canonical feature
space and coarse label taxonomy defined in common.py, so that train-on-A /
test-on-B is well defined. Per-dataset scalers/encoders are fit on that
dataset's own training split only.

Usage:
    python src/preprocess.py --dataset unsw \
        --input data/unsw/UNSW_NB15_training-set.csv \
        --out   data/processed/unsw_train

    # produces <out>.X.npy, <out>.y.npy (coarse int labels), <out>.meta.json
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

from common import (
    CANONICAL_FEATURES,
    CATEGORICAL_FEATURES,
    CANONICAL_LABELS,
    COLUMN_MAPS,
    NSLKDD_LABEL_MAP,
    UNSW_LABEL_MAP,
    CICIDS2017_LABEL_MAP,
    map_ciciot_label,
)

PROTO_BUCKETS = {"tcp": 0, "udp": 1, "icmp": 2}  # everything else -> 3 ("other")
LABEL_TO_ID = {name: i for i, name in enumerate(CANONICAL_LABELS)}


def _proto_to_bucket(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.lower().str.strip()
    return s.map(PROTO_BUCKETS).fillna(3).astype(np.float32)


def _coarse_labels(df: pd.DataFrame, dataset: str) -> np.ndarray:
    if dataset == "nslkdd":
        # NSL-KDD label column is the fine attack name; an attack->category file
        # collapses it. Here we expect a precomputed 'category' column in
        # {normal,dos,probe,r2l,u2r}; adjust if your loader differs.
        raw = df["category"].astype(str).str.lower().str.strip()
        fam = raw.map(NSLKDD_LABEL_MAP).fillna("Other")
    elif dataset == "unsw":
        raw = df["attack_cat"].fillna("Normal").astype(str).str.strip()
        fam = raw.map(UNSW_LABEL_MAP).fillna("Other")
    elif dataset == "cicids2017":
        raw = df["Label"].astype(str).str.strip()
        fam = raw.map(CICIDS2017_LABEL_MAP).fillna("Other")
    elif dataset == "ciciot2023":
        raw = df["label"].astype(str).str.strip()
        fam = raw.apply(map_ciciot_label)
    else:
        raise ValueError(dataset)
    return fam.map(LABEL_TO_ID).astype(np.int64).values


def build(dataset: str, input_path: str, out_prefix: str):
    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
    df = pd.read_csv(input_path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]

    colmap = COLUMN_MAPS[dataset]
    feat = pd.DataFrame(index=df.index)

    for canon in CANONICAL_FEATURES:
        raw = colmap.get(canon)
        if raw is None or raw not in df.columns:
            # No equivalent in this dataset: zero-fill and record it.
            feat[canon] = 0.0
            continue
        if canon == "protocol":
            feat[canon] = _proto_to_bucket(df[raw])
        elif canon in ("syn_flag", "rst_flag") and df[raw].dtype == object:
            # derive a binary flag from a categorical 'flag' column
            token = "S" if canon == "syn_flag" else "R"
            feat[canon] = df[raw].astype(str).str.contains(token, case=False).astype(np.float32)
        else:
            feat[canon] = pd.to_numeric(df[raw], errors="coerce").fillna(0.0).astype(np.float32)

    # clean inf / nan (CICIDS2017 throughput columns are notorious for inf)
    feat = feat.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    y = _coarse_labels(df, dataset)

    X = feat[CANONICAL_FEATURES].values.astype(np.float32)
    np.save(out_prefix + ".X.npy", X)
    np.save(out_prefix + ".y.npy", y)

    counts = {CANONICAL_LABELS[i]: int(n) for i, n in enumerate(np.bincount(y, minlength=len(CANONICAL_LABELS)))}
    meta = {
        "dataset": dataset,
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "features": CANONICAL_FEATURES,
        "categorical": CATEGORICAL_FEATURES,
        "label_order": CANONICAL_LABELS,
        "class_counts": counts,
        "zero_filled_features": [c for c in CANONICAL_FEATURES
                                 if colmap.get(c) is None or colmap.get(c) not in df.columns],
    }
    with open(out_prefix + ".meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[{dataset}] {X.shape[0]} rows x {X.shape[1]} feats -> {out_prefix}.*.npy")
    print("  class counts:", counts)
    if meta["zero_filled_features"]:
        print("  NOTE zero-filled (no equivalent in this dataset):", meta["zero_filled_features"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    choices=["nslkdd", "unsw", "cicids2017", "ciciot2023"])
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True, help="output prefix path")
    args = ap.parse_args()
    build(args.dataset, args.input, args.out)


if __name__ == "__main__":
    main()
