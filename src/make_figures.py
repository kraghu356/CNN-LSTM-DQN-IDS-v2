"""
Generate publication figures from cross_eval results.

Figures:
  1. per_class_recall_<cell>.png  -- grouped bar chart, methods x families,
     for each in-domain cell. This is the figure that shows the rare-class
     story (baselines at 0 on Malware, class-frequency method recovering it).
  2. cross_dataset_heatmap_<metric>.png -- when off-diagonal cells exist, a
     train x test heatmap per method showing generalization degradation.

All figures are generated from the SAME json the tables come from, so figures
and numbers cannot disagree. Nothing is hand-edited.

Usage:
    python src/make_figures.py --inputs results/tables/*.json --out results/figures
"""

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common

LABELS = common.CANONICAL_LABELS
DATASETS = ["nslkdd", "unsw", "cicids2017", "ciciot2023"]


def load_all(patterns):
    rows = []
    for pat in patterns:
        for p in glob.glob(pat):
            with open(p) as f:
                rows.extend(json.load(f))
    return rows


def per_class_recall_fig(rows, out_dir):
    # group by cell -> method -> list of per_class_recall
    cells = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if "per_class_recall" in r:
            cells[(r["train"], r["test"])][r["method"]].append(r["per_class_recall"])
    for (tr, te), bymethod in cells.items():
        methods = sorted(bymethod)
        x = np.arange(len(LABELS))
        w = 0.8 / max(len(methods), 1)
        plt.figure(figsize=(10, 5))
        for i, m in enumerate(methods):
            arr = np.asarray(bymethod[m], float)
            mean = arr.mean(axis=0)
            err = arr.std(axis=0) if arr.shape[0] > 1 else None
            plt.bar(x + i * w, mean, w, yerr=err, capsize=3, label=m)
        plt.xticks(x + 0.4 - w / 2, LABELS, rotation=20)
        plt.ylabel("Recall")
        plt.ylim(0, 1.05)
        plt.title(f"Per-class recall  ({tr} -> {te})")
        plt.legend()
        plt.tight_layout()
        fn = os.path.join(out_dir, f"per_class_recall_{tr}_to_{te}.png")
        plt.savefig(fn, dpi=150)
        plt.close()
        print("wrote", fn)


def heatmap_fig(rows, out_dir, metric="macro_f1"):
    # method -> matrix[train, test]
    bym = defaultdict(lambda: np.full((len(DATASETS), len(DATASETS)), np.nan))
    agg = defaultdict(list)
    for r in rows:
        if metric in r:
            agg[(r["method"], r["train"], r["test"])].append(r[metric])
    for (m, tr, te), vals in agg.items():
        if tr in DATASETS and te in DATASETS:
            bym[m][DATASETS.index(tr), DATASETS.index(te)] = np.mean(vals)
    for m, mat in bym.items():
        if np.isnan(mat).all():
            continue
        plt.figure(figsize=(6, 5))
        im = plt.imshow(mat, vmin=0, vmax=1, cmap="viridis")
        plt.colorbar(im, label=metric)
        plt.xticks(range(len(DATASETS)), DATASETS, rotation=30)
        plt.yticks(range(len(DATASETS)), DATASETS)
        plt.xlabel("test on"); plt.ylabel("train on")
        plt.title(f"{m}: {metric} (diagonal=in-domain)")
        for i in range(len(DATASETS)):
            for j in range(len(DATASETS)):
                if not np.isnan(mat[i, j]):
                    plt.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                             color="white" if mat[i, j] < 0.6 else "black", fontsize=8)
        plt.tight_layout()
        fn = os.path.join(out_dir, f"cross_dataset_{metric}_{m}.png")
        plt.savefig(fn, dpi=150)
        plt.close()
        print("wrote", fn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", default="results/figures")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rows = load_all(args.inputs)
    if not rows:
        print("No results found.")
        return
    per_class_recall_fig(rows, args.out)
    # heatmap only meaningful once off-diagonal cells exist; harmless otherwise
    heatmap_fig(rows, args.out, "macro_f1")
    print("done.")


if __name__ == "__main__":
    main()
