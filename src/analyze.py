"""
Aggregate cross_eval results into paper-ready tables.

Reads one or more cross-matrix JSON files (each a list of per-cell result dicts)
and produces, per (train, test) cell and per method, the mean +/- std across
seeds for every metric. Highlights the per-class recall table (where the
class-frequency method's rare-class advantage lives) and a balanced-accuracy
column (mean of per-class recalls) where that advantage shows even when
macro-F1 ties.

Usage:
    python src/analyze.py --inputs results/tables/unsw_5seed.json
    python src/analyze.py --inputs results/tables/*.json --out results/tables/summary.csv
"""

import argparse
import glob
import json
from collections import defaultdict

import numpy as np

import common  # for CANONICAL_LABELS

LABELS = common.CANONICAL_LABELS


def load_all(patterns):
    rows = []
    for pat in patterns:
        for path in glob.glob(pat):
            with open(path) as f:
                rows.extend(json.load(f))
    return rows


def group(rows):
    """key -> list of result dicts, key = (train, test, method)."""
    g = defaultdict(list)
    for r in rows:
        g[(r["train"], r["test"], r["method"])].append(r)
    return g


def mean_std(vals):
    a = np.asarray(vals, dtype=float)
    return float(a.mean()), float(a.std())


def balanced_accuracy(per_class_recall_list):
    """Mean per-class recall = balanced accuracy; rewards rare-class coverage."""
    arr = np.asarray(per_class_recall_list, dtype=float)  # (seeds, n_classes)
    return arr.mean(axis=1)  # per-seed balanced acc


def summarize(g):
    print("=" * 100)
    for (tr, te, method), cells in sorted(g.items()):
        n = len(cells)
        df1 = mean_std([c["detection_f1"] for c in cells])
        acc = mean_std([c["accuracy"] for c in cells])
        line = (f"{method:10s} {tr}->{te:11s} (n={n})  "
                f"det_f1={df1[0]:.3f}+/-{df1[1]:.3f}  "
                f"acc={acc[0]:.3f}+/-{acc[1]:.3f}")
        if "macro_f1" in cells[0]:
            mf1 = mean_std([c["macro_f1"] for c in cells])
            pcr = [c["per_class_recall"] for c in cells]
            bal = mean_std(balanced_accuracy(pcr))
            line += f"  macroF1={mf1[0]:.3f}+/-{mf1[1]:.3f}  balAcc={bal[0]:.3f}+/-{bal[1]:.3f}"
        print(line)
    print("=" * 100)


def per_class_table(g):
    """Print per-class recall (mean) for multiclass methods, per cell."""
    print("\nPER-CLASS RECALL (mean across seeds)")
    print("-" * 100)
    header = "method      cell             " + " ".join(f"{l[:7]:>8}" for l in LABELS)
    print(header)
    print("-" * 100)
    for (tr, te, method), cells in sorted(g.items()):
        if "per_class_recall" not in cells[0]:
            continue
        arr = np.asarray([c["per_class_recall"] for c in cells], dtype=float)
        m = arr.mean(axis=0)
        cell = f"{tr}->{te}"
        print(f"{method:11s} {cell:15s} " + " ".join(f"{v:8.3f}" for v in m))
    print("-" * 100)
    print("Note: the rare families (e.g. Malware/Worms) are where the "
          "class-frequency method should lead.")


def maybe_csv(g, out):
    if not out:
        return
    import csv
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        head = ["train", "test", "method", "n_seeds", "det_f1_mean", "det_f1_std",
                "acc_mean", "macro_f1_mean", "macro_f1_std", "bal_acc_mean", "bal_acc_std"]
        head += [f"recall_{l}" for l in LABELS]
        w.writerow(head)
        for (tr, te, method), cells in sorted(g.items()):
            df1 = mean_std([c["detection_f1"] for c in cells])
            acc = mean_std([c["accuracy"] for c in cells])
            if "macro_f1" in cells[0]:
                mf1 = mean_std([c["macro_f1"] for c in cells])
                pcr = [c["per_class_recall"] for c in cells]
                bal = mean_std(balanced_accuracy(pcr))
                rec = np.asarray(pcr, float).mean(axis=0).tolist()
            else:
                mf1 = (np.nan, np.nan); bal = (np.nan, np.nan)
                rec = [np.nan] * len(LABELS)
            w.writerow([tr, te, method, len(cells), df1[0], df1[1], acc[0],
                        mf1[0], mf1[1], bal[0], bal[1]] + rec)
    print(f"\nwrote summary CSV: {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", default=None, help="optional summary CSV path")
    args = ap.parse_args()
    rows = load_all(args.inputs)
    if not rows:
        print("No results found in:", args.inputs)
        return
    g = group(rows)
    summarize(g)
    per_class_table(g)
    maybe_csv(g, args.out)


if __name__ == "__main__":
    main()
