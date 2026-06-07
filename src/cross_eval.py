"""
Cross-dataset evaluation for CNN-LSTM-DQN-IDS-v2 -- the core experiment.

Builds the train-on-A / test-on-B matrix over the four datasets. The DIAGONAL
(train==test) is the in-domain baseline; OFF-DIAGONAL cells are the
generalization result the paper is about.

Two prediction granularities are handled honestly:
  - DQN methods (cfa_dqn, plain_dqn) output a BINARY detection (attack/normal),
    so they are scored with binary recall / F1.
  - Supervised methods (ce, focal) output a coarse FAMILY, scored with
    per-family recall and macro-F1.
Both granularities also report binary detection metrics so every method is
comparable on the same axis (the binary column is the apples-to-apples one).

Honest-results contract: every number comes from an actual fit. Cross-dataset
cells are expected to be substantially lower than in-domain; that gap is the
finding, not a bug.
"""

import argparse
import itertools
import json
import os

import numpy as np

DATASETS = ["nslkdd", "unsw", "cicids2017", "ciciot2023"]
METHODS = ["cfa_dqn", "plain_dqn", "ce", "focal"]


def load_split(processed_dir, dataset, split):
    pre = os.path.join(processed_dir, f"{dataset}_{split}")
    return np.load(pre + ".X.npy"), np.load(pre + ".y.npy")


def binary_metrics(y_true_bin, y_pred_bin):
    tp = int(np.sum((y_pred_bin == 1) & (y_true_bin == 1)))
    fp = int(np.sum((y_pred_bin == 1) & (y_true_bin == 0)))
    fn = int(np.sum((y_pred_bin == 0) & (y_true_bin == 1)))
    tn = int(np.sum((y_pred_bin == 0) & (y_true_bin == 0)))
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    return {"detection_recall": rec, "detection_precision": prec,
            "detection_f1": f1, "accuracy": acc}


def macro_f1(y_true, y_pred, n_classes):
    f1s = []
    for c in range(n_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return float(np.mean(f1s)), [float(x) for x in f1s]


def per_class_recall(y_true, y_pred, n_classes):
    out = []
    for c in range(n_classes):
        fn = np.sum((y_pred != c) & (y_true == c))
        tp = np.sum((y_pred == c) & (y_true == c))
        out.append(float(tp / (tp + fn)) if (tp + fn) else 0.0)
    return out


def train_eval_once(train_set, test_set, method, seed, processed_dir, cfg, epochs):
    import common
    import train as T
    n_classes = len(common.CANONICAL_LABELS)

    Xtr, ytr = load_split(processed_dir, train_set, "train")
    Xte, yte = load_split(processed_dir, test_set, "test")

    y_pred, kind = T.predict(method, Xtr, ytr, Xte, cfg, seed, n_classes, epochs)

    yte_bin = (yte != 0).astype(np.int64)
    rec = {"train": train_set, "test": test_set, "method": method,
           "seed": seed, "kind": kind, "in_domain": train_set == test_set}

    if kind == "binary":
        rec.update(binary_metrics(yte_bin, y_pred))
    else:  # multiclass family prediction
        ypred_bin = (y_pred != 0).astype(np.int64)
        rec.update(binary_metrics(yte_bin, ypred_bin))
        mf1, f1s = macro_f1(yte, y_pred, n_classes)
        rec["macro_f1"] = mf1
        rec["per_class_recall"] = per_class_recall(yte, y_pred, n_classes)
        rec["per_class_f1"] = f1s
    return rec


def _cell_key(r):
    return (r["train"], r["test"], r["method"], r["seed"])


def run_matrix(processed_dir, out_path, methods, seeds, cfg, epochs):
    # Resume: if the output file already has completed cells, keep them and
    # skip re-running them. A disconnect mid-run therefore costs nothing -- just
    # re-run the same command and it picks up where it stopped.
    results = []
    done_keys = set()
    if os.path.exists(out_path):
        try:
            with open(out_path) as f:
                results = json.load(f)
            done_keys = {_cell_key(r) for r in results}
            print(f"resuming: {len(results)} cells already done in {out_path}")
        except Exception:
            results = []

    def flush():
        # atomic-ish write: temp then replace, so a crash mid-write can't
        # corrupt the results file.
        tmp = out_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(results, f, indent=2)
        os.replace(tmp, out_path)

    for method in methods:
        for tr, te in itertools.product(DATASETS, DATASETS):
            for seed in seeds:
                if (tr, te, method, seed) in done_keys:
                    continue
                try:
                    res = train_eval_once(tr, te, method, seed, processed_dir, cfg, epochs)
                    results.append(res)
                    flush()  # <-- write after EVERY cell, not just at the end
                    tag = res.get("macro_f1", res["detection_f1"])
                    print(f"done: {method:9s} {tr}->{te:11s} seed={seed} "
                          f"det_recall={res['detection_recall']:.3f} score={tag:.3f}")
                except FileNotFoundError as e:
                    print(f"skip (missing processed data): {tr}/{te} :: {e}")
                except Exception as e:
                    print(f"ERROR {method} {tr}->{te} seed={seed}: {e}")
    flush()
    print(f"\nwrote {len(results)} completed cells to {out_path}")



def main():
    import model as M
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--out", default="results/tables/cross_matrix.json")
    ap.add_argument("--methods", nargs="+", default=METHODS)
    ap.add_argument("--seeds", nargs="+", type=int,
                    default=list(M.DEFAULT_CONFIG["seeds"]))
    ap.add_argument("--epochs", type=int, default=8)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    run_matrix(args.processed, args.out, args.methods, args.seeds,
               M.DEFAULT_CONFIG, args.epochs)


if __name__ == "__main__":
    main()
