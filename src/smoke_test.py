"""
Smoke test for CNN-LSTM-DQN-IDS-v2.

Generates tiny synthetic data in the canonical space for two 'datasets', then
runs every method through one in-domain and one cross-domain cell. Confirms the
whole pipeline executes end-to-end and emits real metrics, WITHOUT needing any
downloaded dataset. This is a plumbing check, not a result.

    python src/smoke_test.py
"""

import os
import tempfile
import numpy as np


def main():
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    import common
    import cross_eval as CE
    import model as M

    n_classes = len(common.CANONICAL_LABELS)
    n_feat = len(common.CANONICAL_FEATURES)
    rng = np.random.default_rng(0)
    tmp = tempfile.mkdtemp()

    def make(name, n, shift):
        # two datasets with slightly different feature distributions so that
        # cross-domain is genuinely harder than in-domain
        X = (rng.standard_normal((n, n_feat)).astype(np.float32) + shift)
        y = rng.integers(0, n_classes, size=n).astype(np.int64)
        # make a couple of features weakly label-dependent so models can learn
        X[:, 0] += y * 0.5
        X[:, 3] += (y == 0) * 1.0
        np.save(os.path.join(tmp, f"{name}_train.X.npy"), X)
        np.save(os.path.join(tmp, f"{name}_train.y.npy"), y)
        np.save(os.path.join(tmp, f"{name}_test.X.npy"), X[: n // 3])
        np.save(os.path.join(tmp, f"{name}_test.y.npy"), y[: n // 3])

    make("dsA", 900, 0.0)
    make("dsB", 900, 1.5)

    cfg = M.DEFAULT_CONFIG
    print("=== smoke test: 2 synthetic datasets, all methods, 1 seed, 2 epochs ===\n")
    for method in CE.METHODS:
        for tr, te in [("dsA", "dsA"), ("dsA", "dsB")]:
            res = CE.train_eval_once(tr, te, method, seed=0,
                                     processed_dir=tmp, cfg=cfg, epochs=2)
            tag = res.get("macro_f1", res["detection_f1"])
            dom = "in-domain " if res["in_domain"] else "cross-dom "
            print(f"{method:9s} {dom} {tr}->{te}  "
                  f"det_recall={res['detection_recall']:.3f}  "
                  f"det_f1={res['detection_f1']:.3f}  score={tag:.3f}")
    print("\nOK -- pipeline runs end-to-end and produces real metrics.")
    print("(Numbers here are meaningless synthetic plumbing, not results.)")


if __name__ == "__main__":
    main()
