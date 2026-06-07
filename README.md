# CNN-LSTM-DQN-IDS-v2 — Cross-Dataset Generalization for Imbalanced Intrusion Detection

Follow-up to *"Class-Frequency-Aware Deep Reinforcement Learning for Imbalanced
Network Intrusion Detection"* (CNN-LSTM-DQN-IDS). The first paper validated the
class-frequency-aware DQN reward (`lambda_c = log(N / N_c)`) **in-domain** on
NSL-KDD and CICIDS2017. Its main acknowledged limitation was reliance on older
datasets and single-dataset evaluation.

This repository addresses that limitation directly. The research question:

> Do imbalance-aware IDS models that score well in-domain still generalize to
> traffic from a **different** network — particularly on rare attack families —
> and does the class-frequency reward help that transfer?

We evaluate **train-on-A / test-on-B** across four datasets and report honest
degradation, not just in-domain accuracy.

## Datasets

NSL-KDD (legacy baseline) · CICIDS2017 (carryover) · UNSW-NB15 (modern) ·
CICIoT2023 (newest). See [`data/README.md`](data/README.md) for downloads.
All datasets are mapped onto a shared 10-feature flow schema and a 6-family
coarse label taxonomy (`src/common.py`) so cross-dataset evaluation is
well-defined.

## What's here

```
src/
├── common.py       # canonical feature + label alignment, per-dataset maps
├── preprocess.py   # any dataset -> canonical .npy + .meta.json
├── model.py        # CNN-LSTM encoder + DQN, class-frequency reward
└── cross_eval.py   # the train-A/test-B matrix runner (the core experiment)
data/               # download targets (git-ignored)
results/            # tables + figures from real runs
models/             # checkpoints (git-ignored)
```

## Pipeline

```bash
python -m venv .venv && .venv\Scripts\activate    # Windows / PowerShell
pip install -r requirements.txt

# 1. preprocess every dataset's train + test split into the canonical space
python src/preprocess.py --dataset unsw --input data/unsw/UNSW_NB15_training-set.csv --out data/processed/unsw_train
python src/preprocess.py --dataset unsw --input data/unsw/UNSW_NB15_testing-set.csv  --out data/processed/unsw_test
# ... repeat for nslkdd, cicids2017, ciciot2023 ...

# 2. run the cross-dataset matrix (4x4 datasets x 4 methods x 5 seeds)
python src/cross_eval.py --processed data/processed --out results/tables/cross_matrix.json
```

## Methods compared

| key         | description                                              |
| ----------- | -------------------------------------------------------- |
| `cfa_dqn`   | CNN-LSTM-DQN with class-frequency reward (ours)          |
| `plain_dqn` | same agent, `lambda_c = 1` (no imbalance correction)     |
| `ce`        | CNN-LSTM with cross-entropy, no RL                       |
| `focal`     | CNN-LSTM with focal loss (conventional imbalance fix)    |

The headline result is the **degradation** from the diagonal (in-domain) to the
off-diagonal (cross-domain) cells, per method, with emphasis on rare-family
recall.

## Status & integrity note

This is the experiment scaffold. `cross_eval.train_eval_once` raises
`NotImplementedError` until the paper-1 DQN training loop and the CE/focal
baselines are wired in — by design, so a half-built run can never emit
placeholder numbers. Every value in `results/` comes from an actual completed
run; cells that haven't been run are absent from the tables, not estimated.
Expect cross-dataset numbers to be substantially lower than in-domain — that
gap is the finding.

## Citation

```
@misc{raghavendra2026crossdataset,
  title  = {Cross-Dataset Generalization for Imbalanced Network Intrusion Detection},
  author = {Raghavendra K and others},
  year   = {2026},
  note   = {In preparation}
}
```

Released under the [MIT License](LICENSE).
