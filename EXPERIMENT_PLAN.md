# Experiment Plan — Cross-Dataset Generalization (Paper 2)

Status tracker and exact run sequence. The goal is a complete, honest set of
results to write the paper from. Nothing is written up until the run that
produces it has actually completed.

## The contribution (what the experiments must support)

Standard IDS models — including conventional imbalance fixes (focal loss) —
achieve good aggregate scores by effectively ignoring rare attack families,
and they degrade sharply when tested on a different network than they were
trained on. A class-frequency-aware approach (reward `lambda_c = log(N/N_c)`
plus class-balanced sampling) provides coverage across all families and is
expected to transfer better across datasets.

Two claims, two experiment blocks:
- **C1 (in-domain rare-class coverage):** on each dataset, the frequency-aware
  method is the only one with non-zero recall on the rarest family, and leads
  on balanced accuracy.
- **C2 (cross-dataset generalization):** train-on-A / test-on-B degradation is
  smaller for the frequency-aware method, especially on rare families.

## Datasets (canonical 6-family taxonomy)

| dataset    | role            | status        |
| ---------- | --------------- | ------------- |
| UNSW-NB15  | modern          | DONE (preprocessed, in-domain run complete) |
| NSL-KDD    | legacy baseline | to download + preprocess |
| CICIDS2017 | carryover       | to download + preprocess |
| CICIoT2023 | newest          | to download + preprocess (subset OK first) |

## Run sequence

### Block A — in-domain, all datasets, 5 seeds  (supports C1)
For each dataset D, preprocess train+test, then:
```
python src/cross_eval.py --processed data/processed \
    --methods cfa_dqn plain_dqn ce focal --seeds 42 7 13 99 123 --epochs 8 \
    --out results/tables/<D>_5seed.json
```
UNSW is done (1 seed so far — rerun with 5 seeds). Repeat per dataset as each
is preprocessed.

### Block B — cross-dataset matrix, 5 seeds  (supports C2)
Once >= 2 datasets are preprocessed, the off-diagonal cells run automatically:
```
python src/cross_eval.py --processed data/processed \
    --methods cfa_dqn plain_dqn ce focal --seeds 42 7 13 99 123 --epochs 8 \
    --out results/tables/cross_full.json
```

### Analysis (after each block)
```
python src/analyze.py --inputs results/tables/*.json --out results/tables/summary.csv
python src/make_figures.py --inputs results/tables/*.json --out results/figures
```

## Metrics reported

- **Balanced accuracy** (mean per-class recall) — primary; rewards rare-class
  coverage, this is where the method leads.
- **Per-class recall table** — shows the rare-family rescue directly.
- Macro-F1 and detection (binary) F1 — reported for completeness; macro-F1 may
  tie, and the paper explains why (baselines win easy classes).
- For C2: degradation = in-domain minus cross-domain, per metric per method.

## Honesty constraints (non-negotiable)

- Every number in the paper comes from a completed run in `results/`.
- Single-seed numbers are never reported as final; 5-seed mean +/- std only.
- Rare-class recall (Malware: 130 train / 44 test on UNSW) is statistically
  noisy at one seed — the 5-seed std must be reported alongside.
- `cfa_dqn` currently bundles two changes (frequency reward + balanced
  sampling) vs `plain_dqn` (neither). This is a fair "frequency-awareness"
  ablation, but to attribute the gain to a single component, add a third
  variant later (balanced sampling + uniform reward). Note this as a limitation
  if that ablation isn't run before submission.
- Expect cross-dataset numbers to be much lower than in-domain. That gap is the
  finding, not a failure.

## Known limitations to disclose in the paper

- NSL-KDD has no Malware family, so models trained on it cannot predict that
  class — a structural transfer limit, reported not hidden.
- Several canonical features are zero-filled per dataset (no equivalent);
  listed in each `.meta.json` and to be tabulated in the paper.
- Coarse 6-family taxonomy is required for cross-dataset comparability; fine
  attack labels are not comparable across datasets.
