# Datasets

Raw datasets are **not** committed (large + licence-restricted). Download them
into the folders below, then run `src/preprocess.py` for each. Processed `.npy`
files land in `data/processed/` and are also git-ignored.

| Dataset      | Role in this paper            | Download |
| ------------ | ----------------------------- | -------- |
| NSL-KDD      | Legacy baseline (from paper 1) | https://www.unb.ca/cic/datasets/nsl.html |
| CICIDS2017   | Carryover (from paper 1)       | https://www.unb.ca/cic/datasets/ids-2017.html |
| UNSW-NB15    | Modern benchmark               | https://research.unsw.edu.au/projects/unsw-nb15-dataset |
| CICIoT2023   | Newest benchmark               | https://www.unb.ca/cic/datasets/iotdataset-2023.html |

## Expected layout

```
data/
├── nslkdd/      KDDTrain+.txt, KDDTest+.txt  (+ attack->category mapping)
├── unsw/        UNSW_NB15_training-set.csv, UNSW_NB15_testing-set.csv
├── cicids2017/  the per-day MachineLearningCVE CSVs (merge or process per-file)
├── ciciot2023/  the CSV part files
└── processed/   generated .npy + .meta.json  (git-ignored)
```

## Preprocess each (train + test splits)

```bash
python src/preprocess.py --dataset unsw \
    --input data/unsw/UNSW_NB15_training-set.csv --out data/processed/unsw_train
python src/preprocess.py --dataset unsw \
    --input data/unsw/UNSW_NB15_testing-set.csv  --out data/processed/unsw_test
# repeat for nslkdd, cicids2017, ciciot2023
```

## Before you run

The column maps in `src/common.py` (`COLUMN_MAPS`) are set to the *typical*
headers for each dataset, but releases vary. Open one CSV from each dataset,
check the exact column names, and adjust the map. The preprocessor zero-fills
any canonical feature with no equivalent and records it in the `.meta.json`
under `zero_filled_features` — review that list, it matters for the paper's
discussion of why transfer is hard.

NSL-KDD has no botnet/worm traffic, so its `Malware` family is empty by design.
A model trained on NSL-KDD cannot predict that class — this is a real transfer
limitation to report, not something to paper over.
