# Is AutoML Worth Its Cost for IoT Intrusion Detection?

Code and results for *Is AutoML Worth Its Cost for IoT Intrusion Detection? An Empirical Evaluation of
Automated Feature Engineering (BigFeat) and Evolutionary Pipeline Search (TPOT) against Low-Cost Baselines*.

The study compares automated feature engineering ([BigFeat](https://github.com/DataSystemsGroupUT/BigFeat))
and evolutionary pipeline search ([TPOT](https://github.com/EpistasisLab/tpot)) with low-cost default tree
ensembles on three intrusion-detection benchmarks (N-BaIoT, Edge-IIoTset, UNSW-NB15). Every learned
component is fitted inside the training split, every configuration is repeated over five seeds, and
detection performance, training cost and the quality of SHAP explanations are reported for each.

## Repository layout

```
experiments/            code for the paper
  datasets.py           data loaders (class capping, deduplication, preprocessing)
  protocol.py           one run: split, BigFeat / TPOT / model fit, performance, cost and explanation metrics
  run_grid.py           runs a grid of (dataset, seed, configuration) in parallel; resumable
  run_round2.sh         round 2: N-BaIoT, Edge-IIoTset, CIC-IoT-2023 (8 classes)
  run_round3.sh         round 3: UNSW-NB15, CIC-IoT-2023 (34 classes)
  analyze.py            summaries, significance tests, Pareto analysis, per-round figures
  ceiling.py            highest Macro-F1 any classifier can reach on each dataset
  paper_tables.py       every number in the paper's tables -> results/paper_tables.md
  paper_figures.py      the paper's figures, drawn at print size -> results/figures/
  diagnose.py           diagnostics of the round-1 runs (min-max scaling and scikit-learn's split threshold)
  extract_rars.py       unpacks the N-BaIoT RAR archives
results/
  ceiling.csv           upper bound on Macro-F1 per dataset
  paper_tables.md       the paper's tables, regenerated from the files below
  figures/              the paper's figures
  review_logs/          script and log of the additional runs (LightGBM reruns, CV grid, model choice on Edge-IIoTset)
  round2/               N-BaIoT, Edge-IIoTset, CIC-IoT-2023 (8 classes)
  round3/               UNSW-NB15, CIC-IoT-2023 (34 classes, incomplete)
legacy/                 the earlier version of the study (see legacy/README.md)
```

Each round directory holds one JSON file (metrics, timings, chosen hyperparameters, BigFeat formulas)
and one `_attr.npz` file (attributions) per run, under
`<dataset>/<feature setting>/unit<task>/seed<k>/<configuration>.json`, plus `summary/` with the aggregated
tables and figures and `logs/`.

### Which results the paper uses

| Paper | Source |
|---|---|
| N-BaIoT (main setting: top-20 features selected in each training split) | `results/round2/nbaiot/infold20` |
| N-BaIoT, all 115 features and the earlier pre-built files (appendix) | `results/round2/nbaiot/all`, `results/round2/nbaiot/prebuilt20` |
| Edge-IIoTset | `results/round2/edgeiiot` |
| UNSW-NB15 | `results/round3/unswnb15` |
| Dataset upper bounds | `results/ceiling.csv` |
| All table values | `results/paper_tables.md` |

The CIC-IoT-2023 runs were exploratory and are **not used in the paper**. The 8-class grid in round 2 is
complete and agrees with the paper's conclusions (default LightGBM is best, BigFeat gives no gain, and
TPOT's gain equals that of `split10`); the 34-class grid in round 3 is incomplete because fully grown
ExtraTrees models exceeded the memory of the test machine. Both are kept for transparency. `results/round2/_archive_lgbm_default_diverged/` holds
LightGBM runs with default settings that diverged; they were repeated with `max_delta_step=1`.

### Configurations

| Name in code | Name in paper | Description |
|---|---|---|
| `original` | Baseline | ExtraTrees, 300 trees, default settings |
| `split10` | ExtraTrees, split 10 | Baseline with `min_samples_split=10` |
| `split_grid` | ExtraTrees, CV grid | Baseline with `min_samples_split` chosen from {2, 5, 10, 20} by 3-fold CV Macro-F1 on the training split |
| `lgbm` | LightGBM | LightGBM with default settings except `max_delta_step=1` |
| `bigfeat` | BigFeat (scaled) | BigFeat features + min-max scaled originals → ExtraTrees |
| `bigfeat_raw` | BigFeat (unscaled) | BigFeat features + unscaled originals → ExtraTrees |
| `tpot` | TPOT (ExtraTrees) | TPOT search over ExtraTrees hyperparameters |
| `both` | BigFeat (scaled) + TPOT | `bigfeat` features → TPOT (ExtraTrees) |
| `both_raw` | BigFeat (unscaled) + TPOT | `bigfeat_raw` features → TPOT (ExtraTrees) |
| `bigfeat_auto` | BigFeat + model selection | BigFeat features → LightGBM, random forest or logistic regression, chosen by 3-fold CV Macro-F1 (re-implementation of BigFeat's selector, which uses binary F1; decision tree omitted) |
| `tpot_auto` | TPOT, full search | TPOT default search space (scaler → selector → transformer → 17 classifiers) |

## Installation

Python 3.13 was used.

```
pip install -r requirements.txt
```

`requirements.txt` installs BigFeat from upstream commit `74e320a`. To use a local checkout instead,
set `BIGFEAT_PATH` to its directory.

## Data

The raw datasets are not included. Download them and arrange them under one directory, pointed to by
`REVISION_DATA` (default `/mnt/Projects/datasets`):

```
$REVISION_DATA/
  nbaiot/extracted/<device name>/...csv        N-BaIoT (UCI repository); RAR archives can be
                                                unpacked with experiments/extract_rars.py
  edgeiiot/ML-EdgeIIoT-dataset.csv              Edge-IIoTset, "Selected dataset for ML and DL"
  ciciot2023/CSV/<attack folder>/*.csv          CIC-IoT-2023 (only needed for the CIC-IoT runs)
  unswnb15/train.csv, unswnb15/test.csv         UNSW-NB15 official training and testing partitions
                                                (they are pooled, so their order does not matter)
```

- N-BaIoT: <https://archive.ics.uci.edu/dataset/442/detection+of+iot+botnet+attacks+n+baiot>
- Edge-IIoTset: <https://www.kaggle.com/datasets/mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot>
- CIC-IoT-2023: <https://www.unb.ca/cic/datasets/iotdataset-2023.html>
- UNSW-NB15: <https://research.unsw.edu.au/projects/unsw-nb15-dataset>

Loaders cache the prepared data in `$REVISION_DATA/_cache/`. The earlier version's pre-built
20-feature N-BaIoT files are included in `legacy/Datasets/`.

## Reproducing the paper

```
bash experiments/run_round2.sh            # N-BaIoT, Edge-IIoTset (and CIC-IoT-2023, 8 classes)
bash experiments/run_round3.sh            # UNSW-NB15 (and CIC-IoT-2023, 34 classes)
python experiments/ceiling.py             # results/ceiling.csv
python experiments/paper_tables.py        # results/paper_tables.md
python experiments/paper_figures.py       # results/figures/
```

The run scripts are resumable: finished runs are skipped. Each run is single-threaded so that training
times are comparable; the number of runs in parallel is set by the `W_*` variables in the scripts. Per-run
training times are listed in `results/paper_tables.md`; explaining a run adds a few minutes, and TPOT with
its full search space takes 1--5 hours per run.

To run a single cell, for example LightGBM on UNSW-NB15 with seed 0:

```
REVISION_OUT=/tmp/out python experiments/run_grid.py --dataset unswnb15 --units all --seeds 0 --configs lgbm
```

## Earlier version

The first version of this study used pre-built 20-feature N-BaIoT files, a single train/test split and
min-max scaling. Its code and results are in `legacy/`, and `legacy/README.md` summarises what changed
and why.
