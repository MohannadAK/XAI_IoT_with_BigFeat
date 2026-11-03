# Earlier version of the study

Kept for the record. The current code and results are in `experiments/` and `results/`.

| Path | Contents |
|---|---|
| `IoT/` | Code of the first version: ExtraTrees, BigFeat and TPOT on the pre-built 20-feature N-BaIoT files, SHAP/LIME feature importance, Friedman tests and figures |
| `Results/IoT/` | Results of the first version |
| `Datasets/IOT Top 20 features datasets/` | Pre-built 20-feature N-BaIoT files used by the first version. Their feature subset was fixed before any train/test split, and 42–49% of their rows are duplicates, which caps Macro-F1 at 0.65–0.78. `experiments/` uses them only for the comparison in the paper's appendix (`prebuilt20`). |
| `bigfeat_gen_feat_formulas.log` | BigFeat formulas generated in the first version |
| `round1_minmax/` | First run of the revised protocol (5 seeds, in-fold feature selection), which still min-max scaled every configuration. The diagnosis in `round1_minmax/diagnostics/` showed that the scaling interacts with scikit-learn's split threshold; rounds 2 and 3 therefore do not scale inputs to tree models. `run_round1.sh` documents how it was run; the current code no longer reproduces it. |

The scripts in `IoT/` use paths relative to their own location and expect to be run from inside
`legacy/IoT/` (or `legacy/IoT/Feature Importances/`).
