"""
Upper bound on accuracy / Macro-F1 imposed by label collisions: identical feature vectors
(at float32 precision) that carry different labels. The bound is the score of predicting the
majority label of each distinct vector, i.e. of a classifier that memorises the data perfectly.

  python experiments/ceiling.py            -> results/ceiling.csv

For N-BaIoT the bound is computed on the raw 115 features and, for the main setting, on the 20
features each run selected inside its training split (infold20_seed<k>).
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import datasets  # noqa: E402

OUT = os.path.join(datasets.REPO, "results")


def bound(X, y):
    k = pd.util.hash_pandas_object(pd.DataFrame(np.asarray(X, dtype=np.float32)), index=False).values
    df = pd.DataFrame({"k": k, "y": np.asarray(y)})
    maj = df.groupby("k").y.agg(lambda s: s.value_counts().idxmax())
    pred = df.k.map(maj)
    return dict(max_accuracy=(pred == df.y).mean(), max_macro_f1=f1_score(df.y, pred, average="macro"),
                distinct_vectors=int(df.k.nunique()), rows=len(df),
                duplicate_rows=float(1 - df.k.nunique() / len(df)))


def main():
    rows = []
    for d in datasets.NBAIOT_DEVICES:
        X, y = datasets.nbaiot_prebuilt(d)
        rows.append(dict(dataset="nbaiot", feature_mode="prebuilt20", unit=d, **bound(X.values, y.values)))
        X, y = datasets.nbaiot_raw(d, 0)
        rows.append(dict(dataset="nbaiot", feature_mode="raw115_seed0", unit=d, **bound(X.values, y.values)))
        # Main setting: the 20 features each run selected on its training split (recorded per run).
        for seed in range(5):
            run = os.path.join(datasets.REPO, "results", "round2", "nbaiot", "infold20", f"unit{d}", f"seed{seed}",
                               "original.json")
            cols = json.load(open(run))["selected_features"]
            X, y = datasets.nbaiot_raw(d, seed)
            rows.append(dict(dataset="nbaiot", feature_mode=f"infold20_seed{seed}", unit=d,
                             **bound(X[cols].values, y.values)))
    X, y = datasets.edgeiiot(0)
    rows.append(dict(dataset="edgeiiot", feature_mode="all_seed0", unit="all", **bound(X.values, y.values)))
    X, y = datasets.ciciot(0)
    rows.append(dict(dataset="ciciot2023", feature_mode="all_seed0", unit="all", **bound(X.values, y.values)))
    X, y = datasets.ciciot34(0)
    rows.append(dict(dataset="ciciot34", feature_mode="all_seed0", unit="all", **bound(X.values, y.values)))
    X, y = datasets.unswnb15(0)
    rows.append(dict(dataset="unswnb15", feature_mode="all_seed0", unit="all", **bound(X.values, y.values)))
    os.makedirs(OUT, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "ceiling.csv"), index=False)
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
