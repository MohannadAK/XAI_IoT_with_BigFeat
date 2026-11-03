"""
Why did automation fail to help (N-BaIoT), and why did TPOT help on Edge-IIoTset / CIC-IoT-2023?

  python experiments/diagnose.py   -> legacy/round1_minmax/diagnostics/*.csv (diagnoses the round-1 runs)

1. headroom.csv        baseline Macro-F1 vs best achievable (ceiling) per N-BaIoT device
2. tpot_params.csv     hyperparameters TPOT chose, per run
3. per_class.csv       per-class F1 by configuration on Edge-IIoTset and CIC-IoT-2023 (seeds 0-2)
4. scaling.csv         ExtraTrees with vs without min-max scaling. scikit-learn's tree splitter treats
                       feature values closer than 1e-7 as equal (FEATURE_THRESHOLD); min-max scaling of
                       columns with huge ranges (e.g. TCP ack numbers up to 2.1e9) pushes real
                       differences below that limit.
5. one_setting.csv     ExtraTrees default vs single setting changes (min_samples_split=10, bootstrap),
                       unscaled, to test whether TPOT's gain needs a search at all.
"""
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import datasets  # noqa: E402

RES = os.path.join(datasets.REPO, "legacy", "round1_minmax")
OUT = os.path.join(RES, "diagnostics")
SEEDS = (0, 1, 2)


def split(X, y, s, scale):
    y = LabelEncoder().fit_transform(y)
    Xtr, Xte, ytr, yte = train_test_split(X.values.astype(float), y, test_size=0.3, random_state=s, stratify=y)
    if scale == "minmax":
        sc = MinMaxScaler().fit(Xtr)
        Xtr, Xte = sc.transform(Xtr), np.clip(sc.transform(Xte), -1, 2)
    return Xtr, Xte, ytr, yte


def fit_eval(Xtr, Xte, ytr, yte, s, **params):
    t = time.perf_counter()
    m = ExtraTreesClassifier(n_estimators=params.pop("n_estimators", 300), random_state=s, n_jobs=1, **params).fit(Xtr, ytr)
    return f1_score(yte, m.predict(Xte), average="macro"), time.perf_counter() - t


def headroom():
    s = pd.read_csv(os.path.join(RES, "summary", "summary_by_unit_config.csv"))
    c = pd.read_csv(os.path.join(RES, "summary", "ceiling.csv"))
    s["unit"], c["unit"] = s.unit.astype(str), c.unit.astype(str)
    b = s[(s.dataset == "nbaiot") & (s.config == "original")].pivot(index="unit", columns="feature_mode", values="f1_macro__mean")
    cp = c[c.dataset == "nbaiot"].pivot(index="unit", columns="feature_mode", values="max_macro_f1")
    t = pd.DataFrame({"baseline_infold20": b["infold20"], "ceiling_raw": cp["raw115_seed0"],
                      "baseline_prebuilt20": b["prebuilt20"], "ceiling_prebuilt20": cp["prebuilt20"]})
    t["headroom_infold20"] = t.ceiling_raw - t.baseline_infold20
    t["headroom_prebuilt20"] = t.ceiling_prebuilt20 - t.baseline_prebuilt20
    return t.reset_index()


def tpot_params():
    rows = []
    for f in glob.glob(os.path.join(RES, "*", "*", "unit*", "seed*", "tpot.json")) + \
             glob.glob(os.path.join(RES, "*", "*", "unit*", "seed*", "both.json")):
        r = json.load(open(f))
        rows.append(dict(dataset=r["dataset"], feature_mode=r["feature_mode"], unit=r["unit"], seed=r["seed"],
                         config=r["config"], **{k: r["tpot_params"][k] for k in
                         ["n_estimators", "criterion", "max_features", "min_samples_split", "min_samples_leaf",
                          "bootstrap", "max_depth"]}))
    return pd.DataFrame(rows)


def per_class():
    rows = []
    for ds in ("edgeiiot", "ciciot2023"):
        for c in ("original", "bigfeat", "tpot", "both"):
            for s in SEEDS:
                r = json.load(open(os.path.join(RES, ds, "all", "unitall", f"seed{s}", f"{c}.json")))
                for k, v in r["per_class_f1"].items():
                    rows.append(dict(dataset=ds, config=c, seed=s, cls=k, f1=v))
    d = pd.DataFrame(rows).groupby(["dataset", "cls", "config"]).f1.mean().unstack().reset_index()
    for c in ("bigfeat", "tpot", "both"):
        d[f"delta_{c}"] = d[c] - d["original"]
    return d


TASKS = [("edgeiiot", "all", datasets.edgeiiot), ("ciciot2023", "all", datasets.ciciot)] + \
        [("nbaiot_all115", d, (lambda d: (lambda s: datasets.nbaiot_raw(d, s)))(d)) for d in ("1", "6", "9")]


def scaling():
    rows = []
    for ds, unit, load in TASKS:
        for s in SEEDS:
            X, y = load(s)
            for scale in ("minmax", "none"):
                f1, t = fit_eval(*split(X, y, s, scale), s)
                rows.append(dict(dataset=ds, unit=unit, seed=s, scaling=scale, f1_macro=f1, train_s=t))
    return pd.DataFrame(rows)


VARIANTS = {"default": {}, "min_samples_split=10": dict(min_samples_split=10), "bootstrap": dict(bootstrap=True),
            "min_samples_split=10+bootstrap": dict(min_samples_split=10, bootstrap=True)}


def one_setting():
    rows = []
    for ds, unit, load in TASKS[:2] + [("nbaiot_all115", "9", lambda s: datasets.nbaiot_raw("9", s))]:
        for s in SEEDS:
            X, y = load(s)
            parts = split(X, y, s, "none")
            for name, p in VARIANTS.items():
                f1, t = fit_eval(*parts, s, **dict(p))
                rows.append(dict(dataset=ds, unit=unit, seed=s, variant=name, f1_macro=f1, train_s=t))
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT, exist_ok=True)
    for name, fn in [("headroom", headroom), ("tpot_params", tpot_params), ("per_class", per_class),
                     ("scaling", scaling), ("one_setting", one_setting)]:
        df = fn()
        df.to_csv(os.path.join(OUT, f"{name}.csv"), index=False)
        print(f"== {name}", flush=True)
        if name in ("scaling", "one_setting"):
            key = "scaling" if name == "scaling" else "variant"
            print(df.groupby(["dataset", "unit", key])[["f1_macro", "train_s"]].mean().round(4).to_string(), flush=True)


if __name__ == "__main__":
    main()
