"""
Leakage-free, multi-seed experimental protocol for the revision.

One call to ``run_one(task, seed, config)`` does, in this order:

1. Load the task's raw data (no preprocessing that looks at labels or at the test rows).
2. Balanced per-class subsample drawn with ``seed`` (cap per class, see datasets.py).
3. Stratified 70/30 split with ``seed``.
4. Everything learned is fit on the training split only:
   optional min-max scaler (only for models that need it; every current model is tree-based, so it
   is off) -> optional in-fold top-k feature selection -> BigFeat -> model.
5. Performance, cost, and explanation-quality metrics on the untouched test split.

Configurations
    original : ExtraTrees(300) on the (scaled, optionally selected) features
    bigfeat  : BigFeat features (fit on train) + ExtraTrees(300)
    tpot     : TPOT restricted to ExtraTrees hyperparameters, original features
    both     : BigFeat features + TPOT
    split10  : ExtraTrees(300) with min_samples_split=10 (the single setting TPOT kept choosing)
    split_grid : ExtraTrees(300) with min_samples_split chosen from {2, 5, 10, 20} by 3-fold CV Macro-F1 on
               the training split (a manual grid, independent of TPOT)
    lgbm     : LightGBM with default settings except max_delta_step=1 (second tree-model family)
    bigfeat_raw : BigFeat's generated features + the *unscaled* original features + ExtraTrees(300).
               BigFeat min-max scales internally and returns scaled originals; this variant keeps its
               generated features but hands the model the original columns unscaled.
    both_raw : bigfeat_raw features + TPOT
    bigfeat_auto : BigFeat features + the model BigFeat's own selection step picks (LightGBM, Random
               Forest or Logistic Regression, by 3-fold CV macro-F1 on the training split)
    tpot_auto : TPOT's default, unrestricted search space ("linear": scaler -> selector -> transformer
               -> classifier, over 17 model families), original features. TPOT decides on scaling.
    lgbm_cv  : LightGBM with max_delta_step chosen from {0, 1} by 3-fold CV macro-F1 on the training split
               (the independent counterpart of lgbm, whose max_delta_step=1 was set after seeing divergence)
    flaml    : FLAML AutoML (low-cost search over LightGBM, random forest, XGBoost, ExtraTrees, linear
               models) with its default settings, macro-F1 metric, one thread and a fixed time budget

Why scaling is off: scikit-learn's tree splitter treats feature values closer than 1e-7 as equal
(FEATURE_THRESHOLD). Min-max scaling columns with huge ranges (e.g. TCP ack numbers up to 2.1e9)
pushes real differences below that limit (see experiments/diagnose.py). Tree models are invariant to
monotone rescaling, so they do not need it.
"""
from __future__ import annotations

import io
import json
import os
import pickle
import sys
import time
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, label_binarize

warnings.filterwarnings("ignore")

BIGFEAT_PATH = os.environ.get("BIGFEAT_PATH", "/mnt/Projects/BigFeat")
if BIGFEAT_PATH not in sys.path:
    sys.path.insert(0, BIGFEAT_PATH)

CONFIGS = ("lgbm_cv", "flaml", "original", "split10", "split_grid", "lgbm", "bigfeat", "bigfeat_raw", "tpot", "both", "both_raw",
           "bigfeat_auto", "tpot_auto")
# split_grid: a small manual grid that a practitioner would try, chosen without TPOT's help.
SPLIT_GRID = (2, 5, 10, 20)
BIGFEAT_CONFIGS = ("bigfeat", "bigfeat_raw", "both", "both_raw", "bigfeat_auto")
# Configurations whose model needs scaled inputs. All current models are tree-based, so none do.
# A non-tree model (e.g. logistic regression, k-NN, a neural net) must be added here.
NEEDS_SCALING = set()
# LightGBM defaults diverge on these data (raw scores reach ~1e6 when some classes become perfectly
# separable; Edge-IIoTset Macro-F1 fell to 0.18-0.36 on 3 of 5 seeds). max_delta_step caps each tree's
# leaf output and is the single stabilising setting used here.
LGBM_PARAMS = dict(n_jobs=1, verbose=-1, max_delta_step=1.0)
LGBM_CV_GRID = (0.0, 1.0)            # max_delta_step values for lgbm_cv (0 = LightGBM's default, no limit)
# FLAML: defaults (estimator list, evaluation method, search strategy), macro-F1, one thread.
FLAML_PARAMS = dict(task="classification", metric="macro_f1", time_budget=300, n_jobs=1, verbose=0)

# Same settings as the manuscript (Experimental setup, Pipeline Configurations).
ET_PARAMS = dict(n_estimators=300, n_jobs=1)
BIGFEAT_PARAMS = dict(gen_size=10, iterations=10, estimator="rf", feat_imps=True,
                      check_corr=True, selection="stability")
TPOT_SPACE = {
    "sklearn.ensemble.ExtraTreesClassifier": {
        "n_estimators": [100, 200, 300],
        "criterion": ["gini", "entropy"],
        "max_features": ["sqrt", "log2"],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 5],
        "bootstrap": [True, False],
        "max_depth": [None, 10, 20],
    }
}
# BigFeat's model-selection candidates (bigfeat_base.select_estimator), restricted to the three
# models it offers besides a single decision tree.
BIGFEAT_AUTO_CANDIDATES = ("lgb", "rf", "lr")
TPOT_PARAMS = dict(generations=8, population_size=15, cv=3, scorers=["f1_macro"],
                   max_time_mins=None, max_eval_time_mins=5, early_stop=5, n_jobs=1, verbose=0)

# Explanation budget (kept constant across configurations so costs are comparable).
N_EXPLAIN = 300          # test rows explained with SHAP
N_LIME = 40              # test rows explained with LIME
N_EVAL_XAI = 2000        # test rows used for PFI and deletion/insertion curves
PFI_REPEATS = 3
MAX_CURVE_STEPS = 20     # deletion/insertion curve resolution (features removed in blocks)
N_RANDOM_ORDERS = 3
N_FAITH_SUBSETS = 30     # random subsets per instance for faithfulness correlation


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def _timer():
    t0 = time.perf_counter()
    c0 = time.process_time()
    return lambda: (time.perf_counter() - t0, time.process_time() - c0)


class _ByteCounter:
    """File-like sink that only counts bytes (model size without memory or disk cost)."""
    def __init__(self):
        self.n = 0

    def write(self, b):
        n = memoryview(b).nbytes
        self.n += n
        return n


def select_topk_in_fold(X_train: np.ndarray, y_train: np.ndarray, k: int, seed: int):
    """Top-k features by ExtraTrees impurity importance, computed on the training split only."""
    et = ExtraTreesClassifier(n_estimators=100, random_state=seed, n_jobs=1)
    et.fit(X_train, y_train)
    order = np.argsort(-et.feature_importances_)
    return np.sort(order[:k])


def bigfeat_formula_stats(bf, feature_names):
    """Formula string, depth, operator count and leaf features for each generated feature."""
    out = []
    for i in range(len(bf.tracking_ids)):
        ops = [o[0] if isinstance(o, (list, tuple, np.ndarray)) else o for o in list(bf.tracking_ops[i])]
        ids = [int(j) for j in list(bf.tracking_ids[i])]
        depth = int(bf.feat_depths[i])
        op_names = [getattr(o, "__name__", str(o)) for o in ops]
        n_ops = sum(1 for n in op_names if n != "original_feat")
        out.append(dict(
            name=f"BF_{i}",
            formula=_formula(depth, list(bf.tracking_ops[i]), list(bf.tracking_ids[i]), feature_names),
            depth=depth, n_ops=n_ops, leaves=ids,
            n_unique_leaves=len(set(ids)),
        ))
    return out


def _formula(depth, op_ls, id_ls, names):
    if depth == 0:
        return names[int(id_ls.pop())]
    depth -= 1
    op = op_ls.pop()
    op = op[0] if isinstance(op, (list, tuple, np.ndarray)) else op
    n = getattr(op, "__name__", str(op))
    if n in ("multiply", "add", "subtract"):
        a = _formula(depth, op_ls, id_ls, names)
        b = _formula(depth, op_ls, id_ls, names)
        return f"({b} {dict(multiply='*', add='+', subtract='-')[n]} {a})"
    a = _formula(depth, op_ls, id_ls, names)
    return {"original_feat": a, "square": f"({a})^2", "absolute": f"|{a}|"}.get(n, f"{n}({a})")


def _n_tree_nodes(model):
    if isinstance(model, Pipeline):
        model = model.steps[-1][1]
    if hasattr(model, "estimators_"):
        try:
            return int(sum(t.tree_.node_count for t in np.ravel(model.estimators_)))
        except AttributeError:
            return None
    if hasattr(model, "booster_"):                       # LightGBM: nodes = 2 * leaves - 1 per tree
        return int(sum(2 * t["num_leaves"] - 1 for t in model.booster_.dump_model()["tree_info"]))
    return None


def unwrap_extratrees(model):
    """Return the fitted ExtraTreesClassifier inside a TPOT result / pipeline."""
    m = getattr(model, "fitted_pipeline_", model)
    if isinstance(m, ExtraTreesClassifier):
        return m
    if isinstance(m, Pipeline):
        for _, step in m.steps:
            if isinstance(step, ExtraTreesClassifier):
                return step
    for attr in ("steps", "estimators_"):
        for step in getattr(m, attr, []) or []:
            s = step[1] if isinstance(step, tuple) else step
            try:
                return unwrap_extratrees(s)
            except TypeError:
                pass
    raise TypeError(f"no ExtraTreesClassifier inside {type(m)}")


def unwrap_tpot(model):
    """TPOT's fitted pipeline; a one-step pipeline is reduced to its estimator."""
    m = getattr(model, "fitted_pipeline_", model)
    while isinstance(m, Pipeline) and len(m.steps) == 1:
        m = m.steps[0][1]
    return m


def bigfeat_select_model(X, y, seed):
    """BigFeat's select_estimator (3-fold CV, keep the best candidate), with two changes:
    macro-F1 instead of its default binary F1, which fails on multiclass labels, and the LightGBM
    stabilising setting used everywhere else. Returns (fitted model, name, CV scores)."""
    from lightgbm import LGBMClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    cands = dict(lgb=LGBMClassifier(random_state=seed, **LGBM_PARAMS),
                 rf=RandomForestClassifier(random_state=seed, n_jobs=1),
                 lr=LogisticRegression(random_state=seed))
    scores = {n: float(cross_val_score(cands[n], X, y, cv=3, scoring="f1_macro").mean())
              for n in BIGFEAT_AUTO_CANDIDATES}
    best = max(scores, key=scores.get)
    return cands[best].fit(X, y), best, scores


TREE_SHAP_MODELS = ("ExtraTreesClassifier", "RandomForestClassifier", "DecisionTreeClassifier",
                    "LGBMClassifier", "XGBClassifier", "GradientBoostingClassifier")
LINEAR_SHAP_MODELS = ("LogisticRegression", "SGDClassifier", "LinearDiscriminantAnalysis")
# Models whose SHAP values explain the raw margin (log-odds) rather than the class probability.
MARGIN_SHAP_MODELS = ("LGBMClassifier", "XGBClassifier", "GradientBoostingClassifier") + LINEAR_SHAP_MODELS


def shap_output_space(model):
    """'margin' if the SHAP values of ``model`` explain raw margins, 'probability' otherwise
    (TreeSHAP on scikit-learn forests, and permutation SHAP on predict_proba)."""
    return "margin" if type(model).__name__ in MARGIN_SHAP_MODELS else "probability"


def model_output(model, X, space):
    """Per-class model output in the space its SHAP values explain."""
    if space == "probability":
        return model.predict_proba(X)
    if type(model).__name__ == "LGBMClassifier":
        out = model.predict(X, raw_score=True)
    elif type(model).__name__ == "XGBClassifier":
        out = model.predict(X, output_margin=True)
    else:
        out = model.decision_function(X)
    out = np.asarray(out)
    return np.stack([-out, out], axis=1) if out.ndim == 1 else out


def shap_values(model, Xs, X_background, seed):
    """SHAP values as (n, d, C). TreeSHAP for tree ensembles (as in every other configuration),
    exact linear SHAP for linear models, and model-agnostic permutation SHAP for anything else,
    including multi-step pipelines (explained end to end, in the input feature space)."""
    import shap
    name = type(model).__name__
    if name in TREE_SHAP_MODELS:
        sv = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent").shap_values(
            Xs, check_additivity=False)
        method = "tree"
    elif name in LINEAR_SHAP_MODELS:
        sv = shap.LinearExplainer(model, shap.maskers.Independent(X_background)).shap_values(Xs)
        method = "linear"
    else:
        ex = shap.PermutationExplainer(model.predict_proba, shap.maskers.Independent(X_background),
                                       seed=seed)
        sv = ex(Xs, max_evals=2 * Xs.shape[1] + 1 + 200, silent=True).values
        method = "permutation"
    sv = np.asarray(sv)
    if sv.ndim == 2:                                  # binary linear output -> two classes
        sv = np.stack([-sv, sv], axis=2)
    if sv.ndim == 3 and sv.shape[0] != Xs.shape[0]:   # (C, n, d) -> (n, d, C)
        sv = np.transpose(sv, (1, 2, 0))
    return sv, method


def performance_metrics(y_true, y_pred, proba, classes):
    labels = np.arange(len(classes))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fp = cm.sum(0) - np.diag(cm)
    fn = cm.sum(1) - np.diag(cm)
    tp = np.diag(cm)
    tn = cm.sum() - fp - fn - tp
    yb = label_binarize(y_true, classes=labels)
    if yb.shape[1] == 1:
        yb = np.hstack([1 - yb, yb])
    m = dict(
        accuracy=accuracy_score(y_true, y_pred),
        f1_macro=f1_score(y_true, y_pred, average="macro", zero_division=0),
        precision_macro=precision_score(y_true, y_pred, average="macro", zero_division=0),
        recall_macro=recall_score(y_true, y_pred, average="macro", zero_division=0),
        roc_auc_macro=roc_auc_score(yb, proba, average="macro", multi_class="ovr"),
        pr_auc_macro=average_precision_score(yb, proba, average="macro"),
        fpr_macro=float(np.mean(fp / np.maximum(fp + tn, 1))),
        fnr_macro=float(np.mean(fn / np.maximum(fn + tp, 1))),
    )
    per_class_f1 = f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    return m, cm, per_class_f1


def _norm(v):
    v = np.abs(np.asarray(v, dtype=float))
    s = v.sum()
    return v / s if s > 0 else np.full_like(v, 1.0 / len(v))


def complexity_metrics(global_imp):
    p = _norm(global_imp)
    ent = float(-(p[p > 0] * np.log(p[p > 0])).sum())
    cum = np.cumsum(np.sort(p)[::-1])
    return dict(
        effective_n_features=float(np.exp(ent)),
        normalized_entropy=ent / np.log(len(p)) if len(p) > 1 else 0.0,
        n_features_80pct=int(np.searchsorted(cum, 0.8) + 1),
        n_features_total=int(len(p)),
    )


def rank_agreement(a, b, k_list=(5, 10)):
    a, b = np.asarray(a, float), np.asarray(b, float)
    out = dict(kendall_tau=float(stats.kendalltau(a, b).statistic),
               spearman=float(stats.spearmanr(a, b).statistic))
    for k in k_list:
        k_ = min(k, len(a))
        ta, tb = set(np.argsort(-a)[:k_]), set(np.argsort(-b)[:k_])
        out[f"top{k}_jaccard"] = len(ta & tb) / len(ta | tb)
    return out


def deletion_insertion(model, X, y, order, baseline, n_random=N_RANDOM_ORDERS, rng=None):
    """Global deletion / insertion curves of macro-F1 when features are replaced by ``baseline``.

    Returns normalised AUCs for the explanation's order and for random orders.
    Faithful explanations give a LOW deletion AUC and a HIGH insertion AUC relative to random.
    """
    rng = rng or np.random.default_rng(0)
    d = X.shape[1]

    cuts = np.unique(np.round(np.linspace(0, d, min(d, MAX_CURVE_STEPS) + 1)).astype(int))

    def curves(ordr):
        ordr = np.asarray(ordr)
        dele, ins = [], []
        for k in cuts:
            Xd = X.copy()
            Xd[:, ordr[:k]] = baseline[ordr[:k]]
            Xi = np.tile(baseline, (X.shape[0], 1))
            Xi[:, ordr[:k]] = X[:, ordr[:k]]
            dele.append(f1_score(y, model.predict(Xd), average="macro"))
            ins.append(f1_score(y, model.predict(Xi), average="macro"))
        x = cuts / d
        return float(np.trapezoid(dele, x)), float(np.trapezoid(ins, x)), dele, ins

    del_auc, ins_auc, dele, ins = curves(list(order))
    rnd = [curves(list(rng.permutation(d)))[:2] for _ in range(n_random)]
    rd, ri = np.mean([r[0] for r in rnd]), np.mean([r[1] for r in rnd])
    return dict(deletion_auc=del_auc, insertion_auc=ins_auc,
                deletion_auc_random=float(rd), insertion_auc_random=float(ri),
                deletion_gain=float(rd - del_auc), insertion_gain=float(ins_auc - ri),
                deletion_curve=dele, insertion_curve=ins)


def faithfulness_correlation(model, X, local_attr, pred_cls, baseline, n_subsets, rng, space="probability"):
    """Bhatt et al. (2020): corr(sum of attributions of a random subset, drop in the predicted class's
    output when that subset is set to baseline). Averaged over instances. ``space`` must be the output
    the attributions explain: the probability, or the raw margin for boosting and linear models."""
    n, d = X.shape
    size = max(1, d // 4)
    rows, masks = [], []
    for i in range(n):
        for _ in range(n_subsets):
            s = rng.choice(d, size=size, replace=False)
            x = X[i].copy()
            x[s] = baseline[s]
            rows.append(x)
            masks.append(s)
    P = model_output(model, np.vstack(rows), space)
    p0 = model_output(model, X, space)
    corrs = []
    for i in range(n):
        c = pred_cls[i]
        sl = slice(i * n_subsets, (i + 1) * n_subsets)
        drop = p0[i, c] - P[sl, c]
        attr = np.array([local_attr[i, m].sum() for m in masks[sl]])
        if np.std(drop) > 1e-12 and np.std(attr) > 1e-12:
            corrs.append(stats.pearsonr(attr, drop).statistic)
    return float(np.mean(corrs)) if corrs else float("nan"), len(corrs)


def project_to_original(global_imp, n_generated, formulas, n_orig):
    """Map importances on [generated..., original...] back onto original features.
    A generated feature's importance is split over its leaf features in proportion to how
    often each leaf appears in the formula."""
    out = np.zeros(n_orig)
    out += global_imp[n_generated:n_generated + n_orig]
    for g in range(n_generated):
        leaves = formulas[g]["leaves"]
        if not leaves:
            continue
        w = global_imp[g] / len(leaves)
        for j in leaves:
            out[j] += w
    return out


# --------------------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------------------
@dataclass
class Task:
    dataset: str          # 'nbaiot', 'edgeiiot', 'ciciot2023'
    unit: str             # device id for N-BaIoT, 'all' otherwise
    feature_mode: str     # 'prebuilt20', 'all', 'infold20'


def run_one(task: Task, seed: int, config: str, out_dir: str, load_fn):
    assert config in CONFIGS
    rng = np.random.default_rng(seed)
    res = dict(dataset=task.dataset, unit=task.unit, feature_mode=task.feature_mode,
               seed=seed, config=config)

    # 1-2. data + balanced subsample drawn with this seed
    X_df, y_raw = load_fn(task, seed)
    feat_names = list(X_df.columns)
    le = LabelEncoder().fit(y_raw)
    y = le.transform(y_raw)
    classes = [str(c) for c in le.classes_]
    X = X_df.to_numpy(dtype=np.float64)

    # 3. split first
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)

    # 4. everything below is fit on the training split only
    t_prep = _timer()
    res["scaled"] = config in NEEDS_SCALING
    if res["scaled"]:
        scaler = MinMaxScaler().fit(X_tr)
        X_tr, X_te = scaler.transform(X_tr), np.clip(scaler.transform(X_te), -1, 2)
    if task.feature_mode.startswith("infold"):
        k = int(task.feature_mode.replace("infold", ""))
        keep = select_topk_in_fold(X_tr, y_tr, k, seed)
        X_tr, X_te = X_tr[:, keep], X_te[:, keep]
        feat_names = [feat_names[j] for j in keep]
        res["selected_features"] = feat_names
    prep_wall, prep_cpu = t_prep()
    n_orig = len(feat_names)

    formulas, n_gen = [], 0
    t_fe = _timer()
    if config in BIGFEAT_CONFIGS:
        from bigfeat.bigfeat_base import BigFeat
        bf = BigFeat(task_type="classification")
        bf.n_jobs = 1
        # BigFeat subsamples rows with the *global* numpy RNG (bigfeat_base.get_feature_importances),
        # which its random_state argument does not control; seed it so runs are reproducible.
        np.random.seed(seed)
        bf.fit(X_tr, y_tr, random_state=seed, **BIGFEAT_PARAMS)
        X_tr_m, X_te_m = bf.transform(X_tr), bf.transform(X_te)
        formulas = bigfeat_formula_stats(bf, feat_names)
        n_gen = len(formulas)
        if config.endswith("_raw"):       # generated features + unscaled originals
            X_tr_m = np.hstack([X_tr_m[:, :n_gen], X_tr])
            X_te_m = np.hstack([X_te_m[:, :n_gen], X_te])
        model_names = [f["name"] for f in formulas] + feat_names
    else:
        X_tr_m, X_te_m = X_tr, X_te
        model_names = feat_names
    fe_wall, fe_cpu = t_fe()

    t_fit = _timer()
    if config in ("tpot", "both", "both_raw"):
        from tpot import TPOTClassifier
        from tpot.old_config_utils import convert_config_dict_to_choicepipeline
        tp = TPOTClassifier(search_space=convert_config_dict_to_choicepipeline(TPOT_SPACE),
                            random_state=seed, **TPOT_PARAMS)
        tp.fit(X_tr_m, y_tr)
        model = unwrap_extratrees(tp)
        res["tpot_params"] = {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                              for k, v in model.get_params().items()}
    elif config == "tpot_auto":
        from tpot import TPOTClassifier
        tp = TPOTClassifier(search_space="linear", random_state=seed, **TPOT_PARAMS)
        tp.fit(X_tr_m, y_tr)
        model = unwrap_tpot(tp)
        res["tpot_pipeline"] = str(model)
        res["tpot_model"] = type(model.steps[-1][1] if isinstance(model, Pipeline) else model).__name__
    elif config == "bigfeat_auto":
        model, res["bigfeat_model"], res["bigfeat_model_cv"] = bigfeat_select_model(X_tr_m, y_tr, seed)
    elif config == "lgbm_cv":
        from lightgbm import LGBMClassifier
        from sklearn.model_selection import GridSearchCV, StratifiedKFold
        base = {k: v for k, v in LGBM_PARAMS.items() if k != "max_delta_step"}
        gs = GridSearchCV(LGBMClassifier(random_state=seed, **base), {"max_delta_step": list(LGBM_CV_GRID)},
                          scoring="f1_macro", n_jobs=1, refit=True,
                          cv=StratifiedKFold(3, shuffle=True, random_state=seed)).fit(X_tr_m, y_tr)
        model = gs.best_estimator_
        res["lgbm_cv_choice"] = float(gs.best_params_["max_delta_step"])
        res["lgbm_cv_scores"] = {str(v): float(m) for v, m in zip(LGBM_CV_GRID, gs.cv_results_["mean_test_score"])}
    elif config == "flaml":
        from flaml import AutoML
        automl = AutoML()
        automl.fit(X_tr_m, y_tr, seed=seed, **FLAML_PARAMS)
        model = automl.model.estimator    # the fitted final model; predicts exactly as automl does
        res["flaml_estimator"] = automl.best_estimator
        res["flaml_config"] = {k: (float(v) if isinstance(v, (np.floating, float)) else v)
                               for k, v in automl.best_config.items()}
        res["flaml_trials"] = len(automl.config_history)
        res["flaml_model"] = type(model).__name__
    elif config == "lgbm":
        from lightgbm import LGBMClassifier
        model = LGBMClassifier(random_state=seed, **LGBM_PARAMS).fit(X_tr_m, y_tr)
    elif config == "split_grid":
        from sklearn.model_selection import GridSearchCV, StratifiedKFold
        gs = GridSearchCV(ExtraTreesClassifier(random_state=seed, **ET_PARAMS),
                          {"min_samples_split": list(SPLIT_GRID)}, scoring="f1_macro", n_jobs=1, refit=True,
                          cv=StratifiedKFold(3, shuffle=True, random_state=seed)).fit(X_tr_m, y_tr)
        model = gs.best_estimator_
        res["split_grid_choice"] = int(gs.best_params_["min_samples_split"])
        res["split_grid_cv"] = {str(v): float(m) for v, m in zip(SPLIT_GRID, gs.cv_results_["mean_test_score"])}
    elif config == "split10":
        model = ExtraTreesClassifier(random_state=seed, min_samples_split=10, **ET_PARAMS).fit(X_tr_m, y_tr)
    else:
        model = ExtraTreesClassifier(random_state=seed, **ET_PARAMS).fit(X_tr_m, y_tr)
    fit_wall, fit_cpu = t_fit()

    # 5a. performance
    t_inf = time.perf_counter()
    proba = model.predict_proba(X_te_m)
    infer_s = time.perf_counter() - t_inf
    y_pred = proba.argmax(1)
    perf, cm, pcf1 = performance_metrics(y_te, y_pred, proba, classes)
    res.update(perf)
    res["per_class_f1"] = dict(zip(classes, map(float, pcf1)))
    res["confusion_matrix"] = cm.tolist()
    res["classes"] = classes

    # 5b. cost
    counter = _ByteCounter()
    pickle.dump(model, counter, protocol=pickle.HIGHEST_PROTOCOL)
    model_bytes = counter.n
    res.update(
        n_train=int(len(y_tr)), n_test=int(len(y_te)),
        n_features_original=n_orig, n_features_generated=n_gen,
        n_features_model=int(X_tr_m.shape[1]),
        time_prep_s=prep_wall, time_featgen_s=fe_wall, time_fit_s=fit_wall,
        train_time_s=prep_wall + fe_wall + fit_wall,
        train_cpu_s=prep_cpu + fe_cpu + fit_cpu,
        infer_ms_per_1k=1000 * infer_s / len(y_te) * 1000,
        model_size_mb=model_bytes / 1e6,
        n_tree_nodes=_n_tree_nodes(model),
    )
    if formulas:
        res["formulas"] = formulas
        res["formula_mean_depth"] = float(np.mean([f["depth"] for f in formulas]))
        res["formula_max_depth"] = int(max(f["depth"] for f in formulas))
        res["formula_mean_ops"] = float(np.mean([f["n_ops"] for f in formulas]))
        res["formula_total_ops"] = int(sum(f["n_ops"] for f in formulas))
    else:
        res.update(formula_mean_depth=0.0, formula_max_depth=0, formula_mean_ops=0.0, formula_total_ops=0)

    # 5c. explanation quality (all on test rows)
    t_xai = _timer()
    idx_eval = rng.choice(len(y_te), size=min(N_EVAL_XAI, len(y_te)), replace=False)
    idx_exp = idx_eval[:N_EXPLAIN]
    Xe, ye = X_te_m[idx_eval], y_te[idx_eval]
    Xs = X_te_m[idx_exp]
    baseline = X_tr_m.mean(0)

    X_bg = X_tr_m[rng.choice(len(X_tr_m), 100, replace=False)]
    sv, res["shap_method"] = shap_values(model, Xs, X_bg, seed)
    pred_cls = model.predict(Xs)
    local_pred = sv[np.arange(len(idx_exp)), :, pred_cls]        # (n, d) attribution to predicted class
    shap_global = np.abs(sv).mean(axis=(0, 2))

    pfi = permutation_importance(model, Xe, ye, scoring="f1_macro", n_repeats=PFI_REPEATS,
                                 random_state=seed, n_jobs=1).importances_mean
    pfi_pos = np.clip(pfi, 0, None)

    from lime.lime_tabular import LimeTabularExplainer
    lime_exp = LimeTabularExplainer(X_tr_m[rng.choice(len(X_tr_m), 5000, replace=False)],
                                    feature_names=model_names, discretize_continuous=False,
                                    random_state=seed, mode="classification")
    lime_local = np.zeros((N_LIME, X_te_m.shape[1]))
    for r in range(N_LIME):
        c = int(pred_cls[r])
        e = lime_exp.explain_instance(Xs[r], model.predict_proba, labels=(c,),
                                      num_features=X_te_m.shape[1], num_samples=1000)
        for j, w in e.as_map()[c]:
            lime_local[r, j] = w
    lime_global = np.abs(lime_local).mean(0)

    res["xai_shap_vs_pfi"] = rank_agreement(shap_global, pfi_pos)
    res["xai_shap_vs_lime"] = rank_agreement(shap_global, lime_global)
    res["xai_pfi_vs_lime"] = rank_agreement(pfi_pos, lime_global)
    loc = [stats.spearmanr(np.abs(local_pred[r]), np.abs(lime_local[r])).statistic for r in range(N_LIME)]
    res["xai_local_shap_lime_spearman"] = float(np.nanmean(loc))

    res["xai_complexity_shap"] = complexity_metrics(shap_global)
    loc_ent = [complexity_metrics(local_pred[r])["normalized_entropy"] for r in range(len(idx_exp))]
    res["xai_local_normalized_entropy"] = float(np.mean(loc_ent))

    order = np.argsort(-shap_global)
    di = deletion_insertion(model, Xe, ye, order, baseline, rng=np.random.default_rng(seed))
    res["xai_deletion_insertion_shap"] = di
    # Faithfulness is measured in the output space the SHAP values explain (margin for boosting).
    space = shap_output_space(model)
    fc, fc_n = faithfulness_correlation(model, Xs[:100], local_pred[:100], pred_cls[:100], baseline,
                                        N_FAITH_SUBSETS, np.random.default_rng(seed), space=space)
    res["shap_output_space"] = space
    res["xai_faithfulness_corr"] = fc
    res["xai_faithfulness_n"] = fc_n
    if space != "probability":      # the earlier, mismatched measure, kept for comparison
        res["xai_faithfulness_corr_proba"] = faithfulness_correlation(
            model, Xs[:100], local_pred[:100], pred_cls[:100], baseline, N_FAITH_SUBSETS,
            np.random.default_rng(seed))[0]

    # bootstrap stability of the global SHAP ranking over explained instances
    taus = []
    brng = np.random.default_rng(seed + 1000)
    for _ in range(50):
        b = brng.choice(len(idx_exp), len(idx_exp), replace=True)
        taus.append(stats.kendalltau(shap_global, np.abs(sv[b]).mean(axis=(0, 2))).statistic)
    res["xai_bootstrap_tau_mean"] = float(np.mean(taus))

    # share of attribution carried by generated features, and projection back to raw features
    res["xai_generated_share"] = float(_norm(shap_global)[:n_gen].sum()) if n_gen else 0.0
    shap_orig = project_to_original(shap_global, n_gen, formulas, n_orig) if n_gen else shap_global
    pfi_orig = project_to_original(pfi_pos, n_gen, formulas, n_orig) if n_gen else pfi_pos
    xai_wall, _ = t_xai()
    res["time_xai_s"] = xai_wall

    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, f"{config}")
    np.savez_compressed(stem + "_attr.npz", shap_global=shap_global, pfi=pfi, lime_global=lime_global,
                        shap_orig=shap_orig, pfi_orig=pfi_orig, model_feature_names=np.array(model_names),
                        orig_feature_names=np.array(feat_names))
    with open(stem + ".json", "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    return res
