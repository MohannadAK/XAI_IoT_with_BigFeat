"""
Aggregate per-seed results into the revision's tables, statistics and figures.

  python experiments/analyze.py            # everything found under $REVISION_OUT (default results/round3)

Statistics
  * per unit (device or dataset) and config: mean, std, 95% t-CI over seeds
  * per unit, config vs original: Nadeau-Bengio corrected resampled t-test (repeated 70/30
    splits share training rows, so the naive paired t-test is anti-conservative) and the
    Bayesian correlated t-test (Corani & Benavoli 2015) with a ROPE
  * across the 9 N-BaIoT devices, on seed-averaged scores: Wilcoxon signed-rank (Holm-corrected),
    Friedman + Nemenyi omnibus over the four configs, and the Bayesian signed-rank test
    (Benavoli et al. 2017) with a ROPE
"""
import glob
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.environ.get("REVISION_OUT", os.path.join(REPO, "results", "round3"))
OUT = os.path.join(RES, "summary")
CONFIGS = ["original", "split10", "split_grid", "lgbm", "bigfeat", "bigfeat_raw", "tpot", "both", "both_raw", "bigfeat_auto", "tpot_auto"]
# Figure legend labels: "features + model". BigFeat returns its generated features plus min-max scaled
# originals; the "_raw" variants swap in the unscaled originals.
LABEL = dict(original="Baseline: ExtraTrees (default)",
             split10="ExtraTrees, min_samples_split=10",
             split_grid="ExtraTrees, min_samples_split by CV grid {2, 5, 10, 20}",
             lgbm="LightGBM (default)",
             bigfeat="BigFeat + scaled originals \u2192 ExtraTrees",
             bigfeat_raw="BigFeat + unscaled originals \u2192 ExtraTrees",
             tpot="TPOT, ExtraTrees-only search",
             both="BigFeat + scaled originals \u2192 TPOT (ExtraTrees-only)",
             both_raw="BigFeat + unscaled originals \u2192 TPOT (ExtraTrees-only)",
             bigfeat_auto="BigFeat + scaled originals \u2192 BigFeat-selected model\n(LightGBM / RF / LogReg, 3-fold CV)",
             tpot_auto="TPOT, full default search\n(scaler \u2192 selector \u2192 transformer \u2192 17 classifiers)")
DATASET_LABEL = dict(nbaiot="N-BaIoT", edgeiiot="Edge-IIoTset", ciciot2023="CIC-IoT-2023 (8 classes)",
                     ciciot34="CIC-IoT-2023 (34 classes)", unswnb15="UNSW-NB15 (10 classes)")
FEATURE_MODE_LABEL = dict(all="all features", infold20="top-20 features selected in-fold",
                          prebuilt20="manuscript's pre-built top-20 features")
ROPE_F1 = 0.005
TEST_FRAC = 0.3

PERF = ["f1_macro", "accuracy", "precision_macro", "recall_macro", "roc_auc_macro", "pr_auc_macro",
        "fpr_macro", "fnr_macro"]
COST = ["train_time_s", "train_cpu_s", "time_featgen_s", "time_fit_s", "infer_ms_per_1k",
        "model_size_mb", "n_tree_nodes", "n_features_model"]
XAI = ["agree_shap_pfi_tau", "agree_shap_lime_tau", "agree_pfi_lime_tau", "agree_shap_pfi_top5",
       "agree_shap_pfi_top10", "local_shap_lime_rho", "faithfulness_corr", "deletion_gain",
       "insertion_gain", "effective_n_features", "n_features_80pct", "local_entropy",
       "bootstrap_tau", "generated_share", "formula_mean_depth", "formula_mean_ops",
       "n_features_generated"]


# ------------------------------------------------------------------------------- loading
def load_runs():
    rows, attrs = [], {}
    for f in glob.glob(os.path.join(RES, "*", "*", "unit*", "seed*", "*.json")):
        r = json.load(open(f))
        row = {k: r.get(k) for k in ["dataset", "feature_mode", "unit", "seed", "config"] + PERF + COST}
        row.update(
            agree_shap_pfi_tau=r["xai_shap_vs_pfi"]["kendall_tau"],
            agree_shap_lime_tau=r["xai_shap_vs_lime"]["kendall_tau"],
            agree_pfi_lime_tau=r["xai_pfi_vs_lime"]["kendall_tau"],
            agree_shap_pfi_top5=r["xai_shap_vs_pfi"]["top5_jaccard"],
            agree_shap_pfi_top10=r["xai_shap_vs_pfi"]["top10_jaccard"],
            local_shap_lime_rho=r["xai_local_shap_lime_spearman"],
            faithfulness_corr=r["xai_faithfulness_corr"],
            deletion_gain=r["xai_deletion_insertion_shap"]["deletion_gain"],
            insertion_gain=r["xai_deletion_insertion_shap"]["insertion_gain"],
            effective_n_features=r["xai_complexity_shap"]["effective_n_features"],
            n_features_80pct=r["xai_complexity_shap"]["n_features_80pct"],
            local_entropy=r["xai_local_normalized_entropy"],
            bootstrap_tau=r["xai_bootstrap_tau_mean"],
            generated_share=r["xai_generated_share"],
            formula_mean_depth=r["formula_mean_depth"], formula_mean_ops=r["formula_mean_ops"],
            n_features_generated=r["n_features_generated"],
            n_classes=len(r["classes"]),
        )
        rows.append(row)
        z = np.load(f.replace(".json", "_attr.npz"))
        key = tuple(row[k] for k in ["dataset", "feature_mode", "unit", "config", "seed"])
        attrs[key] = dict(zip(z["orig_feature_names"].tolist(), z["shap_orig"]))
    df = pd.DataFrame(rows)
    if not df.empty:
        df["seed"] = df["seed"].astype(int)
        df["unit"] = df["unit"].astype(str)
    return df, attrs


def cross_seed_stability(attrs, seeds=None):
    """Mean pairwise Kendall tau of the (projected) global SHAP importance across seeds, over the
    union of feature names (features absent in a seed count as zero importance).
    ``seeds`` restricts the computation to a common seed set so configurations with different
    numbers of seeds are compared fairly."""
    groups = {}
    for (ds, fm, u, c, s), v in attrs.items():
        if seeds is not None and s not in seeds:
            continue
        groups.setdefault((ds, fm, u, c), []).append(v)
    out = []
    for (ds, fm, u, c), vs in groups.items():
        if len(vs) < 2:
            continue
        names = sorted(set().union(*[v.keys() for v in vs]))
        M = np.array([[v.get(n, 0.0) for n in names] for v in vs])
        taus = [stats.kendalltau(M[i], M[j]).statistic for i, j in itertools.combinations(range(len(vs)), 2)]
        k = min(5, len(names))
        tops = [set(np.argsort(-m)[:k]) for m in M]
        jac = [len(a & b) / len(a | b) for a, b in itertools.combinations(tops, 2)]
        out.append(dict(dataset=ds, feature_mode=fm, unit=u, config=c, seed_stability_tau=np.mean(taus),
                        seed_stability_top5=np.mean(jac), n_seeds=len(vs)))
    return pd.DataFrame(out)


# ------------------------------------------------------------------------------- statistics
def ci95(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return np.nan
    return stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))


def ci95_corrected(diff, test_frac=TEST_FRAC):
    """95% interval half-width for a mean difference over repeated 70/30 splits, with the
    Nadeau-Bengio variance correction (consistent with corrected_t)."""
    d = np.asarray(diff, float)
    n = len(d)
    if n < 2:
        return np.nan
    rho = test_frac / (1 - test_frac)
    return stats.t.ppf(0.975, n - 1) * np.sqrt((1 / n + rho) * d.var(ddof=1))


def corrected_t(diff, test_frac=TEST_FRAC):
    """Nadeau & Bengio (2003) corrected resampled t-test."""
    d = np.asarray(diff, float)
    n = len(d)
    if n < 2 or d.std(ddof=1) == 0:
        return np.nan, np.nan
    rho = test_frac / (1 - test_frac)
    t = d.mean() / np.sqrt((1 / n + rho) * d.var(ddof=1))
    return t, 2 * stats.t.sf(abs(t), n - 1)


def bayes_corr_t(diff, rope, test_frac=TEST_FRAC):
    """Corani & Benavoli (2015) Bayesian correlated t-test -> P(left), P(rope), P(right)."""
    d = np.asarray(diff, float)
    n = len(d)
    if n < 2:
        return np.nan, np.nan, np.nan
    rho = test_frac
    scale = np.sqrt((1 / n + rho / (1 - rho)) * max(d.var(ddof=1), 1e-24))
    dist = stats.t(df=n - 1, loc=d.mean(), scale=scale)
    pl = dist.cdf(-rope)
    pr = 1 - dist.cdf(rope)
    return pl, 1 - pl - pr, pr


def bayes_signed_rank(diff, rope, n_samples=50000, seed=0):
    """Benavoli et al. (2017) Bayesian signed-rank test with a Dirichlet-process prior."""
    d = np.concatenate([[0.0], np.asarray(diff, float)])      # pseudo-observation at 0 (prior)
    n = len(d)
    rng = np.random.default_rng(seed)
    w = rng.dirichlet(np.ones(n), n_samples)
    s = d[:, None] + d[None, :]
    left = (s < -2 * rope).astype(float)
    right = (s > 2 * rope).astype(float)
    rp = 1 - left - right
    pl = np.einsum("ki,ij,kj->k", w, left, w)
    pr = np.einsum("ki,ij,kj->k", w, right, w)
    pm = np.einsum("ki,ij,kj->k", w, rp, w)
    win = np.argmax(np.stack([pl, pm, pr]), axis=0)
    return (win == 0).mean(), (win == 1).mean(), (win == 2).mean()


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0
    for rank, i in enumerate(order):
        running = max(running, min(1, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj


# ------------------------------------------------------------------------------- tables
def fmt(m, s, nd=4):
    if pd.isna(m):
        return "--"
    return f"{m:.{nd}f} $\\pm$ {s:.{nd}f}" if not pd.isna(s) else f"{m:.{nd}f}"


def summarize(df, stab):
    g = df.groupby(["dataset", "feature_mode", "unit", "config"])
    agg = g[PERF + COST + XAI].agg(["mean", "std", "count"])
    agg.columns = [f"{a}__{b}" for a, b in agg.columns]
    agg = agg.reset_index()
    for m in PERF + COST + XAI:
        agg[f"{m}__ci95"] = g[m].apply(ci95).values
    if not stab.empty:
        agg = agg.merge(stab, on=["dataset", "feature_mode", "unit", "config"], how="left")
    return agg


def per_unit_tests(df, metric="f1_macro", rope=ROPE_F1):
    out = []
    for (ds, fm, u), sub in df.groupby(["dataset", "feature_mode", "unit"]):
        base = sub[sub.config == "original"].set_index("seed")[metric]
        for c in [c for c in CONFIGS if c != "original"]:
            other = sub[sub.config == c].set_index("seed")[metric]
            common = base.index.intersection(other.index)
            if len(common) < 2:
                continue
            d = (other.loc[common] - base.loc[common]).values
            t, p = corrected_t(d)
            pl, pm, pr = bayes_corr_t(d, rope)
            out.append(dict(dataset=ds, feature_mode=fm, unit=u, config=c, metric=metric, n_seeds=len(d),
                            mean_diff=d.mean(), std_diff=d.std(ddof=1), wins=int((d > 0).sum()),
                            corrected_t=t, p_corrected=p, p_worse=pl, p_rope=pm, p_better=pr))
    out = pd.DataFrame(out)
    if not out.empty:       # Holm over the comparisons with the baseline within each task
        out["p_holm"] = out.groupby(["dataset", "feature_mode", "unit"]).p_corrected.transform(
            lambda p: holm(p.fillna(1).values))
    return out


def across_device_tests(df, dataset="nbaiot", feature_mode=None, metric="f1_macro", rope=ROPE_F1):
    sub = df[(df.dataset == dataset) & (df.feature_mode == feature_mode)]
    piv = sub.groupby(["unit", "config"])[metric].mean().unstack()
    piv = piv.dropna(axis=1, how="all").dropna()
    res = dict(dataset=dataset, feature_mode=feature_mode, metric=metric, n_units=len(piv))
    if len(piv) < 3:
        return res, None
    cfgs = [c for c in CONFIGS if c in piv.columns]
    if len(cfgs) >= 3:
        chi, p = stats.friedmanchisquare(*[piv[c] for c in cfgs])
        res.update(friedman_chi2=chi, friedman_p=p, friedman_k=len(cfgs))
        try:
            import scikit_posthocs as sp
            nem = sp.posthoc_nemenyi_friedman(piv[cfgs].values)
            nem.index = nem.columns = cfgs
            res["nemenyi"] = nem.round(4).to_dict()
        except Exception as e:  # pragma: no cover
            res["nemenyi_error"] = str(e)
        ranks = piv[cfgs].rank(axis=1, ascending=False).mean()
        res["mean_ranks"] = ranks.to_dict()
    pw = []
    for c in [c for c in cfgs if c != "original"]:
        d = (piv[c] - piv["original"]).values
        try:
            w = stats.wilcoxon(d, zero_method="wilcox")
            wp = w.pvalue
            ws = w.statistic
        except ValueError:
            ws, wp = np.nan, 1.0
        pl, pm, pr = bayes_signed_rank(d, rope)
        pw.append(dict(config=c, mean_diff=d.mean(), median_diff=np.median(d), wins=int((d > 0).sum()),
                       losses=int((d < 0).sum()), wilcoxon_W=ws, wilcoxon_p=wp,
                       p_worse=pl, p_rope=pm, p_better=pr))
    pw = pd.DataFrame(pw)
    if not pw.empty:
        pw["wilcoxon_p_holm"] = holm(pw["wilcoxon_p"].values)
    return res, pw


# ------------------------------------------------------------------------------- Pareto
def pareto(summary, rope=ROPE_F1):
    """Cost-explainability frontier per task, with accuracy as a tolerance constraint.

    Objectives: Macro-F1 (up), training time (down), cross-seed explanation stability on the
    seeds shared by all configurations (up). Configuration j dominates i when j is no worse on
    all three and strictly better on at least one, where F1 differences inside the ROPE count as
    ties (neither better nor worse)."""
    rows = []
    stab_col = "seed_stability_tau"
    for key, sub in summary.groupby(["dataset", "feature_mode", "unit"]):
        sub = sub.reset_index(drop=True)
        f1 = sub["f1_macro__mean"].values
        t = sub["train_time_s__mean"].values
        st = sub[stab_col].values
        best = f1.max()
        for i in range(len(sub)):
            dom_by = []
            for j in range(len(sub)):
                if j == i:
                    continue
                f1_no_worse = f1[j] >= f1[i] - rope
                f1_better = f1[j] > f1[i] + rope
                no_worse = f1_no_worse and t[j] <= t[i] and st[j] >= st[i]
                better = f1_better or t[j] < t[i] or st[j] > st[i]
                if no_worse and better:
                    dom_by.append(sub.loc[j, "config"])
            rows.append(dict(dataset=key[0], feature_mode=key[1], unit=key[2], config=sub.loc[i, "config"],
                             f1=f1[i], f1_gap_to_best=best - f1[i], within_rope_of_best=bool(f1[i] >= best - rope),
                             train_time_s=t[i], stability_all_seeds=st[i],
                             pareto_optimal=not dom_by, dominated_by=",".join(dom_by)))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------- figures
def figures(summary, df, par):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
    colors = dict(original="#4C72B0", split10="#DD8452", split_grid="#A0522D", lgbm="#937860", bigfeat="#55A868", bigfeat_raw="#64B5CD",
                  tpot="#C44E52", both="#8172B2", both_raw="#B5559A",
                  bigfeat_auto="#555555", tpot_auto="#A68A00")
    markers = dict(original="o", split10="v", split_grid="<", lgbm="P", bigfeat="s", bigfeat_raw="X", tpot="^", both="D", both_raw="*", bigfeat_auto="p", tpot_auto="h")
    for (ds, fm), sub in summary.groupby(["dataset", "feature_mode"]):
        pf = par[(par.dataset == ds) & (par.feature_mode == fm)].set_index(["unit", "config"])
        fig, ax = plt.subplots(figsize=(9.6, 4.2))
        for u, su in sub.groupby("unit"):
            su = su.set_index("config")
            if "original" in su.index:
                for c in [c for c in CONFIGS if c != "original"]:
                    if c in su.index:
                        ax.plot([su.loc["original", "train_time_s__mean"], su.loc[c, "train_time_s__mean"]],
                                [su.loc["original", "seed_stability_tau"],
                                 su.loc[c, "seed_stability_tau"]], color="#bbbbbb", lw=0.6, zorder=0)
        handles = []
        for c in CONFIGS:
            s_ = sub[sub.config == c]
            if s_.empty:
                continue
            opt = np.array([bool(pf.loc[(u, c), "pareto_optimal"]) for u in s_.unit])
            x, y = s_["train_time_s__mean"].values, s_["seed_stability_tau"].values
            ax.scatter(x[opt], y[opt], c=colors[c], marker=markers[c], s=40, edgecolors="none")
            ax.scatter(x[~opt], y[~opt], facecolors="none", edgecolors=colors[c], marker=markers[c], s=40)
            handles.append(Line2D([], [], ls="none", marker=markers[c], color=colors[c], label=LABEL[c]))
        if sub.unit.nunique() == 1:        # single-task dataset: show the F1 each point buys
            for _, r in sub.iterrows():
                ax.annotate(f"F1 {r['f1_macro__mean']:.3f}", (r["train_time_s__mean"], r["seed_stability_tau"]),
                            textcoords="offset points", xytext=(6, 4), fontsize=7)
        ax.set_xscale("log")
        ax.set_xlabel("Training time incl. feature generation / search (s, log scale)")
        ax.set_ylabel("Cross-seed stability of SHAP ranking\n(mean Kendall τ, seeds 0–2)")
        ax.grid(alpha=0.3)
        ax.legend(handles=handles, fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1))
        ax.set_title(f"{DATASET_LABEL.get(ds, ds)}, {FEATURE_MODE_LABEL.get(fm, fm)}\n"
                     f"filled = Pareto-optimal on Macro-F1, training time and stability "
                     f"(F1 within ±{ROPE_F1} counts as a tie); labels give mean Macro-F1", fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "figures", f"pareto_{ds}_{fm}.pdf"))
        plt.close(fig)

    # F1 difference vs original, per unit, with 95% CI over seeds
    for (ds, fm), sub in df.groupby(["dataset", "feature_mode"]):
        units = sorted(sub.unit.unique(), key=lambda x: (len(x), x))
        fig, ax = plt.subplots(figsize=(max(4, 0.8 * len(units) + 2) + 4.6, 3.8))
        others = [c for c in CONFIGS if c != "original" and c in set(sub.config)]
        for k, c in enumerate(others):
            ms, cs = [], []
            for u in units:
                b = sub[(sub.unit == u) & (sub.config == "original")].set_index("seed").f1_macro
                o = sub[(sub.unit == u) & (sub.config == c)].set_index("seed").f1_macro
                d = (o - b).dropna()
                ms.append(d.mean() if len(d) else np.nan)
                cs.append(ci95_corrected(d) if len(d) > 1 else 0)
            x = np.arange(len(units)) + (k - (len(others) - 1) / 2) * (0.8 / max(len(others), 1))
            ax.errorbar(x, ms, yerr=cs, fmt=markers[c], color=colors[c], label=LABEL[c], capsize=2, ms=5)
        ax.axhspan(-ROPE_F1, ROPE_F1, color="#dddddd", alpha=0.6, label=f"Practical equivalence (±{ROPE_F1})")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(range(len(units)))
        ax.set_xticklabels([f"D{u}" if ds == "nbaiot" else DATASET_LABEL.get(ds, ds) for u in units])
        ax.set_ylabel("ΔMacro-F1 vs baseline\n(mean over 5 seeds, corrected 95% CI)")
        ax.set_title(f"{DATASET_LABEL.get(ds, ds)}, {FEATURE_MODE_LABEL.get(fm, fm)}", fontsize=8)
        ax.legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1))
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "figures", f"f1_delta_{ds}_{fm}.pdf"))
        plt.close(fig)


# ------------------------------------------------------------------------------- main
def main():
    os.makedirs(OUT, exist_ok=True)
    df, attrs = load_runs()
    if df.empty:
        print("no runs found")
        return
    df.to_csv(os.path.join(OUT, "all_runs.csv"), index=False)
    stab = cross_seed_stability(attrs)          # every configuration has the same 5 seeds
    summary = summarize(df, stab)
    summary.to_csv(os.path.join(OUT, "summary_by_unit_config.csv"), index=False)

    tests = pd.concat([per_unit_tests(df, m, r) for m, r in
                       [("f1_macro", ROPE_F1), ("agree_shap_pfi_tau", 0.05), ("faithfulness_corr", 0.05)]])
    tests.to_csv(os.path.join(OUT, "per_unit_tests.csv"), index=False)

    omni = []
    for (ds, fm) in df[["dataset", "feature_mode"]].drop_duplicates().itertuples(index=False):
        for m, rope in [("f1_macro", ROPE_F1), ("agree_shap_pfi_tau", 0.05), ("faithfulness_corr", 0.05),
                        ("train_time_s", 0.0)]:
            res, pw = across_device_tests(df, ds, fm, m, rope)
            if pw is not None:
                pw.insert(0, "metric", m)
                pw.insert(0, "feature_mode", fm)
                pw.insert(0, "dataset", ds)
                omni.append((res, pw))
    if omni:
        pd.concat([p for _, p in omni]).to_csv(os.path.join(OUT, "across_unit_pairwise.csv"), index=False)
        json.dump([r for r, _ in omni], open(os.path.join(OUT, "across_unit_omnibus.json"), "w"),
                  indent=1, default=float)

    par = pareto(summary)
    par.to_csv(os.path.join(OUT, "pareto.csv"), index=False)
    figures(summary, df, par)
    print("\nPareto-optimal count per dataset / feature setting (tasks on frontier / tasks):")
    print(par.groupby(["dataset", "feature_mode", "config"]).pareto_optimal.agg(["sum", "count"]).unstack())

    counts = df.groupby(["dataset", "feature_mode", "config"]).size().unstack(fill_value=0)
    print("runs per dataset/feature_mode/config:\n", counts, "\n")
    cols = ["f1_macro__mean", "f1_macro__std", "train_time_s__mean", "agree_shap_pfi_tau__mean",
            "faithfulness_corr__mean", "effective_n_features__mean"]
    cols += [c for c in ["seed_stability_tau"] if c in summary.columns]
    with pd.option_context("display.width", 200, "display.max_rows", 500):
        print(summary[["dataset", "feature_mode", "unit", "config"] + cols].round(4).to_string(index=False))
        if not tests.empty:
            print("\nper-unit tests (F1):")
            print(tests[tests.metric == "f1_macro"].round(4).to_string(index=False))
        for r, p in omni:
            print("\n", {k: v for k, v in r.items() if k != "nemenyi"})
            print(p.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
