"""
Figures for the manuscript, drawn at their printed size so text stays legible.

The sn-jnl single-column text width is 31pc (about 5.15 in); both figures are drawn at that width
and included at width=\\linewidth, so the font sizes below are the sizes on the page.

Reads results/round2 (N-BaIoT, Edge-IIoTset) and results/round3 (UNSW-NB15) summaries and writes
    results/figures/f1_delta.pdf
    results/figures/pareto.pdf
"""
import os
import shutil
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze as A  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R2 = os.path.join(REPO, "results", "round2", "summary")
R3 = os.path.join(REPO, "results", "round3", "summary")
OUT = os.path.join(REPO, "results", "figures")
# The (untracked) manuscript keeps its own copy; refresh it when present.
MANUSCRIPT_FIGS = os.path.join(REPO, "manuscript", "figures", "F1 - Time")
WIDTH = 31 / 6.0225          # 31pc in inches

plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8.5, "xtick.labelsize": 7.5,
                     "ytick.labelsize": 8, "legend.fontsize": 7.5, "pdf.fonttype": 42})

# Names as in the manuscript's configuration table (Table 2).
NAME = dict(original="Baseline (ExtraTrees)", split10="ExtraTrees, split 10", lgbm="LightGBM",
            bigfeat="BigFeat (scaled)", bigfeat_raw="BigFeat (unscaled)", tpot="TPOT (ExtraTrees)",
            both="BigFeat (scaled) + TPOT", both_raw="BigFeat (unscaled) + TPOT",
            bigfeat_auto="BigFeat, model chosen", tpot_auto="TPOT, full search")
ORDER = ["split10", "lgbm", "bigfeat", "bigfeat_raw", "tpot", "both", "both_raw", "bigfeat_auto", "tpot_auto"]
# One colour per kind of change: cheap manual choice, BigFeat, TPOT, both, tool picks the model.
GROUP = dict(original="#4D4D4D", split10="#1F5FA8", lgbm="#1F5FA8", bigfeat="#2E8B3E", bigfeat_raw="#2E8B3E",
             tpot="#C0392B", both="#7B4FA0", both_raw="#7B4FA0", bigfeat_auto="#B8860B", tpot_auto="#B8860B")
MARK = dict(original="o", split10="v", lgbm="P", bigfeat="s", bigfeat_raw="X", tpot="^", both="D", both_raw="*",
            bigfeat_auto="p", tpot_auto="h")
DATASETS = [("nbaiot", "infold20", "N-BaIoT"), ("edgeiiot", "all", "Edge-IIoTset"), ("unswnb15", "all", "UNSW-NB15")]


def load():
    runs = pd.concat([pd.read_csv(os.path.join(R2, "all_runs.csv")), pd.read_csv(os.path.join(R3, "all_runs.csv"))])
    runs["unit"] = runs.unit.astype(str)
    summ = pd.concat([pd.read_csv(os.path.join(R2, "summary_by_unit_config.csv")),
                      pd.read_csv(os.path.join(R3, "summary_by_unit_config.csv"))])
    par = pd.concat([pd.read_csv(os.path.join(R2, "pareto.csv")), pd.read_csv(os.path.join(R3, "pareto.csv"))])
    return runs, summ, par


def f1_deltas(runs, ds, fm):
    """Per configuration: mean change vs baseline, 95% CI half-width, significant after Holm."""
    sub = runs[(runs.dataset == ds) & (runs.feature_mode == fm)]
    out = {}
    if sub.unit.nunique() > 1:      # N-BaIoT: seed-averaged change per device, t-interval over devices
        dev = sub.groupby(["unit", "config"]).f1_macro.mean().unstack()
        pw = pd.read_csv(os.path.join(R2, "across_unit_pairwise.csv"))
        pw = pw[(pw.dataset == ds) & (pw.feature_mode == fm) & (pw.metric == "f1_macro")].set_index("config")
        for c in ORDER:
            if c in dev:
                d = (dev[c] - dev["original"]).values
                h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
                out[c] = (d.mean(), h, bool(pw.loc[c, "wilcoxon_p_holm"] < 0.05))
        return out
    cs = [c for c in ORDER if c in set(sub.config)]
    ps, rows = [], {}
    for c in cs:
        b = sub[sub.config == "original"].set_index("seed").f1_macro
        o = sub[sub.config == c].set_index("seed").f1_macro
        d = (o - b).dropna().values
        rows[c] = (d.mean(), A.ci95_corrected(d))
        ps.append(A.corrected_t(d)[1])
    for c, ph in zip(cs, A.holm(np.array(ps))):
        out[c] = (*rows[c], bool(ph < 0.05))
    return out


def fig_f1(runs):
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.9), sharey=True)
    y = np.arange(len(ORDER))[::-1]
    for ax, (ds, fm, title) in zip(axes, DATASETS):
        res = f1_deltas(runs, ds, fm)
        ax.axvspan(-A.ROPE_F1, A.ROPE_F1, color="#E3E3E3", zorder=0)
        ax.axvline(0, color="k", lw=0.6, zorder=1)
        for yi, c in zip(y, ORDER):
            if c not in res:
                ax.text(0, yi, "not run", ha="center", va="center", fontsize=6.5, color="#9A9A9A")
                continue
            m, h, sig = res[c]
            ax.errorbar(m, yi, xerr=h, fmt="none", ecolor=GROUP[c], elinewidth=1, capsize=2, zorder=2)
            ax.plot(m, yi, MARK[c], ms=5.5, color=GROUP[c], mfc=GROUP[c] if sig else "white", mew=1.1, zorder=3)
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.3)
        lim = max(0.008, max(abs(m) + h for m, h, _ in res.values()) * 1.08)
        ax.set_xlim(-lim, lim)
        ax.tick_params(axis="x", labelrotation=0)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3, symmetric=True))
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([NAME[c] for c in ORDER])
    axes[0].set_ylim(-0.6, len(ORDER) - 0.4)
    axes[1].set_xlabel("Change in Macro-F1 relative to the baseline")
    handles = [Line2D([], [], ls="none", marker="o", color="k", mfc="k", label="significant (Holm)"),
               Line2D([], [], ls="none", marker="o", color="k", mfc="white", label="not significant"),
               plt.Rectangle((0, 0), 1, 1, color="#E3E3E3", label="equivalent (±0.005)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.01),
               columnspacing=1.2, handletextpad=0.3)
    fig.tight_layout(rect=(0, 0.07, 1, 1), w_pad=0.6)
    fig.savefig(os.path.join(OUT, "f1_delta.pdf"))
    plt.close(fig)


# Label offsets (points) where points would otherwise collide.
OFFSET = {("edgeiiot", "original"): (5, 4), ("edgeiiot", "split10"): (5, -9), ("edgeiiot", "lgbm"): (5, 2),
          ("edgeiiot", "bigfeat"): (-30, 3), ("edgeiiot", "bigfeat_raw"): (-30, -10), ("edgeiiot", "tpot"): (-12, 6),
          ("edgeiiot", "both"): (7, 1), ("edgeiiot", "both_raw"): (7, -8),
          ("unswnb15", "original"): (5, 4), ("unswnb15", "split10"): (-6, 6), ("unswnb15", "lgbm"): (-14, -10),
          ("unswnb15", "bigfeat"): (5, 3), ("unswnb15", "bigfeat_raw"): (5, -8), ("unswnb15", "tpot"): (-12, 6),
          ("unswnb15", "both"): (5, 4), ("unswnb15", "both_raw"): (5, -8), ("unswnb15", "bigfeat_auto"): (5, -8),
          ("unswnb15", "tpot_auto"): (-30, 5)}


def fig_pareto(summ, par):
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.25), sharey=True)
    seen = []
    for ax, (ds, title) in zip(axes, [("edgeiiot", "Edge-IIoTset"), ("unswnb15", "UNSW-NB15")]):
        s = summ[summ.dataset == ds].set_index("config")
        p = par[par.dataset == ds].set_index("config")
        bx, by = s.loc["original", "train_time_s__mean"], s.loc["original", "seed_stability_tau_common"]
        for c in ["original"] + ORDER:
            if c not in s.index:
                continue
            x, yv = s.loc[c, "train_time_s__mean"], s.loc[c, "seed_stability_tau_common"]
            if c != "original":
                ax.plot([bx, x], [by, yv], color="#C8C8C8", lw=0.6, zorder=0)
            opt = bool(p.loc[c, "pareto_optimal"])
            ax.plot(x, yv, MARK[c], ms=6.5 if MARK[c] in "*Ph" else 5.5, color=GROUP[c],
                    mfc=GROUP[c] if opt else "white", mew=1.1, zorder=3)
            ax.annotate(f"{s.loc[c, 'f1_macro__mean']:.3f}", (x, yv), textcoords="offset points",
                        xytext=OFFSET.get((ds, c), (5, 3)), fontsize=7, color="#333333")
            if c not in seen:
                seen.append(c)
        ax.set_xscale("log")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.set_xlim(3, 12000)
    axes[0].set_ylabel("Cross-seed stability of\nthe SHAP ranking (Kendall τ)")
    fig.supxlabel("Training time, s (log scale)", fontsize=8.5, y=0.2)
    handles = [Line2D([], [], ls="none", marker=MARK[c], color=GROUP[c], ms=5.5, label=NAME[c]) for c in seen]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.01),
               columnspacing=1.0, handletextpad=0.3)
    fig.tight_layout(rect=(0, 0.2, 1, 1), w_pad=0.8)
    fig.savefig(os.path.join(OUT, "pareto.pdf"))
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    runs, summ, par = load()
    fig_f1(runs)
    fig_pareto(summ, par)
    if os.path.isdir(MANUSCRIPT_FIGS):
        for f in ("f1_delta.pdf", "pareto.pdf"):
            shutil.copy(os.path.join(OUT, f), MANUSCRIPT_FIGS)


if __name__ == "__main__":
    main()
