#!/usr/bin/env python3
"""
Generate publication-quality figures for the IoT AutoML manuscript.

Figure A: automl_impact_overview.pdf   — 3-panel overview across all 9 devices
Figure B: attribution_drift_grid.pdf   — 2×2 slope charts (one per behavioral group)
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import matplotlib.ticker as ticker

# ── Paths ──────────────────────────────────────────────────────────────────────
SHAP_DIR = '/home/victoryst/personal/dev/XAI_IoT_with_BigFeat/Results/IoT/Feature_Importance/SHAP_Feature_Importance_Analysis'
OUT_DIR  = '/home/victoryst/personal/dev/XAI_IoT_with_BigFeat/manuscript/figures/Feature Importance'
os.makedirs(OUT_DIR, exist_ok=True)

# ── Palette ───────────────────────────────────────────────────────────────────
C = {
    'BigFeat': '#1565C0',   # deep blue
    'TPOT':    '#B71C1C',   # deep red
    'Both':    '#6A1B9A',   # deep purple
}
GRP_COLORS = {
    'GGG': '#2E7D32',   # dark green
    'GBG': '#1565C0',   # dark blue
    'BGB': '#E65100',   # deep orange
    'BBB': '#B71C1C',   # dark red
}
BIGFEAT_FEATURE_COLOR = '#F57C00'  # orange — for generated features in slope charts

# ── Constants ─────────────────────────────────────────────────────────────────
DEVICE_GROUP = {
    1: 'GBG', 2: 'GBG', 3: 'GBG',
    4: 'GGG', 5: 'GGG', 6: 'GGG', 8: 'GGG',
    7: 'BBB',
    9: 'BGB',
}

# Training times (seconds) from Table 4 of the manuscript
TRAIN_TIMES = {
    1: dict(Original=7,  BigFeat=12, TPOT=491, Both=571),
    2: dict(Original=4,  BigFeat=6,  TPOT=522, Both=518),
    3: dict(Original=4,  BigFeat=5,  TPOT=559, Both=572),
    4: dict(Original=7,  BigFeat=13, TPOT=576, Both=580),
    5: dict(Original=9,  BigFeat=16, TPOT=537, Both=592),
    6: dict(Original=10, BigFeat=18, TPOT=603, Both=601),
    7: dict(Original=4,  BigFeat=6,  TPOT=551, Both=587),
    8: dict(Original=9,  BigFeat=14, TPOT=513, Both=574),
    9: dict(Original=7,  BigFeat=12, TPOT=770, Both=591),
}

# Representative device per behavioral group (for Figure B)
GROUP_REPRESENTATIVE = {'GGG': 4, 'GBG': 1, 'BBB': 7, 'BGB': 9}

# ── Data loaders ──────────────────────────────────────────────────────────────
def load_shap(d):
    path = os.path.join(SHAP_DIR, f'Device{d}_shap_feature_importance_comparison.csv')
    return pd.read_csv(path)

def load_perf(d):
    path = os.path.join(SHAP_DIR, f'Device{d}_performance_comparison.csv')
    return pd.read_csv(path)

# ── Aggregate data ────────────────────────────────────────────────────────────
def get_f1_deltas():
    model_map = {
        'BigFeat': 'ExtraTrees BigFeat',
        'TPOT':    'TPOT Original',
        'Both':    'TPOT BigFeat',
    }
    result = {}
    for d in range(1, 10):
        df = load_perf(d).set_index('Model')
        base = df.loc['ExtraTrees Original', 'f1_macro']
        result[d] = {k: df.loc[v, 'f1_macro'] - base for k, v in model_map.items()}
    return result

def get_bigfeat_best_ranks():
    result = {}
    for d in range(1, 10):
        df = load_shap(d)
        bf = df[(df['Model'] == 'ExtraTrees BigFeat') & (df['Feature_Type'] == 'BigFeat_Generated')]
        result[d] = int(bf['Rank'].min()) if len(bf) > 0 else 16
    return result

def get_time_ratios():
    result = {}
    for d in range(1, 10):
        orig = TRAIN_TIMES[d]['Original']
        result[d] = {k: TRAIN_TIMES[d][k] / orig for k in ['BigFeat', 'TPOT', 'Both']}
    return result

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE A — AutoML Impact Overview
# ─────────────────────────────────────────────────────────────────────────────
def make_figure_A(f1_deltas, bf_ranks, time_ratios):
    DEVICE_LABELS = [f'D{i}' for i in range(1, 10)]
    n = 9
    configs = ['BigFeat', 'TPOT', 'Both']
    y_pos = np.arange(n)
    bw = 0.24   # bar width

    fig, axes = plt.subplots(
        1, 3, figsize=(13, 5.5),
        gridspec_kw={'width_ratios': [2, 1.2, 2.2], 'wspace': 0.55}
    )

    plt.rcParams.update({'font.size': 10})

    # ── Panel 1: F1 Delta (×10⁻³) ─────────────────────────────────────────────
    ax = axes[0]
    offsets = [-bw, 0, bw]
    for i, cfg in enumerate(configs):
        vals = [f1_deltas[d + 1][cfg] * 1000 for d in range(n)]
        ax.barh(y_pos + offsets[i], vals, height=bw * 0.88,
                color=C[cfg], alpha=0.88, label=cfg, zorder=3)

    ax.axvline(0, color='#333333', linewidth=1.0, zorder=5)
    lim = max(abs(f1_deltas[d][c] * 1000) for d in range(1, 10) for c in configs) * 1.35
    ax.set_xlim(-lim, lim)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(DEVICE_LABELS, fontsize=10, fontweight='bold')
    ax.invert_yaxis()
    ax.set_xlabel(r'$\Delta$F1 ($\times 10^{-3}$, vs. Original)', fontsize=10, labelpad=5)
    ax.set_title('(a) F1-Score Gain Over Baseline', fontsize=11, fontweight='bold', pad=8)
    ax.grid(axis='x', alpha=0.25, linewidth=0.6, zorder=0)
    ax.axvspan(0, lim,  alpha=0.04, color='green', zorder=0)
    ax.axvspan(-lim, 0, alpha=0.04, color='red',   zorder=0)

    for tick, d_idx in zip(ax.get_yticklabels(), range(1, 10)):
        tick.set_color(GRP_COLORS[DEVICE_GROUP[d_idx]])

    handles = [mpatches.Patch(facecolor=C[c], label=c) for c in configs]
    ax.legend(handles=handles, loc='lower right', fontsize=8.5, framealpha=0.85)
    _style_ax(ax)

    # ── Panel 2: BigFeat Feature Rank ─────────────────────────────────────────
    ax = axes[1]
    for i, d in enumerate(range(1, 10)):
        rank = bf_ranks[d]
        grp  = DEVICE_GROUP[d]
        color = GRP_COLORS[grp]
        ax.hlines(i, 1, rank, color=color, linewidth=2.2, alpha=0.65)
        ax.plot(rank, i, 'o', color=color, markersize=9, zorder=5)
        ax.text(rank + 0.15, i, str(rank), va='center', fontsize=8.5, color=color, fontweight='bold')

    ax.axvline(5.5, color='#9E9E9E', linewidth=0.8, linestyle='--', alpha=0.7)
    ax.text(5.6, -0.6, 'top 5', fontsize=8, color='#9E9E9E')
    ax.set_yticks(range(n))
    ax.set_yticklabels(DEVICE_LABELS, fontsize=10, fontweight='bold')
    ax.invert_yaxis()
    ax.set_xlim(0.5, 10)
    ax.set_xlabel('SHAP Rank of Best BigFeat Feature\n(lower = more prominent)', fontsize=10, labelpad=5)
    ax.set_title('(b) BigFeat Feature Prominence', fontsize=11, fontweight='bold', pad=8)
    ax.grid(axis='x', alpha=0.25, linewidth=0.6)

    for tick, d_idx in zip(ax.get_yticklabels(), range(1, 10)):
        tick.set_color(GRP_COLORS[DEVICE_GROUP[d_idx]])

    # Add group legend
    grp_handles = [mpatches.Patch(facecolor=GRP_COLORS[g], label=g) for g in ['GGG', 'GBG', 'BGB', 'BBB']]
    ax.legend(handles=grp_handles, loc='lower right', fontsize=8, title='Group', title_fontsize=8, framealpha=0.85)
    _style_ax(ax)

    # ── Panel 3: Training Time Ratio (log scale) ───────────────────────────────
    ax = axes[2]
    for i, cfg in enumerate(configs):
        vals = [time_ratios[d + 1][cfg] for d in range(n)]
        ax.barh(y_pos + offsets[i], vals, height=bw * 0.88,
                color=C[cfg], alpha=0.88, label=cfg, zorder=3)

    ax.set_xscale('log')
    ax.axvline(1, color='#333333', linewidth=1.0, zorder=5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(DEVICE_LABELS, fontsize=10, fontweight='bold')
    ax.invert_yaxis()
    ax.set_xlabel('Training Time Ratio vs. Original (log scale)', fontsize=10, labelpad=5)
    ax.set_title('(c) Computational Overhead', fontsize=11, fontweight='bold', pad=8)
    ax.grid(axis='x', alpha=0.25, linewidth=0.6, zorder=0)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:.0f}×'))

    for tick, d_idx in zip(ax.get_yticklabels(), range(1, 10)):
        tick.set_color(GRP_COLORS[DEVICE_GROUP[d_idx]])

    handles = [mpatches.Patch(facecolor=C[c], label=c) for c in configs]
    ax.legend(handles=handles, loc='lower right', fontsize=8.5, framealpha=0.85)
    _style_ax(ax)

    fig.suptitle(
        'AutoML Impact on IoT Anomaly Detection: Performance, Interpretability, and Cost',
        fontsize=12, fontweight='bold', y=1.01
    )

    out = os.path.join(OUT_DIR, 'automl_impact_overview.pdf')
    fig.savefig(out, dpi=300, bbox_inches='tight')
    fig.savefig(out.replace('.pdf', '.jpg'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out}')
    return out


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE B — Feature Attribution Drift (4 SEPARATE slope chart figures)
# ─────────────────────────────────────────────────────────────────────────────

def _shorten(name):
    """Shorten HH_ prefix and underscores for cleaner labels."""
    return name.replace('HH_', '').replace('BigFeat_', 'BigFeat ').replace('_', ' ')


def make_slope_figure(device_num, group_label, panel_letter, outpath_base):
    """
    One standalone slope chart figure for a single device/group.
    Designed to fill a full journal column (7 × 7 inches) with large fonts.
    """
    BASE_FONT = 13

    df = load_shap(device_num)
    orig_df = (df[df['Model'] == 'ExtraTrees Original']
               .sort_values('Rank').head(12).set_index('Feature'))
    bf_df   = (df[df['Model'] == 'ExtraTrees BigFeat']
               .sort_values('Rank').head(12).set_index('Feature'))

    # Union: top 10 from each side (preserving order, deduplicated)
    orig_feats = list(orig_df.index[:10])
    bf_feats   = list(bf_df.index[:10])
    all_feats  = list(dict.fromkeys(orig_feats + bf_feats))

    gc = GRP_COLORS[group_label]

    fig, ax = plt.subplots(figsize=(7, 7))

    for feat in all_feats:
        r_orig = orig_df.loc[feat, 'Rank'] if feat in orig_df.index else None
        r_bf   = bf_df.loc[feat, 'Rank']   if feat in bf_df.index   else None
        is_gen = 'BigFeat' in feat

        if is_gen:
            color, lw, alpha, zorder = BIGFEAT_FEATURE_COLOR, 2.8, 1.0, 5
        else:
            color, lw, alpha, zorder = '#607D8B', 1.6, 0.55, 2

        xs, ys = [], []
        if r_orig is not None: xs.append(0); ys.append(r_orig)
        if r_bf   is not None: xs.append(1); ys.append(r_bf)

        if len(xs) == 2:
            ax.plot(xs, ys, color=color, lw=lw, alpha=alpha, zorder=zorder,
                    solid_capstyle='round')
        elif len(xs) == 1:
            ax.plot(xs, ys, 'o', color=color, markersize=7, alpha=alpha, zorder=zorder)

        dot_alpha = min(1.0, alpha + 0.3)
        if r_orig is not None:
            ax.plot(0, r_orig, 'o', color=color, markersize=6,
                    alpha=dot_alpha, zorder=zorder + 1)
        if r_bf is not None:
            ax.plot(1, r_bf, 'o', color=color, markersize=6,
                    alpha=dot_alpha, zorder=zorder + 1)
            # Right-side label (BigFeat column)
            short  = _shorten(feat)
            weight = 'bold' if is_gen else 'normal'
            fsize  = BASE_FONT if is_gen else BASE_FONT - 1
            ax.text(1.05, r_bf, short, va='center', fontsize=fsize,
                    color=color, fontweight=weight, clip_on=False)

        # Left-side label (Original column) for original features only
        if r_orig is not None and not is_gen:
            short = _shorten(feat)
            ax.text(-0.05, r_orig, short, va='center', fontsize=BASE_FONT - 1,
                    color=color, ha='right', clip_on=False)

    # ── Axes styling ───────────────────────────────────────────────────────────
    all_ranks = (
        [orig_df.loc[f, 'Rank'] for f in all_feats if f in orig_df.index] +
        [bf_df.loc[f, 'Rank']   for f in all_feats if f in bf_df.index]
    )
    max_rank = min(max(all_ranks) if all_ranks else 12, 13)

    ax.set_ylim(max_rank + 0.7, 0.3)   # rank 1 at top
    ax.set_xlim(-0.55, 1.55)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Original\nConfig', 'BigFeat\nConfig'],
                       fontsize=BASE_FONT + 1, fontweight='bold')
    ax.set_ylabel('SHAP Feature Rank  (1 = most important)', fontsize=BASE_FONT)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.tick_params(axis='y', labelsize=BASE_FONT - 1)
    ax.grid(axis='y', alpha=0.2, linewidth=0.6)

    title = (f'Feature Attribution Drift — Device {device_num} '
             f'[{group_label} group]')
    ax.set_title(title, fontsize=BASE_FONT + 1, fontweight='bold', color=gc, pad=10)

    ax.text(0.02, 0.98, group_label, transform=ax.transAxes,
            fontsize=BASE_FONT, fontweight='bold', color='white',
            va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.3', fc=gc, ec='none', alpha=0.92))

    # Legend
    legend_handles = [
        Line2D([0], [0], color=BIGFEAT_FEATURE_COLOR, lw=2.5,
               label='BigFeat-generated feature'),
        Line2D([0], [0], color='#607D8B', lw=1.8, alpha=0.7,
               label='Original feature'),
    ]
    ax.legend(handles=legend_handles, loc='lower right', fontsize=BASE_FONT - 1,
              framealpha=0.9)

    _style_ax(ax)
    ax.spines['bottom'].set_visible(False)
    ax.spines['top'].set_visible(False)

    fig.tight_layout()

    out_pdf = f'{outpath_base}.pdf'
    out_jpg = f'{outpath_base}.jpg'
    fig.savefig(out_pdf, dpi=300, bbox_inches='tight')
    fig.savefig(out_jpg, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out_pdf}')
    return out_pdf


def make_figure_B():
    """Generate 4 separate slope chart figures, one per behavioral group."""
    groups       = ['GGG', 'GBG', 'BBB', 'BGB']
    letters      = ['a', 'b', 'c', 'd']
    saved_paths  = []

    for letter, grp in zip(letters, groups):
        dev      = GROUP_REPRESENTATIVE[grp]
        basename = os.path.join(OUT_DIR, f'attribution_drift_{grp}')
        path     = make_slope_figure(dev, grp, letter, basename)
        saved_paths.append(path)

    return saved_paths


# ── Helpers ───────────────────────────────────────────────────────────────────
def _style_ax(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#BDBDBD')
    ax.spines['bottom'].set_color('#BDBDBD')
    ax.tick_params(axis='both', which='both', length=3, color='#BDBDBD')


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    plt.rcParams.update({
        'font.family':       'DejaVu Sans',
        'axes.linewidth':    0.8,
        'xtick.labelsize':   9,
        'ytick.labelsize':   9,
        'figure.dpi':        150,
    })

    print('Loading data...')
    f1_deltas  = get_f1_deltas()
    bf_ranks   = get_bigfeat_best_ranks()
    time_ratios = get_time_ratios()

    print('\nGenerating Figure A — AutoML Impact Overview...')
    make_figure_A(f1_deltas, bf_ranks, time_ratios)

    print('\nGenerating Figure B — Feature Attribution Drift...')
    make_figure_B()

    print('\nDone. Both figures saved to:')
    print(f'  {OUT_DIR}')
