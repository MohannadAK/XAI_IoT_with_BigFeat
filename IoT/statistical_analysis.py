"""
Statistical Analysis for Manuscript
====================================
Option A: Friedman test + Nemenyi post-hoc + Critical Distance diagram
         → operates on per-device RANKS of F1-score
Option B: Wilcoxon signed-rank pairwise tests
         → operates on raw F1-score values across devices

Both analyses are run on F1-score (primary metric) and training time (secondary).
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from pathlib import Path
from itertools import combinations

# ── paths ──────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "Results/IoT/Feature_Importance/SHAP_Feature_Importance_Analysis"
OUT_DIR     = ROOT / "manuscript/figures/Friedman Analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONFIGS = ['ExtraTrees Original', 'ExtraTrees BigFeat', 'TPOT Original', 'TPOT BigFeat']
LABELS  = ['Original', 'BigFeat', 'TPOT', 'Both']
COLORS  = ['#4C72B0', '#55A868', '#C44E52', '#8172B2']
N_DEVICES = 9

# ── training times (seconds) from Table 3 ──────────────────────────────────
TRAIN_TIMES = {
    'Original': [7, 4, 4, 7,  9, 10, 4, 9,  7],
    'BigFeat':  [12, 6, 5, 13, 16, 18, 6, 14, 12],
    'TPOT':     [491, 522, 559, 576, 537, 603, 551, 513, 770],
    'Both':     [571, 518, 572, 580, 592, 601, 587, 574, 591],
}

# ── load F1 data ───────────────────────────────────────────────────────────
f1_matrix = []
for i in range(1, N_DEVICES + 1):
    csv = RESULTS_DIR / f"Device{i}_performance_comparison.csv"
    df  = pd.read_csv(csv, index_col=0)
    f1_matrix.append([df.loc[c, 'f1_weighted'] for c in CONFIGS])

f1_arr   = np.array(f1_matrix)            # (9, 4)
time_arr = np.array([TRAIN_TIMES[l] for l in LABELS]).T   # (9, 4)

# ══════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════

def compute_ranks(data, higher_is_better=True):
    """Rank k methods for each dataset (1=best). Shape: (N, k)"""
    N, k = data.shape
    rank_matrix = np.zeros_like(data, dtype=float)
    for i in range(N):
        if higher_is_better:
            rank_matrix[i] = stats.rankdata(-data[i])   # negate: higher value → rank 1
        else:
            rank_matrix[i] = stats.rankdata(data[i])    # lower value → rank 1
    return rank_matrix


def nemenyi_cd(k, N, alpha=0.05):
    """Critical Distance for Nemenyi post-hoc test (Demšar 2006)."""
    # q_alpha critical values (k vs alpha=0.05)
    q_table = {2: 1.960, 3: 2.344, 4: 2.569, 5: 2.728,
               6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}
    q = q_table[k]
    return q * np.sqrt(k * (k + 1) / (6 * N))


def plot_cd_diagram(avg_ranks, labels, cd, title, filename, colors):
    """Standard Demšar CD diagram with cliques."""
    k = len(labels)
    sorted_idx = np.argsort(avg_ranks)
    sorted_ranks  = avg_ranks[sorted_idx]
    sorted_labels = [labels[i] for i in sorted_idx]
    sorted_colors = [colors[i] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.set_xlim(0.5, k + 0.5)
    ax.set_ylim(-1.2, 1.8)
    ax.axis('off')

    # Draw rank axis
    ax.annotate('', xy=(k + 0.3, 0.8), xytext=(0.7, 0.8),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax.text(0.5, 0.9, 'Rank', ha='left', va='bottom', fontsize=10, fontweight='bold')

    # Tick marks
    for rank in range(1, k + 1):
        ax.plot([rank, rank], [0.72, 0.88], color='black', lw=1)
        ax.text(rank, 0.65, str(rank), ha='center', va='top', fontsize=9)

    # Method markers & labels
    for pos, (rank, label, color) in enumerate(
            zip(sorted_ranks, sorted_labels, sorted_colors)):
        y = 1.25 if pos % 2 == 0 else 0.1
        ax.plot(rank, 0.8, 'o', color=color, markersize=10, zorder=5)
        ax.plot([rank, rank], [0.8, y + (0.05 if y > 0.8 else -0.05)],
                color=color, lw=1, ls='--', alpha=0.6)
        ax.text(rank, y + (0.12 if y > 0.8 else -0.12), label,
                ha='center', va='center', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', fc=color, alpha=0.25, ec='none'))

    # Draw CD bar (top-left)
    ax.annotate('', xy=(1.0 + cd, 1.65), xytext=(1.0, 1.65),
                arrowprops=dict(arrowstyle='<->', color='dimgray', lw=1.5))
    ax.text(1.0 + cd / 2, 1.72, f'CD = {cd:.2f}', ha='center',
            va='bottom', fontsize=9, color='dimgray', style='italic')

    # Find maximal cliques (groups not significantly different)
    cliques = []
    for i in range(k):
        for j in range(k - 1, i, -1):
            if sorted_ranks[j] - sorted_ranks[i] < cd:
                clique = set(range(i, j + 1))
                if not any(clique.issubset(c) for c in cliques):
                    cliques.append(clique)
                break

    bar_y = -0.15
    for clique in cliques:
        lo = sorted_ranks[min(clique)]
        hi = sorted_ranks[max(clique)]
        ax.plot([lo, hi], [bar_y, bar_y], color='black', lw=4, solid_capstyle='round')
        bar_y -= 0.25

    ax.set_title(title, fontsize=12, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(OUT_DIR / filename, bbox_inches='tight', dpi=200)
    plt.close()
    print(f"  Saved: {filename}")


def plot_wilcoxon_heatmap(data, labels, title, filename, alpha=0.05):
    """Heatmap of Wilcoxon signed-rank p-values with significance stars."""
    k = len(labels)
    p_matrix  = np.ones((k, k))
    w_matrix  = np.zeros((k, k))
    for i, j in combinations(range(k), 2):
        w, p = stats.wilcoxon(data[:, i], data[:, j], alternative='two-sided',
                              zero_method='wilcox')
        p_matrix[i, j] = p_matrix[j, i] = p
        w_matrix[i, j] = w_matrix[j, i] = w

    # Bonferroni correction
    n_comparisons = k * (k - 1) // 2
    alpha_corrected = alpha / n_comparisons

    fig, ax = plt.subplots(figsize=(6, 5))
    # Color by -log10(p)
    log_p = -np.log10(np.where(p_matrix == 1, np.nan, p_matrix))
    log_p_fill = np.where(p_matrix == 1, 0, log_p)

    cmap = plt.cm.YlOrRd
    im = ax.imshow(log_p_fill, cmap=cmap, vmin=0, vmax=3, aspect='auto')

    ax.set_xticks(range(k)); ax.set_yticks(range(k))
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)

    for i in range(k):
        for j in range(k):
            if i == j:
                ax.text(j, i, '—', ha='center', va='center', fontsize=13, color='gray')
            else:
                p = p_matrix[i, j]
                stars = ('***' if p < alpha_corrected else
                         '**'  if p < 0.01 else
                         '*'   if p < alpha else '')
                text  = f'p={p:.3f}\n{stars}' if stars else f'p={p:.3f}'
                color = 'white' if log_p_fill[i, j] > 1.5 else 'black'
                ax.text(j, i, text, ha='center', va='center',
                        fontsize=8.5, color=color)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('−log₁₀(p-value)', fontsize=9)
    cbar.set_ticks([0, 1, 2, 3])
    cbar.set_ticklabels(['1', '0.1', '0.01', '0.001'])

    sig_line = -np.log10(alpha)
    cbar.ax.axhline(sig_line, color='blue', lw=1.5, ls='--')
    cbar.ax.text(1.15, sig_line / 3, 'α=0.05', fontsize=7,
                 color='blue', transform=cbar.ax.transAxes)

    ax.set_title(title, fontsize=11, fontweight='bold', pad=10)

    note = (f'* p<0.05  ** p<0.01  *** p<{alpha_corrected:.3f} (Bonferroni, n={n_comparisons})\n'
            f'Wilcoxon signed-rank test, N={data.shape[0]} datasets')
    fig.text(0.5, -0.04, note, ha='center', fontsize=8, color='dimgray', style='italic')

    plt.tight_layout()
    plt.savefig(OUT_DIR / filename, bbox_inches='tight', dpi=200)
    plt.close()
    print(f"  Saved: {filename}")

    return p_matrix, alpha_corrected


# ══════════════════════════════════════════════════════════════════════════
# OPTION A: Friedman + Nemenyi CD diagram
# ══════════════════════════════════════════════════════════════════════════
print("\n══ OPTION A: Friedman + Nemenyi (Rankings) ══")

for metric_name, data in [('F1-Score', f1_arr), ('Training Time', time_arr)]:
    print(f"\n  [{metric_name}]")
    higher_better = (metric_name == 'F1-Score')
    rank_matrix = compute_ranks(data, higher_is_better=higher_better)
    avg_ranks   = rank_matrix.mean(axis=0)
    cd          = nemenyi_cd(k=4, N=N_DEVICES)

    stat, p = stats.friedmanchisquare(*[rank_matrix[:, j] for j in range(4)])
    print(f"  Average ranks: " +
          ", ".join(f"{l}={r:.3f}" for l, r in zip(LABELS, avg_ranks)))
    print(f"  Friedman χ² = {stat:.4f}, p = {p:.6f}  "
          f"({'significant ✅' if p < 0.05 else 'NOT significant ❌'})")
    print(f"  Nemenyi CD = {cd:.4f}")

    for i, j in combinations(range(4), 2):
        diff = abs(avg_ranks[i] - avg_ranks[j])
        sig  = "✅" if diff > cd else "—"
        print(f"    {LABELS[i]} vs {LABELS[j]}: Δrank={diff:.3f} {sig}")

    slug  = metric_name.lower().replace(' ', '_').replace('-', '_')
    title = (f"Critical Distance Diagram — {metric_name}\n"
             f"Friedman χ²={stat:.2f}, p={p:.4f}, CD={cd:.2f} (Nemenyi, α=0.05)")
    plot_cd_diagram(avg_ranks, LABELS, cd, title,
                    f"critical_distance_{slug}.pdf", COLORS)

# ══════════════════════════════════════════════════════════════════════════
# OPTION B: Wilcoxon signed-rank pairwise tests
# ══════════════════════════════════════════════════════════════════════════
print("\n══ OPTION B: Wilcoxon Signed-Rank Pairwise Tests ══")

for metric_name, data in [('F1-Score', f1_arr), ('Training Time', time_arr)]:
    print(f"\n  [{metric_name}]")
    slug  = metric_name.lower().replace(' ', '_').replace('-', '_')
    title = f"Wilcoxon Signed-Rank Pairwise p-values\n{metric_name} across {N_DEVICES} devices"
    p_mat, alpha_b = plot_wilcoxon_heatmap(
        data, LABELS, title, f"wilcoxon_{slug}.pdf")

    for i, j in combinations(range(4), 2):
        p   = p_mat[i, j]
        sig = ('*** (Bonferroni)' if p < alpha_b else
               '**'              if p < 0.01   else
               '*'               if p < 0.05   else '— ns')
        print(f"    {LABELS[i]} vs {LABELS[j]}: p={p:.4f}  {sig}")

# ══════════════════════════════════════════════════════════════════════════
# COMBINED SUMMARY FIGURE
# ══════════════════════════════════════════════════════════════════════════
print("\n  Generating summary comparison figure...")

rank_f1   = compute_ranks(f1_arr, higher_is_better=True)
avg_f1    = rank_f1.mean(axis=0)
rank_time = compute_ranks(time_arr, higher_is_better=False)
avg_time  = rank_time.mean(axis=0)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, avg_r, title in [(axes[0], avg_f1, 'Average F1-Score Rank'),
                          (axes[1], avg_time, 'Average Training-Time Rank')]:
    bars = ax.bar(LABELS, avg_r, color=COLORS, edgecolor='white', linewidth=0.8,
                  alpha=0.88, width=0.55)
    ax.set_ylim(0, 4.5)
    ax.set_ylabel('Average Rank (1 = best)', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.axhline(2.5, ls='--', color='gray', lw=0.8, alpha=0.5)
    for bar, val in zip(bars, avg_r):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.08,
                f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig.suptitle('Average Rankings Across 9 IoT Devices', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(OUT_DIR / "f1_score_rankings.pdf", bbox_inches='tight', dpi=200)
plt.savefig(OUT_DIR / "f1_score_rankings.png", bbox_inches='tight', dpi=150)
plt.close()
print("  Saved: f1_score_rankings.pdf")

print("\nAll figures saved to:", OUT_DIR)
