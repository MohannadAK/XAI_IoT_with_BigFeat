#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
log_bigfeat_formulas.py
-----------------------
Lightweight standalone script that runs only the BigFeat feature-generation step
(equivalent to Scenario 2) for all 9 N-BaIoT devices and writes every generated
feature's algebraic formula to disk.

Outputs (written to Results/IoT/Feature_Importance/BigFeat_Formulas/):
  - bigfeat_formulas.csv      : one row per feature per device (machine-readable)
  - bigfeat_formulas.txt      : human-readable summary of all formulas
  - DeviceX_bigfeat_formulas.csv : per-device detail file

Uses the exact same BigFeat parameters as shap_feature_importance_comparison.py:
  gen_size=10, iterations=10, estimator='rf', check_corr=True, selection='stability'
"""

import os
import sys
import time
import csv
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# ── BigFeat import (same path as main pipeline) ────────────────────────────────
BIGFEAT_PATH = '/mnt/Projects/BigFeat'
if 'bigfeat' in sys.modules:
    del sys.modules['bigfeat']
if BIGFEAT_PATH not in sys.path:
    sys.path.insert(0, BIGFEAT_PATH)

try:
    from bigfeat.bigfeat_base import BigFeat
    _ = BigFeat(task_type='classification')   # smoke-test
    print(f"BigFeat loaded from: {BIGFEAT_PATH}")
except Exception as e:
    print(f"ERROR: Could not load BigFeat — {e}")
    sys.exit(1)

# ── Paths (same relative layout as main pipeline) ──────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR  = os.path.join(BASE_DIR, '..', 'Datasets', 'IOT Top 20 features datasets')
OUTPUT_DIR   = os.path.join(BASE_DIR, '..', 'Results', 'IoT',
                            'Feature_Importance', 'BigFeat_Formulas')
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICES = list(range(1, 10))   # Device 1 – 9


# ── Formula reconstruction (identical logic to generate_bigfeat_features_with_mapping) ──
def reconstruct_formula(depth, op_ls, id_ls, feature_names):
    """Recursively reconstruct the algebraic formula from BigFeat's tracking lists."""
    if depth == 0:
        feat_ind = id_ls.pop()
        return feature_names[feat_ind]
    depth -= 1
    op = op_ls.pop()[0]
    op_name = getattr(op, '__name__', str(op))
    if op_name in ('multiply', 'add', 'subtract'):
        f1 = reconstruct_formula(depth, op_ls, id_ls, feature_names)
        f2 = reconstruct_formula(depth, op_ls, id_ls, feature_names)
        sym = {'multiply': '*', 'add': '+', 'subtract': '-'}[op_name]
        return f'({f2} {sym} {f1})'
    else:
        f1 = reconstruct_formula(depth, op_ls, id_ls, feature_names)
        if op_name == 'original_feat': return f1
        if op_name == 'square':        return f'({f1}^2)'
        if op_name in ('absolute', 'abs'): return f'abs({f1})'
        return f'{op_name}({f1})'


def get_formulas(bigfeat_model, original_feature_names):
    """Return a list of (gen_name, formula_str) for every generated feature."""
    results = []
    has_attrs = (hasattr(bigfeat_model, 'tracking_ids') and
                 hasattr(bigfeat_model, 'tracking_ops') and
                 hasattr(bigfeat_model, 'feat_depths'))
    if not has_attrs:
        print("  WARNING: BigFeat model is missing tracking attributes — formulas unavailable.")
        return results

    for i in range(len(bigfeat_model.tracking_ids)):
        name = f'BigFeat_Gen_{i}'
        try:
            op_ls  = bigfeat_model.tracking_ops[i].copy()
            id_ls  = bigfeat_model.tracking_ids[i].copy()
            depth  = bigfeat_model.feat_depths[i]
            formula = reconstruct_formula(depth, op_ls, id_ls, list(original_feature_names))
        except Exception as e:
            formula = f'[reconstruction error: {e}]'
        results.append((name, formula))
    return results


# ── Per-device processing ───────────────────────────────────────────────────────
all_rows = []   # accumulated across all devices for the master CSV

txt_log_path = os.path.join(OUTPUT_DIR, 'bigfeat_formulas.txt')
csv_log_path = os.path.join(OUTPUT_DIR, 'bigfeat_formulas.csv')

with open(txt_log_path, 'w') as txt_fh:
    txt_fh.write("BigFeat Generated Feature Formulas — N-BaIoT Dataset\n")
    txt_fh.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    txt_fh.write("Parameters: gen_size=10, iterations=10, estimator='rf', "
                 "check_corr=True, selection='stability', random_state=42\n")
    txt_fh.write("=" * 70 + "\n\n")

    for dev_id in DEVICES:
        device_name = f'Device{dev_id}'
        csv_path    = os.path.join(DATASET_DIR, f'device{dev_id}_top_20_features.csv')

        if not os.path.exists(csv_path):
            print(f"[{device_name}] Dataset not found — skipping: {csv_path}")
            txt_fh.write(f"[{device_name}] Dataset not found — skipped.\n\n")
            continue

        print(f"\n{'='*60}")
        print(f"[{device_name}] Loading data...")

        data = pd.read_csv(csv_path)
        X    = data.drop(columns=['label'])
        y    = data['label']
        original_feature_names = X.columns.tolist()

        scaler   = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, _, y_train, _ = train_test_split(
            X_scaled, y, test_size=0.3, random_state=42, stratify=y
        )

        print(f"[{device_name}] Fitting BigFeat on {X_train.shape[0]} training samples, "
              f"{X_train.shape[1]} features...")
        t0 = time.time()

        try:
            bigfeat = BigFeat(task_type='classification')
            bigfeat.fit(
                X_train, y_train.values,
                gen_size=10,
                random_state=42,
                iterations=10,
                estimator='rf',
                feat_imps=True,
                check_corr=True,
                selection='stability'
            )
        except Exception as e:
            print(f"[{device_name}] BigFeat.fit failed: {e}. Trying fallback config...")
            try:
                bigfeat = BigFeat(task_type='classification')
                bigfeat.fit(X_train, y_train.values, gen_size=3,
                            random_state=42, iterations=2, estimator=None)
            except Exception as e2:
                print(f"[{device_name}] Fallback also failed: {e2}. Skipping.")
                txt_fh.write(f"[{device_name}] BigFeat fitting failed: {e2}\n\n")
                continue

        elapsed = time.time() - t0

        # Count generated features
        n_generated = len(bigfeat.tracking_ids) if hasattr(bigfeat, 'tracking_ids') else '?'
        print(f"[{device_name}] Done in {elapsed:.1f}s — {n_generated} features generated.")

        formulas = get_formulas(bigfeat, original_feature_names)

        # ── Write to txt log ────────────────────────────────────────────────
        txt_fh.write(f"{'='*60}\n")
        txt_fh.write(f"{device_name}  ({len(formulas)} generated features, "
                     f"fit time: {elapsed:.1f}s)\n")
        txt_fh.write(f"{'='*60}\n")
        if formulas:
            for gen_name, formula in formulas:
                line = f"  {gen_name}: {formula}"
                txt_fh.write(line + "\n")
                print(line)
        else:
            txt_fh.write("  [No features generated]\n")
        txt_fh.write("\n")
        txt_fh.flush()   # flush after each device so partial results are safe

        # ── Per-device CSV ───────────────────────────────────────────────────
        device_rows = [
            {
                'device':       device_name,
                'feature_name': gen_name,
                'formula':      formula,
                'fit_time_s':   round(elapsed, 2),
            }
            for gen_name, formula in formulas
        ]
        if device_rows:
            dev_csv = os.path.join(OUTPUT_DIR, f'{device_name}_bigfeat_formulas.csv')
            pd.DataFrame(device_rows).to_csv(dev_csv, index=False)
            print(f"[{device_name}] Per-device CSV saved → {dev_csv}")

        all_rows.extend(device_rows)

# ── Master CSV ──────────────────────────────────────────────────────────────────
if all_rows:
    pd.DataFrame(all_rows).to_csv(csv_log_path, index=False)
    print(f"\nMaster CSV saved  → {csv_log_path}")
else:
    print("\nWARNING: No formulas were captured — master CSV not written.")

print(f"Text log saved    → {txt_log_path}")
print(f"\nOutput directory: {OUTPUT_DIR}")
print("Done.")
