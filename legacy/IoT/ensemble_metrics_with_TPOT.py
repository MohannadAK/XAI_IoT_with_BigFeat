#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Enhanced Ensemble Learning with TPOT and BigFeat Comparison

This script compares the performance of:
1. Traditional ensemble models with original features
2. TPOT AutoML with original features (limited to ExtraTreesClassifier)
3. Traditional ensemble models with BigFeat features
4. TPOT AutoML with BigFeat features (limited to ExtraTreesClassifier)

Focus: F1 Score and Training Time for all models, with explicit comparison between TPOT (ExtraTrees)
and ExtraTreesClassifier to evaluate AutoML's impact on performance and explainability.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier, StackingClassifier, VotingClassifier, RandomForestClassifier, \
    ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.exceptions import ConvergenceWarning
import warnings
from tpot import TPOTClassifier

# Filter specific warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message="The default value of `dual` will change.*")

# Get base directory for robust path handling
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Enforce local BigFeat import
bigfeat_local_path = '/mnt/Projects/BigFeat'
bigfeat_available = False

# Remove any existing bigfeat module to avoid conflicts
if 'bigfeat' in sys.modules:
    del sys.modules['bigfeat']

# Add local BigFeat path to sys.path
if bigfeat_local_path not in sys.path:
    sys.path.insert(0, bigfeat_local_path)

# Import BigFeat
try:
    from bigfeat.bigfeat_base import BigFeat
    bigfeat_path = os.path.abspath(os.path.join(bigfeat_local_path, 'bigfeat', 'bigfeat_base.py'))
    print(f"Using BigFeat from: {bigfeat_path}")
    if bigfeat_local_path not in bigfeat_path:
        raise ImportError(f"Loaded BigFeat from {bigfeat_path}, but expected {bigfeat_local_path}")
    try:
        test_instance = BigFeat(task_type='classification')
        bigfeat_available = True
        print("Successfully initialized BigFeat with task_type parameter!")
    except TypeError as e:
        raise TypeError(f"Local BigFeat does not support task_type parameter: {str(e)}")
except ImportError as e:
    print(f"ERROR importing BigFeat: {str(e)}")
    print("WARNING: BigFeat not found. Will only evaluate original features.")
except Exception as e:
    print(f"Unexpected error loading BigFeat: {str(e)}")
    print("WARNING: BigFeat not found. Will only evaluate original features.")

# TPOT Configuration - Limited to ExtraTreesClassifier only
tpot_config = {
    'sklearn.ensemble.ExtraTreesClassifier': {
        'n_estimators': [10, 50, 100, 200],
        'criterion': ['gini', 'entropy'],
        'max_features': ['sqrt', 'log2', None],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 5, 10],
        'bootstrap': [True, False],
        'max_depth': [None, 5, 10, 15, 20]
    }
}

print("TPOT Configuration: Limited to ExtraTreesClassifier only")
print(f"Available hyperparameters: {list(tpot_config['sklearn.ensemble.ExtraTreesClassifier'].keys())}")

# List of CSV files
csv_files = [
    os.path.join(BASE_DIR, '..', 'Datasets', 'IOT Top 20 features datasets', f'device{i}_top_20_features.csv')
    for i in range(1, 10)
]

# Ensure output directory exists
output_dir = os.path.join(BASE_DIR, '..', 'Results', 'IoT', 'Enhanced_Ensemble_TPOT_ExtraTrees_Results')
try:
    os.makedirs(output_dir, exist_ok=True)
except OSError as e:
    print(f"ERROR creating output directory: {str(e)}")
    sys.exit(1)

def evaluate_ensemble_models_with_tpot(X_data, y_data, title, output_filename, device_name):
    """
    Evaluate ensemble models and TPOT AutoML (limited to ExtraTreesClassifier), focusing on F1 Score
    and Training Time. Highlight TPOT (ExtraTrees) vs. ExtraTrees comparison.
    """
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)
    metrics_df = pd.DataFrame(columns=['Model', 'F1 Score', 'Training Time'])

    print(f"Evaluating traditional ensemble models for {title}...")
    ensemble_models = {
        'Bagging': BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=100, random_state=42),
        'Stacking': StackingClassifier(
            estimators=[('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
                        ('dt', DecisionTreeClassifier(random_state=42))],
            final_estimator=LogisticRegression(max_iter=1000),
            stack_method='predict_proba'
        ),
        'Voting': VotingClassifier(
            estimators=[('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
                        ('dt', DecisionTreeClassifier(random_state=42))],
            voting='hard'
        ),
        'ExtraTrees': ExtraTreesClassifier(n_estimators=100, random_state=42)
    }

    for model_name, model in ensemble_models.items():
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred, average='weighted')
        new_row = pd.DataFrame([{
            'Model': model_name,
            'F1 Score': f1,
            'Training Time': training_time
        }]).astype(metrics_df.dtypes)
        metrics_df = pd.concat([metrics_df, new_row], ignore_index=True)

    print(f"Evaluating blending for {title}...")
    start_time = time.time()
    base_learners = [
        RandomForestClassifier(n_estimators=50, random_state=42),
        DecisionTreeClassifier(random_state=42)
    ]
    blend_pred = np.zeros((len(X_test), len(np.unique(y_data))))
    for base_learner in base_learners:
        base_learner.fit(X_train, y_train)
        blend_pred += base_learner.predict_proba(X_test)
    blend_pred = np.argmax(blend_pred, axis=1)
    if 0 not in np.unique(y_data):
        blend_pred += 1
    blending_time = time.time() - start_time
    f1 = f1_score(y_test, blend_pred, average='weighted')
    new_row = pd.DataFrame([{
        'Model': 'Blending',
        'F1 Score': f1,
        'Training Time': blending_time
    }]).astype(metrics_df.dtypes)
    metrics_df = pd.concat([metrics_df, new_row], ignore_index=True)

    print(f"Running TPOT AutoML optimization for {title} (ExtraTreesClassifier only)...")
    tpot_classifier = TPOTClassifier(
        config_dict=tpot_config,
        generations=8,
        population_size=15,
        verbosity=2,
        random_state=42,
        n_jobs=-1,
        scoring='f1_weighted',
        cv=3,
        max_time_mins=8,
        max_eval_time_mins=1.5,
        early_stop=4
    )
    start_time = time.time()
    tpot_classifier.fit(X_train, y_train)
    tpot_training_time = time.time() - start_time
    y_pred_tpot = tpot_classifier.predict(X_test)
    f1_tpot = f1_score(y_test, y_pred_tpot, average='weighted')
    new_row = pd.DataFrame([{
        'Model': 'TPOT (ExtraTrees)',
        'F1 Score': f1_tpot,
        'Training Time': tpot_training_time
    }]).astype(metrics_df.dtypes)
    metrics_df = pd.concat([metrics_df, new_row], ignore_index=True)

    pipeline_filename = os.path.join(output_dir,
                                    f"tpot_extratrees_pipeline_{device_name}_{title.replace(' ', '_').lower()}.py")
    try:
        tpot_classifier.export(pipeline_filename)
        print(f"Best TPOT ExtraTrees pipeline exported to: {pipeline_filename}")
    except Exception as e:
        print(f"Warning: Could not export TPOT pipeline: {str(e)}")

    # Highlight TPOT vs ExtraTrees comparison
    f1_comparison = metrics_df[metrics_df['Model'].isin(['ExtraTrees', 'TPOT (ExtraTrees)'])][['Model', 'F1 Score']]
    tpot_f1 = f1_comparison[f1_comparison['Model'] == 'TPOT (ExtraTrees)']['F1 Score'].iloc[0]
    extratrees_f1 = f1_comparison[f1_comparison['Model'] == 'ExtraTrees']['F1 Score'].iloc[0]
    f1_diff = tpot_f1 - extratrees_f1
    print(f"\nF1 Score Comparison for {title} (AutoML vs. Baseline):")
    print(f1_comparison)
    print(f"TPOT (ExtraTrees) F1 Score: {tpot_f1:.4f}")
    print(f"ExtraTrees F1 Score: {extratrees_f1:.4f}")
    print(f"Difference (TPOT - ExtraTrees): {f1_diff:+.4f}")
    print("Note: Positive difference indicates AutoML (TPOT) improves F1 score over the baseline ExtraTreesClassifier through hyperparameter optimization. However, TPOT's complex configurations may reduce explainability.")

    # Metrics visualization for all models
    plt.figure(figsize=(10, 8))
    plt.axis('off')
    cell_text = []
    for _, row in metrics_df.iterrows():
        formatted_row = [row['Model'], f"{row['F1 Score']:.4f}", f"{row['Training Time']:.2f}s"]
        cell_text.append(formatted_row)
    table = plt.table(
        cellText=cell_text,
        colLabels=['Model', 'F1 Score', 'Training Time'],
        cellLoc='center',
        loc='center',
        colColours=['#f2f2f2'] * 3
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.3, 2.0)
    tpot_row_idx = metrics_df[metrics_df['Model'] == 'TPOT (ExtraTrees)'].index[0] + 1
    extratrees_row_idx = metrics_df[metrics_df['Model'] == 'ExtraTrees'].index[0] + 1
    for j in range(3):
        table[(tpot_row_idx, j)].set_facecolor('#e6f3ff')  # Blue for TPOT
        table[(extratrees_row_idx, j)].set_facecolor('#f0fff0')  # Green for ExtraTrees
    plt.title(f'Performance Comparison: {title}\n(TPOT vs. ExtraTrees highlights AutoML impact)', fontsize=12)
    plt.tight_layout()
    try:
        plt.savefig(output_filename, format='jpg', dpi=300, bbox_inches='tight')
        plt.close()
    except OSError as e:
        print(f"ERROR saving visualization: {str(e)}")

    # Save metrics to CSV
    try:
        metrics_df.to_csv(os.path.join(output_dir, f"{device_name}_{title.replace(' ', '_').lower()}_metrics.csv"), index=False)
        print(f"Metrics saved to {device_name}_{title.replace(' ', '_').lower()}_metrics.csv")
    except OSError as e:
        print(f"ERROR saving metrics CSV: {str(e)}")

    return metrics_df, tpot_classifier

def create_comprehensive_comparison(original_metrics, bigfeat_metrics, device_name):
    """Create comprehensive comparison of F1 Score and Training Time, emphasizing TPOT vs. ExtraTrees."""
    print(f"Creating comprehensive comparison for {device_name}...")
    comparison_df = pd.DataFrame()
    comparison_df['Model'] = original_metrics['Model']
    comparison_df['Original F1 Score'] = original_metrics['F1 Score']
    comparison_df['BigFeat F1 Score'] = bigfeat_metrics['F1 Score']
    comparison_df['F1 Diff (BigFeat - Original)'] = bigfeat_metrics['F1 Score'] - original_metrics['F1 Score']
    comparison_df['Original Training Time'] = original_metrics['Training Time']
    comparison_df['BigFeat Training Time'] = bigfeat_metrics['Training Time']
    comparison_df['Training Time Ratio (BigFeat/Original)'] = bigfeat_metrics['Training Time'] / original_metrics['Training Time']
    comparison_df['F1 Diff (TPOT - ExtraTrees)'] = 0.0
    tpot_idx = comparison_df[comparison_df['Model'] == 'TPOT (ExtraTrees)'].index[0]
    extratrees_idx = comparison_df[comparison_df['Model'] == 'ExtraTrees'].index[0]
    # Use Original F1 Scores for TPOT vs. ExtraTrees difference as requested
    comparison_df.loc[tpot_idx, 'F1 Diff (TPOT - ExtraTrees)'] = (
        comparison_df.loc[tpot_idx, 'Original F1 Score'] - comparison_df.loc[extratrees_idx, 'Original F1 Score']
    )

    comparison_csv_file = os.path.join(output_dir, f"{device_name}_comprehensive_comparison.csv")
    try:
        comparison_df.to_csv(comparison_csv_file, index=False)
        print(f"Comprehensive comparison saved to {comparison_csv_file}")
    except OSError as e:
        print(f"ERROR saving comparison CSV: {str(e)}")

    plt.figure(figsize=(14, 8))
    plt.axis('off')
    cell_text = []
    for _, row in comparison_df.iterrows():
        row_values = [
            row['Model'],
            f"{row['Original F1 Score']:.4f}",
            f"{row['BigFeat F1 Score']:.4f}",
            f"{row['F1 Diff (BigFeat - Original)']:+.4f}",
            f"{row['Original Training Time']:.2f}s",
            f"{row['BigFeat Training Time']:.2f}s",
            f"{row['Training Time Ratio (BigFeat/Original)']:.2f}x",
            f"{row['F1 Diff (TPOT - ExtraTrees)']:+.4f}" if row['Model'] == 'TPOT (ExtraTrees)' else ""
        ]
        cell_text.append(row_values)
    table = plt.table(
        cellText=cell_text,
        colLabels=['Model', 'Original F1 Score', 'BigFeat F1 Score', 'F1 Diff (BigFeat - Original)',
                   'Original Training Time', 'BigFeat Training Time', 'Training Time Ratio (BigFeat/Original)',
                   'F1 Diff (TPOT - ExtraTrees)'],
        cellLoc='center',
        loc='center',
        colColours=['#f2f2f2'] * 8
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.5, 2.0)
    for i in range(len(comparison_df)):
        for j in range(8):
            if comparison_df.iloc[i]['Model'] == 'TPOT (ExtraTrees)':
                table[(i + 1, j)].set_facecolor('#e6f3ff')  # Blue for TPOT
            elif comparison_df.iloc[i]['Model'] == 'ExtraTrees':
                table[(i + 1, j)].set_facecolor('#f0fff0')  # Green for ExtraTrees
            if j in [3, 7] and pd.notna(comparison_df.iloc[i][comparison_df.columns[j]]):
                val = comparison_df.iloc[i][comparison_df.columns[j]]
                if val > 0:
                    table[(i + 1, j)].set_facecolor('#d4f7d4')  # Green for positive diff
                elif val < 0:
                    table[(i + 1, j)].set_facecolor('#f7d4d4')  # Red for negative diff
            if j == 6 and pd.notna(comparison_df.iloc[i][comparison_df.columns[j]]):
                val = comparison_df.iloc[i][comparison_df.columns[j]]
                if val < 1.0:
                    table[(i + 1, j)].set_facecolor('#d4f7d4')  # Green for faster
                elif val > 2.0:
                    table[(i + 1, j)].set_facecolor('#f7d4d4')  # Red for slower
    plt.title(f'{device_name} Comprehensive Comparison: Original vs BigFeat Features\n'
              f'Positive F1 Diff (BigFeat - Original) indicates feature engineering improvement\n'
              f'Positive F1 Diff (TPOT - ExtraTrees) indicates AutoML improvement over baseline', fontsize=12)
    plt.tight_layout()
    comparison_output_file = os.path.join(output_dir, f"{device_name}_comprehensive_comparison.jpg")
    try:
        plt.savefig(comparison_output_file, format='jpg', dpi=300, bbox_inches='tight')
        plt.close()
    except OSError as e:
        print(f"ERROR saving comparison visualization: {str(e)}")

    print(f"\nComprehensive Comparison for {device_name}:")
    print(comparison_df)
    print("Note: The 'F1 Diff (TPOT - ExtraTrees)' column highlights AutoML's performance improvement over the baseline ExtraTreesClassifier for Original features. Positive differences indicate TPOT's hyperparameter optimization enhances F1 scores, but complex configurations may reduce explainability.")

    return comparison_df

def add_summary_row(df):
    """Add summary row with averages to comparison dataframe."""
    summary_row = {'Model': 'AVERAGE'}
    for col in df.columns:
        if col != 'Model' and df[col].dtype in ['float64', 'int64']:
            summary_row[col] = df[col].mean()
    summary_df = pd.DataFrame([summary_row])
    return pd.concat([df, summary_df], ignore_index=True)

def generate_feature_importance_analysis(X_bigfeat, y, device_name):
    """Generate feature importance analysis for BigFeat features."""
    try:
        print(f"Generating feature importance analysis for {device_name}...")
        et = ExtraTreesClassifier(random_state=42, n_estimators=100)
        et.fit(X_bigfeat, y)
        importances = et.feature_importances_
        feature_names = X_bigfeat.columns
        indices = np.argsort(importances)[::-1]
        top_n = min(15, len(indices))
        plt.figure(figsize=(12, 8))
        plt.title(f'{device_name} BigFeat Feature Importances (ExtraTreesClassifier)')
        bars = plt.bar(range(top_n), importances[indices[:top_n]], align='center')
        plt.xticks(range(top_n), [feature_names[i] for i in indices[:top_n]], rotation=45, ha='right')
        plt.ylabel('Feature Importance')
        plt.xlabel('Features (Note: BigFeat features may be less interpretable due to automated generation)')
        for i, bar in enumerate(bars):
            if i < 5:
                bar.set_color('#2E8B57')
            else:
                bar.set_color('#87CEEB')
        plt.tight_layout()
        feature_importance_file = os.path.join(output_dir, f"{device_name}_bigfeat_feature_importance.jpg")
        plt.savefig(feature_importance_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Feature importance visualization saved to {feature_importance_file}")
        print(f"Top BigFeat features for {device_name} by importance (ExtraTreesClassifier):")
        for i in range(min(10, top_n)):
            print(f"  {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
        return True
    except Exception as e:
        print(f"Error generating feature importance for {device_name}: {str(e)}")
        return False

# Process each dataset
overall_results = []
for i, csv_file in enumerate(csv_files):
    print(f"\n{'=' * 100}")
    print(f"Processing device {i + 1} of {len(csv_files)}: {csv_file}")
    print(f"{'=' * 100}")
    device_name = f"device{i + 1}"
    try:
        print(f"Loading dataset {device_name}...")
        data = pd.read_csv(csv_file)
    except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError) as e:
        print(f"ERROR loading dataset: {str(e)}")
        continue
    try:
        X = data.drop(columns=['label'])
        y = data['label']
    except KeyError:
        print(f"ERROR: 'label' column not found in {csv_file}")
        continue
    print(f"Dataset shape: {X.shape}")
    print(f"Label distribution: {np.unique(y, return_counts=True)}")
    print(f"Scaling features for {device_name}...")
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    print(f"\n{'=' * 60}")
    print(f"PHASE 1: Evaluating Original Features with Ensemble + TPOT ExtraTrees for {device_name}")
    print(f"{'=' * 60}")
    original_output_file = os.path.join(output_dir, f"{device_name}_original_with_tpot_extratrees_metrics.jpg")
    original_metrics, tpot_original = evaluate_ensemble_models_with_tpot(
        X_scaled_df, y,
        f"{device_name} - Original Features",
        original_output_file,
        device_name
    )
    if not bigfeat_available:
        print(f"Skipping BigFeat evaluation for {device_name} as BigFeat is not available.")
        best_original_idx = original_metrics['F1 Score'].idxmax()
        overall_results.append({
            'Device': device_name,
            'Best_Original_Model': original_metrics.loc[best_original_idx, 'Model'],
            'Best_Original_F1': original_metrics.loc[best_original_idx, 'F1 Score'],
            'BigFeat_Available': False
        })
        continue
    print(f"\n{'=' * 60}")
    print(f"PHASE 2: Generating BigFeat Features for {device_name}")
    print(f"{'=' * 60}")
    try:
        print(f"Generating BigFeat features for {device_name}...")
        bigfeat = BigFeat(task_type='classification')
        X_bigfeat = bigfeat.fit(
            X, y,
            gen_size=5,
            random_state=42,
            iterations=3,
            estimator='rf',
            feat_imps=True,
            check_corr=True,
            selection='stability'
        )
    except ValueError as e:
        if "probabilities contain NaN" in str(e):
            print(f"Caught NaN probabilities error for {device_name}, using simpler approach...")
            bigfeat = BigFeat(task_type='classification')
            X_bigfeat = bigfeat.fit(
                X, y,
                gen_size=3,
                random_state=42,
                iterations=2,
                estimator='rf',
                feat_imps=False,
                check_corr=True,
                selection=None
            )
        elif "No axis named None" in str(e):
            print(f"Caught axis error for {device_name}, using modified approach...")
            bigfeat = BigFeat(task_type='classification')
            y_array = np.array(y)
            X_bigfeat = bigfeat.fit(
                X, y_array,
                gen_size=4,
                random_state=42,
                iterations=2,
                estimator='rf',
                feat_imps=False,
                check_corr=True,
                selection=None
            )
        else:
            print(f"ERROR with BigFeat for {device_name}: {str(e)}")
            continue
    print(f"Transforming data with BigFeat for {device_name}...")
    try:
        X_bigfeat = bigfeat.transform(X)
    except Exception as transform_error:
        print(f"Error transforming data with BigFeat for {device_name}: {str(transform_error)}")
        try:
            print("Attempting alternative transformation approach...")
            bigfeat_alt = BigFeat(task_type='classification')
            X_bigfeat = bigfeat_alt.fit(
                X.values, np.array(y),
                gen_size=2,
                random_state=42,
                iterations=1,
                estimator='dt',
                feat_imps=False,
                check_corr=False,
                selection=None
            )
            X_bigfeat = bigfeat_alt.transform(X.values)
        except Exception as alt_error:
            print(f"Alternative transformation also failed: {str(alt_error)}")
            X_bigfeat = None
    if X_bigfeat is not None:
        X_bigfeat = pd.DataFrame(X_bigfeat, columns=[f'feat_{i}' for i in range(X_bigfeat.shape[1])])
        print(f"BigFeat generated {X_bigfeat.shape[1]} features from {X.shape[1]} original features")
        print(f"\n{'=' * 60}")
        print(f"PHASE 3: Evaluating BigFeat Features with Ensemble + TPOT ExtraTrees for {device_name}")
        print(f"{'=' * 60}")
        bigfeat_output_file = os.path.join(output_dir, f"{device_name}_bigfeat_with_tpot_extratrees_metrics.jpg")
        bigfeat_metrics, tpot_bigfeat = evaluate_ensemble_models_with_tpot(
            X_bigfeat, y,
            f"{device_name} - With BigFeat Features",
            bigfeat_output_file,
            device_name
        )
        print(f"\n{'=' * 60}")
        print(f"PHASE 4: Creating Comprehensive Comparison for {device_name}")
        print(f"{'=' * 60}")
        comparison_df = create_comprehensive_comparison(original_metrics, bigfeat_metrics, device_name)
        print(f"\n{'=' * 60}")
        print(f"PHASE 5: Feature Importance Analysis for {device_name}")
        print(f"{'=' * 60}")
        generate_feature_importance_analysis(X_bigfeat, y, device_name)
        best_original_idx = original_metrics['F1 Score'].idxmax()
        best_bigfeat_idx = bigfeat_metrics['F1 Score'].idxmax()
        overall_results.append({
            'Device': device_name,
            'Best_Original_Model': original_metrics.loc[best_original_idx, 'Model'],
            'Best_Original_F1': original_metrics.loc[best_original_idx, 'F1 Score'],
            'Best_BigFeat_Model': bigfeat_metrics.loc[best_bigfeat_idx, 'Model'],
            'Best_BigFeat_F1': bigfeat_metrics.loc[best_bigfeat_idx, 'F1 Score'],
            'BigFeat_Available': True,
            'F1_Improvement': bigfeat_metrics.loc[best_bigfeat_idx, 'F1 Score'] - original_metrics.loc[best_original_idx, 'F1 Score']
        })
    else:
        print(f"Skipping BigFeat evaluation for {device_name} due to transformation failure.")
        best_original_idx = original_metrics['F1 Score'].idxmax()
        overall_results.append({
            'Device': device_name,
            'Best_Original_Model': original_metrics.loc[best_original_idx, 'Model'],
            'Best_Original_F1': original_metrics.loc[best_original_idx, 'F1 Score'],
            'BigFeat_Available': False
        })

print(f"\n{'=' * 100}")
print("FINAL SUMMARY")
print(f"{'=' * 100}")
if overall_results:
    overall_df = pd.DataFrame(overall_results)
    print("\nOverall F1 Score Summary:")
    print(overall_df.to_string(index=False))
    overall_summary_file = os.path.join(output_dir, "overall_summary_extratrees.csv")
    try:
        overall_df.to_csv(overall_summary_file, index=False)
        print(f"Overall summary saved to {overall_summary_file}")
    except OSError as e:
        print(f"ERROR saving overall summary: {str(e)}")
    if 'BigFeat_Available' in overall_df.columns:
        available_count = overall_df['BigFeat_Available'].sum()
        total_count = len(overall_df)
        print(f"\nBigFeat Processing Statistics:")
        print(f"Successfully processed: {available_count}/{total_count} devices")
        if available_count > 0:
            bigfeat_results = overall_df[overall_df['BigFeat_Available'] == True]
            avg_original_f1 = bigfeat_results['Best_Original_F1'].mean()
            avg_bigfeat_f1 = bigfeat_results['Best_BigFeat_F1'].mean()
            avg_improvement = bigfeat_results['F1_Improvement'].mean()
            print(f"Average Original F1 Score: {avg_original_f1:.4f}")
            print(f"Average BigFeat F1 Score: {avg_bigfeat_f1:.4f}")
            print(f"Average F1 Improvement: {avg_improvement:+.4f}")
            positive_improvements = (bigfeat_results['F1_Improvement'] > 0).sum()
            print(f"Devices with positive F1 improvement: {positive_improvements}/{available_count}")
            print(f"\nBest Original Models Distribution:")
            original_model_counts = bigfeat_results['Best_Original_Model'].value_counts()
            for model, count in original_model_counts.items():
                print(f"  {model}: {count}")
            print(f"\nBest BigFeat Models Distribution:")
            bigfeat_model_counts = bigfeat_results['Best_BigFeat_Model'].value_counts()
            for model, count in bigfeat_model_counts.items():
                print(f"  {model}: {count}")
else:
    print("No results to summarize.")
print(f"\n{'=' * 100}")
print("ANALYSIS COMPLETE")
print(f"All results saved to: {output_dir}")
print("Key changes made:")
print("1. Focused metrics on F1 Score and Training Time only")
print("2. Emphasized TPOT (ExtraTrees) vs. ExtraTrees comparison in outputs and visualizations")
print("3. Retained comprehensive comparison with all models, highlighting AutoML's impact")
print("4. Added notes on explainability implications of AutoML and BigFeat")
print("5. Fixed syntax error in base_learners list by removing erroneous 'meas' token")
print("6. Updated 'F1 Diff (TPOT - ExtraTrees)' to use Original F1 Scores instead of BigFeat F1 Scores")
print(f"{'=' * 100}")