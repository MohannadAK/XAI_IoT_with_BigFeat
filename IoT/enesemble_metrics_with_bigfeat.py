#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Enhanced Ensemble Learning with TPOT and BigFeat Comparison

This script compares the performance of:
1. Traditional ensemble models with original features
2. TPOT AutoML with original features
3. Traditional ensemble models with BigFeat features
4. TPOT AutoML with BigFeat features
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier, StackingClassifier, VotingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
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

    # Verify it's the local version
    if bigfeat_local_path not in bigfeat_path:
        raise ImportError(f"Loaded BigFeat from {bigfeat_path}, but expected {bigfeat_local_path}")

    # Test BigFeat initialization with task_type
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

# List of CSV files
csv_files = [
    os.path.join(BASE_DIR, '..', 'Datasets', 'IOT Top 20 features datasets', f'device{i}_top_20_features.csv')
    for i in range(1, 10)
]

# Ensure output directory exists
output_dir = os.path.join(BASE_DIR, '..', 'Results', 'IoT', 'Enhanced_Ensemble_TPOT_Results')
try:
    os.makedirs(output_dir, exist_ok=True)
except OSError as e:
    print(f"ERROR creating output directory: {str(e)}")
    sys.exit(1)


def evaluate_ensemble_models_with_tpot(X_data, y_data, title, output_filename, device_name):
    """
    Evaluate ensemble models and TPOT AutoML, saving performance metrics.

    Args:
        X_data: Feature DataFrame
        y_data: Target labels
        title: Title for output visualizations
        output_filename: Filename for saving metrics visualization
        device_name: Name of the device for pipeline export

    Returns:
        DataFrame with performance metrics, TPOT classifier
    """
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

    # Initialize DataFrame to store metrics
    metrics_df = pd.DataFrame(columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'Training Time'])

    print(f"Evaluating traditional ensemble models for {title}...")

    # Traditional Ensemble Models
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
        )
    }

    # Evaluate traditional ensemble models
    for model_name, model in ensemble_models.items():
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        new_row = pd.DataFrame([{
            'Model': model_name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'Training Time': training_time
        }]).astype(metrics_df.dtypes)
        metrics_df = pd.concat([metrics_df, new_row], ignore_index=True)

    # Manual Blending Implementation
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

    accuracy = accuracy_score(y_test, blend_pred)
    precision = precision_score(y_test, blend_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, blend_pred, average='weighted')
    f1 = f1_score(y_test, blend_pred, average='weighted')

    new_row = pd.DataFrame([{
        'Model': 'Blending',
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'Training Time': blending_time
    }]).astype(metrics_df.dtypes)
    metrics_df = pd.concat([metrics_df, new_row], ignore_index=True)

    # TPOT AutoML Optimization
    print(f"Running TPOT AutoML optimization for {title}...")

    tpot_classifier = TPOTClassifier(
        generations=8,
        population_size=15,
        verbosity=2,
        random_state=42,
        n_jobs=-1,
        scoring='accuracy',
        cv=3,
        max_time_mins=8,
        max_eval_time_mins=1.5,
        early_stop=4
    )

    start_time = time.time()
    tpot_classifier.fit(X_train, y_train)
    tpot_training_time = time.time() - start_time

    y_pred_tpot = tpot_classifier.predict(X_test)

    accuracy_tpot = accuracy_score(y_test, y_pred_tpot)
    precision_tpot = precision_score(y_test, y_pred_tpot, average='weighted', zero_division=0)
    recall_tpot = recall_score(y_test, y_pred_tpot, average='weighted')
    f1_tpot = f1_score(y_test, y_pred_tpot, average='weighted')

    new_row = pd.DataFrame([{
        'Model': 'TPOT (AutoML)',
        'Accuracy': accuracy_tpot,
        'Precision': precision_tpot,
        'Recall': recall_tpot,
        'F1 Score': f1_tpot,
        'Training Time': tpot_training_time
    }]).astype(metrics_df.dtypes)
    metrics_df = pd.concat([metrics_df, new_row], ignore_index=True)

    pipeline_filename = os.path.join(output_dir,
                                     f"tpot_best_pipeline_{device_name}_{title.replace(' ', '_').lower()}.py")
    try:
        tpot_classifier.export(pipeline_filename)
        print(f"Best TPOT pipeline exported to: {pipeline_filename}")
    except Exception as e:
        print(f"Warning: Could not export TPOT pipeline: {str(e)}")

    print(f"\nPerformance Metrics for {title} (including TPOT):")
    print(metrics_df)

    plt.figure(figsize=(14, 10))
    plt.axis('off')

    cell_text = []
    for _, row in metrics_df.iterrows():
        formatted_row = []
        for col in metrics_df.columns:
            if col == 'Training Time':
                formatted_row.append(f"{row[col]:.2f}s" if isinstance(row[col], float) else row[col])
            elif isinstance(row[col], float):
                formatted_row.append(f"{row[col]:.4f}")
            else:
                formatted_row.append(str(row[col]))
        cell_text.append(formatted_row)

    table = plt.table(
        cellText=cell_text,
        colLabels=metrics_df.columns,
        cellLoc='center',
        loc='center',
        colColours=['#f2f2f2'] * len(metrics_df.columns)
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.3, 2.0)

    tpot_row_idx = len(metrics_df)
    for j in range(len(metrics_df.columns)):
        table[(tpot_row_idx, j)].set_facecolor('#e6f3ff')

    plt.title(f'Enhanced Ensemble Classification Performance: {title} (with TPOT AutoML)', fontsize=14)
    plt.tight_layout()

    try:
        plt.savefig(output_filename, format='jpg', dpi=300, bbox_inches='tight')
        plt.close()
    except OSError as e:
        print(f"ERROR saving visualization: {str(e)}")

    return metrics_df, tpot_classifier


def add_summary_row(df):
    """Add summary row with averages to comparison dataframe."""
    summary_row = {'Model': 'AVERAGE'}
    for col in df.columns:
        if col != 'Model' and df[col].dtype in ['float64', 'int64']:
            summary_row[col] = df[col].mean()
    summary_df = pd.DataFrame([summary_row])
    return pd.concat([df, summary_df], ignore_index=True)


def create_comprehensive_comparison(original_metrics, bigfeat_metrics, device_name):
    """Create comprehensive comparison between original and BigFeat features."""
    print(f"Creating comprehensive comparison for {device_name}...")

    comparison_df = pd.DataFrame()
    comparison_df['Model'] = original_metrics['Model']

    for metric in ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Training Time']:
        comparison_df[f'Original {metric}'] = original_metrics[metric]
        comparison_df[f'BigFeat {metric}'] = bigfeat_metrics[metric]
        if metric != 'Training Time':
            comparison_df[f'{metric} Diff'] = bigfeat_metrics[metric] - original_metrics[metric]
        else:
            comparison_df[f'{metric} Ratio'] = bigfeat_metrics[metric] / original_metrics[metric]

    comparison_df = add_summary_row(comparison_df)

    comparison_csv_file = os.path.join(output_dir, f"{device_name}_comprehensive_comparison.csv")
    try:
        comparison_df.to_csv(comparison_csv_file, index=False)
        print(f"Comprehensive comparison saved to {comparison_csv_file}")
    except OSError as e:
        print(f"ERROR saving comparison CSV: {str(e)}")

    plt.figure(figsize=(20, 12))
    plt.axis('off')

    cell_text = []
    for _, row in comparison_df.iterrows():
        row_values = []
        for col in comparison_df.columns:
            if isinstance(row[col], float):
                if 'Diff' in col:
                    value_str = f"{row[col]:+.4f}"
                elif 'Ratio' in col:
                    value_str = f"{row[col]:.2f}x"
                elif 'Time' in col:
                    value_str = f"{row[col]:.2f}s"
                else:
                    value_str = f"{row[col]:.4f}"
            else:
                value_str = str(row[col])
            row_values.append(value_str)
        cell_text.append(row_values)

    table = plt.table(
        cellText=cell_text,
        colLabels=comparison_df.columns,
        cellLoc='center',
        loc='center',
        colColours=['#f2f2f2'] * len(comparison_df.columns)
    )

    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.8, 2.2)

    for i in range(len(comparison_df)):
        model_name = comparison_df.iloc[i]['Model']
        for j, col in enumerate(comparison_df.columns):
            cell = table[(i + 1, j)]
            if 'TPOT' in str(model_name):
                cell.set_facecolor('#ffe6cc')
            if 'Diff' in col:
                val = comparison_df.iloc[i][col]
                if pd.notna(val):
                    if val > 0:
                        cell.set_facecolor('#d4f7d4')
                    elif val < 0:
                        cell.set_facecolor('#f7d4d4')
            if 'Ratio' in col:
                val = comparison_df.iloc[i][col]
                if pd.notna(val):
                    if val < 1.0:
                        cell.set_facecolor('#d4f7d4')
                    elif val > 2.0:
                        cell.set_facecolor('#f7d4d4')

    plt.title(f'{device_name} Comprehensive Comparison: Original vs BigFeat Features (with TPOT AutoML)', fontsize=16)
    plt.tight_layout()

    comparison_output_file = os.path.join(output_dir, f"{device_name}_comprehensive_comparison.jpg")
    try:
        plt.savefig(comparison_output_file, format='jpg', dpi=300, bbox_inches='tight')
        plt.close()
    except OSError as e:
        print(f"ERROR saving comparison visualization: {str(e)}")

    return comparison_df


def generate_feature_importance_analysis(X_bigfeat, y, device_name):
    """Generate feature importance analysis for BigFeat features."""
    try:
        print(f"Generating feature importance analysis for {device_name}...")

        rf = RandomForestClassifier(random_state=42, n_estimators=100)
        rf.fit(X_bigfeat, y)

        importances = rf.feature_importances_
        feature_names = X_bigfeat.columns
        indices = np.argsort(importances)[::-1]
        top_n = min(15, len(indices))

        plt.figure(figsize=(12, 8))
        plt.title(f'{device_name} BigFeat Feature Importances')
        bars = plt.bar(range(top_n), importances[indices[:top_n]], align='center')
        plt.xticks(range(top_n), [feature_names[i] for i in indices[:top_n]], rotation=45, ha='right')
        plt.ylabel('Feature Importance')
        plt.xlabel('Features')

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
        print(f"Top BigFeat features for {device_name} by importance:")
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
    print(f"PHASE 1: Evaluating Original Features with Ensemble + TPOT for {device_name}")
    print(f"{'=' * 60}")

    original_output_file = os.path.join(output_dir, f"{device_name}_original_with_tpot_metrics.jpg")
    original_metrics, tpot_original = evaluate_ensemble_models_with_tpot(
        X_scaled_df, y,
        f"{device_name} - Original Features",
        original_output_file,
        device_name
    )

    try:
        original_metrics.to_csv(
            os.path.join(output_dir, f"{device_name}_original_with_tpot_metrics.csv"), index=False)
        print(f"Original metrics with TPOT saved to {device_name}_original_with_tpot_metrics.csv")
    except OSError as e:
        print(f"ERROR saving original metrics CSV: {str(e)}")

    if not bigfeat_available:
        print(f"Skipping BigFeat evaluation for {device_name} as BigFeat is not available.")
        best_original_idx = original_metrics['Accuracy'].idxmax()
        overall_results.append({
            'Device': device_name,
            'Best_Original_Model': original_metrics.loc[best_original_idx, 'Model'],
            'Best_Original_Accuracy': original_metrics.loc[best_original_idx, 'Accuracy'],
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
        print(f"PHASE 3: Evaluating BigFeat Features with Ensemble + TPOT for {device_name}")
        print(f"{'=' * 60}")

        bigfeat_output_file = os.path.join(output_dir, f"{device_name}_bigfeat_with_tpot_metrics.jpg")
        bigfeat_metrics, tpot_bigfeat = evaluate_ensemble_models_with_tpot(
            X_bigfeat, y,
            f"{device_name} - With BigFeat Features",
            bigfeat_output_file,
            device_name
        )

        try:
            bigfeat_metrics.to_csv(
                os.path.join(output_dir, f"{device_name}_bigfeat_with_tpot_metrics.csv"),
                index=False)
            print(f"BigFeat metrics with TPOT saved to {device_name}_bigfeat_with_tpot_metrics.csv")
        except OSError as e:
            print(f"ERROR saving BigFeat metrics CSV: {str(e)}")

        print(f"\n{'=' * 60}")
        print(f"PHASE 4: Creating Comprehensive Comparison for {device_name}")
        print(f"{'=' * 60}")

        comparison_df = create_comprehensive_comparison(original_metrics, bigfeat_metrics, device_name)

        print(f"\n{'=' * 60}")
        print(f"PHASE 5: Feature Importance Analysis for {device_name}")
        print(f"{'=' * 60}")

        generate_feature_importance_analysis(X_bigfeat, y, device_name)

        best_original_idx = original_metrics['Accuracy'].idxmax()
        best_bigfeat_idx = bigfeat_metrics['Accuracy'].idxmax()

        overall_results.append({
            'Device': device_name,
            'Best_Original_Model': original_metrics.loc[best_original_idx, 'Model'],
            'Best_Original_Accuracy': original_metrics.loc[best_original_idx, 'Accuracy'],
            'Best_BigFeat_Model': bigfeat_metrics.loc[best_bigfeat_idx, 'Model'],
            'Best_BigFeat_Accuracy': bigfeat_metrics.loc[best_bigfeat_idx, 'Accuracy'],
            'BigFeat_Available': True,
            'Improvement': bigfeat_metrics.loc[best_bigfeat_idx, 'Accuracy'] - original_metrics.loc[
                best_original_idx, 'Accuracy']
        })

        print(f"\n{'=' * 60}")
        print(f"DEVICE SUMMARY for {device_name}")
        print(f"{'=' * 60}")
        print(f"Best Original Model: {original_metrics.loc[best_original_idx, 'Model']} "
              f"(Accuracy: {original_metrics.loc[best_original_idx, 'Accuracy']:.4f})")
        print(f"Best BigFeat Model: {bigfeat_metrics.loc[best_bigfeat_idx, 'Model']} "
              f"(Accuracy: {bigfeat_metrics.loc[best_bigfeat_idx, 'Accuracy']:.4f})")