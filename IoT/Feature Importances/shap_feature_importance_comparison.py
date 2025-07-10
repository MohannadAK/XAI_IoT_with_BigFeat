#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SHAP Feature Importance Comparison for ExtraTrees and TPOT with/without BigFeat

This script compares SHAP feature importance across four scenarios:
1. ExtraTrees with original features
2. ExtraTrees with BigFeat features (original feature names preserved)
3. TPOT (ExtraTrees) with original features
4. TPOT (ExtraTrees) with BigFeat features (original feature names preserved)

Focus: SHAP values analysis, feature importance comparison, and visualization
Outputs are saved immediately after each scenario.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from tpot import TPOTClassifier
import shap

# Filter warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message="The default value of `dual` will change.*")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

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
    bigfeat_available = False
except Exception as e:
    print(f"Unexpected error loading BigFeat: {str(e)}")
    print("WARNING: BigFeat not found. Will only evaluate original features.")
    bigfeat_available = False

# TPOT Configuration - Limited to ExtraTreesClassifier only
tpot_config = {
    'sklearn.ensemble.ExtraTreesClassifier': {
        'n_estimators': [10, 50, 100],
        'criterion': ['gini', 'entropy'],
        'max_features': ['sqrt', 'log2'],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 5, 10],
        'bootstrap': [True, False],
        'max_depth': [None, 5, 10, 15, 20]
    }
}

print("TPOT Configuration: Limited to ExtraTreesClassifier only")

DEVICE_START_INDEX = 1

csv_files = [
    os.path.join(BASE_DIR, '..', '..', 'Datasets', 'IOT Top 20 features datasets', f'device{i}_top_20_features.csv')
    for i in range(DEVICE_START_INDEX, 10)
]
# Ensure output directory exists
output_dir = os.path.join(BASE_DIR, '..', '..', 'Results', 'IoT', 'Feature_Importance',
                          'SHAP_Feature_Importance_Analysis')
try:
    os.makedirs(output_dir, exist_ok=True)
except OSError as e:
    print(f"ERROR creating directory: {str(e)}")
    sys.exit(1)


def generate_bigfeat_features_with_mapping(X, y, original_feature_names):
    """
    Generate BigFeat features while preserving original feature name mapping
    Returns:
    --------
    X_bigfeat : array-like
        BigFeat-transformed features
    feature_names : list
        Feature names with original names preserved
    original_indices : list
        Indices of original features in the BigFeat output
    bigfeat_model : BigFeat
        Fitted BigFeat model for later transformations
    """
    if not bigfeat_available:
        print("BigFeat not available, skipping feature generation")
        return None, None, None, None

    print("Generating BigFeat features...")
    # Ensure inputs are NumPy arrays
    X_np = np.array(X) if isinstance(X, (pd.DataFrame, pd.Series)) else X
    y_np = np.array(y) if isinstance(y, (pd.DataFrame, pd.Series)) else y
    print(f"X_np shape: {X_np.shape}, y_np type: {type(y_np)}")
    try:
        bigfeat = BigFeat(task_type='classification')
        X_bigfeat = bigfeat.fit(
            X_np, y_np,
            gen_size=5,
            random_state=42,
            iterations=3,
            estimator='rf',
            feat_imps=True,
            check_corr=True,
            selection='stability'
        )
    except Exception as e:
        print(f"Caught error during BigFeat fitting: {str(e)}")
        print("Falling back to simpler BigFeat configuration...")
        try:
            bigfeat = BigFeat(task_type='classification')
            X_bigfeat = bigfeat.fit(
                X_np, y_np,
                gen_size=3,
                random_state=42,
                iterations=2,
                estimator=None
            )
        except Exception as e:
            print(f"BigFeat generation failed: {str(e)}")
            return None, None, None, None

    # Transform the data
    try:
        X_bigfeat_transformed = bigfeat.transform(X_np)
    except Exception as e:
        print(f"Error transforming data with BigFeat: {str(e)}")
        return None, None, None, None

    # Create feature names
    n_generated = len(bigfeat.tracking_ids) if hasattr(bigfeat, 'tracking_ids') else X_bigfeat_transformed.shape[
                                                                                         1] - len(
        original_feature_names)
    n_original = len(original_feature_names)
    generated_names = [f'BigFeat_Gen_{i}' for i in range(n_generated)]
    preserved_names = list(original_feature_names)
    feature_names = generated_names + preserved_names
    original_indices = list(range(n_generated, n_generated + n_original))

    print(f"BigFeat generated {n_generated} new features")
    print(f"Original features preserved: {n_original}")
    print(f"Total features: {X_bigfeat_transformed.shape[1]}")

    return X_bigfeat_transformed, feature_names, original_indices, bigfeat


def train_extratrees_model(X, y, random_state=42):
    """Train ExtraTreesClassifier model"""
    print("Training ExtraTreesClassifier...")
    model = ExtraTreesClassifier(
        n_estimators=100,
        random_state=random_state,
        n_jobs=-1,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1
    )
    model.fit(X, y)
    print("ExtraTreesClassifier training completed!")
    return model


def train_tpot_model(X, y, random_state=42):
    """
    Train TPOT model with improved error handling
    """
    print("Training TPOT model (ExtraTreesClassifier only)...")

    # Reduced configuration for more stable training
    tpot_config = {
        'sklearn.ensemble.ExtraTreesClassifier': {
            'n_estimators': [50, 100],
            'criterion': ['gini', 'entropy'],
            'max_features': ['sqrt', 'log2'],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 5],
            'bootstrap': [True, False],
            'max_depth': [None, 10, 20]
        }
    }

    try:
        tpot_classifier = TPOTClassifier(
            config_dict=tpot_config,
            generations=3,  # Reduced for faster training
            population_size=8,  # Reduced for faster training
            verbosity=1,
            random_state=random_state,
            n_jobs=1,  # Use single job to avoid issues
            scoring='f1_weighted',
            cv=3,
            max_time_mins=3,  # Reduced time limit
            max_eval_time_mins=1,
            early_stop=2
        )
        tpot_classifier.fit(X, y)
        print("TPOT model training completed!")
        return tpot_classifier
    except Exception as e:
        print(f"TPOT training failed: {str(e)}")


def compute_shap_values(model, X, feature_names, model_name, max_samples=500):
    """
    Compute SHAP values for the given model with improved error handling for multi-class and TPOT models
    """
    print(f"Computing SHAP values for {model_name}...")

    # Sample data if too large
    if X.shape[0] > max_samples:
        sample_indices = np.random.choice(X.shape[0], max_samples, replace=False)
        X_sample = X[sample_indices] if hasattr(X, '__getitem__') else X[sample_indices, :]
    else:
        X_sample = X

    try:
        # Handle TPOT models - extract the actual classifier
        actual_model = model
        if hasattr(model, 'fitted_pipeline_'):
            pipeline = model.fitted_pipeline_
            if isinstance(pipeline, Pipeline):
                # Find the ExtraTreesClassifier in the pipeline
                for step_name, step_model in pipeline.steps:
                    if isinstance(step_model, ExtraTreesClassifier):
                        actual_model = step_model
                        print(f"Extracted ExtraTreesClassifier from TPOT pipeline")
                        break
                else:
                    # If no ExtraTreesClassifier found, use the whole pipeline
                    print(f"Using full TPOT pipeline for SHAP")
                    actual_model = pipeline
            else:
                # Direct classifier (not a pipeline) - handle nested classifiers
                if isinstance(pipeline, ExtraTreesClassifier):
                    actual_model = pipeline
                    print(f"Using direct TPOT classifier for SHAP")
                else:
                    # Fallback for complex nested structures
                    actual_model = model
                    print(f"Using full TPOT model for SHAP")

        # Create SHAP explainer with improved error handling
        if isinstance(actual_model, ExtraTreesClassifier):
            try:
                # First attempt with tree_path_dependent (more stable than interventional)
                explainer = shap.TreeExplainer(actual_model, feature_perturbation='tree_path_dependent', check_additivity=False)
                shap_values = explainer.shap_values(X_sample)
            except Exception as tree_error:
                print(f"TreeExplainer with tree_path_dependent failed: {str(tree_error)}")
                try:
                    # Second attempt with no feature perturbation and additivity check disabled
                    explainer = shap.TreeExplainer(actual_model, check_additivity=False)
                    shap_values = explainer.shap_values(X_sample)
                except Exception as tree_error2:
                    print(f"TreeExplainer with default settings failed: {str(tree_error2)}")
                    # Fallback to KernelExplainer
                    raise Exception("TreeExplainer failed, falling back to KernelExplainer")
        else:
            # Use KernelExplainer for complex pipelines
            raise Exception("Using KernelExplainer for non-tree model")

        print(f"SHAP values computed successfully for {model_name}")
        print(f"SHAP values type: {type(shap_values)}")

        if isinstance(shap_values, list):
            print(f"SHAP values classes: {len(shap_values)}")
            print(f"First class SHAP shape: {shap_values[0].shape}")
        else:
            print(f"SHAP values shape: {shap_values.shape}")

        return shap_values, explainer, X_sample

    except Exception as e:
        print(f"Error computing SHAP values for {model_name}: {str(e)}")
        print("Falling back to KernelExplainer...")
        try:
            # Reduce sample size for KernelExplainer
            kernel_sample_size = min(50, X_sample.shape[0])  # Reduced from 100
            background_size = min(20, kernel_sample_size // 2)  # Reduced from 50

            # Create background sample
            background_indices = np.random.choice(X_sample.shape[0], background_size, replace=False)
            background = X_sample[background_indices]

            # Create explainer with the original model (not extracted)
            explainer = shap.KernelExplainer(model.predict_proba, background)

            # Use smaller sample for explanation
            explain_indices = np.random.choice(X_sample.shape[0], kernel_sample_size, replace=False)
            X_explain = X_sample[explain_indices]

            # Compute SHAP values with reduced samples
            shap_values = explainer.shap_values(X_explain, nsamples=50)

            print(f"SHAP values computed using KernelExplainer for {model_name}")
            print(f"Explanation sample size: {X_explain.shape[0]}")

            return shap_values, explainer, X_explain

        except Exception as e2:
            print(f"KernelExplainer also failed: {str(e2)}")
            try:
                # Final fallback with minimal configuration
                print("Attempting minimal KernelExplainer configuration...")

                # Use very small samples
                min_background_size = min(10, X_sample.shape[0] // 4)  # Reduced from 20
                min_explain_size = min(25, X_sample.shape[0] // 2)     # Reduced from 50

                background_indices = np.random.choice(X_sample.shape[0], min_background_size, replace=False)
                background = X_sample[background_indices]

                explain_indices = np.random.choice(X_sample.shape[0], min_explain_size, replace=False)
                X_explain = X_sample[explain_indices]

                # Create simple prediction function with better error handling
                def predict_fn(x):
                    try:
                        proba = model.predict_proba(x)
                        # Ensure proper shape
                        if proba.ndim == 1:
                            proba = proba.reshape(-1, 1)
                        return proba
                    except Exception as pred_error:
                        print(f"predict_proba failed: {pred_error}, using predict")
                        # If predict_proba fails, use predict and convert to probabilities
                        pred = model.predict(x)
                        unique_classes = np.unique(pred)
                        n_classes = len(unique_classes)
                        proba = np.zeros((len(pred), n_classes))
                        for i, p in enumerate(pred):
                            class_idx = np.where(unique_classes == p)[0][0]
                            proba[i, class_idx] = 1.0
                        return proba

                explainer = shap.KernelExplainer(predict_fn, background)
                shap_values = explainer.shap_values(X_explain, nsamples=25)  # Reduced from 50

                print(f"SHAP values computed using minimal KernelExplainer for {model_name}")
                return shap_values, explainer, X_explain

            except Exception as e3:
                print(f"All SHAP computation methods failed for {model_name}: {str(e3)}")
                return None, None, None


def process_shap_values_for_visualization(shap_values, feature_names):
    """
    Process SHAP values for visualization, handling multi-class cases properly
    """
    if shap_values is None or feature_names is None:
        return None, None

    try:
        # Handle multi-class SHAP values
        if isinstance(shap_values, list):
            # For multi-class, aggregate across classes using mean absolute values
            shap_values_agg = np.mean([np.abs(v) for v in shap_values], axis=0)
        else:
            shap_values_agg = np.abs(shap_values)

        # Ensure we have 2D array (samples x features)
        if shap_values_agg.ndim > 2:
            shap_values_agg = np.mean(shap_values_agg, axis=tuple(range(shap_values_agg.ndim))[2:])

        # Calculate mean importance across samples
        mean_shap = np.mean(shap_values_agg, axis=0)

        # Handle dimension mismatch
        n_features = min(len(mean_shap), len(feature_names))
        mean_shap = mean_shap[:n_features]
        feature_names_truncated = feature_names[:n_features]

        if len(mean_shap) != len(feature_names_truncated):
            print(
                f"Warning: Adjusted SHAP values ({len(mean_shap)}) to match feature names ({len(feature_names_truncated)})")

        return mean_shap, feature_names_truncated

    except Exception as e:
        print(f"Error processing SHAP values: {str(e)}")
        return None, None


def safe_array_conversion(value):
    """Safely convert array values to scalars for plotting"""
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return float(value.item())
        else:
            return float(value.flatten()[0])
    elif hasattr(value, '__len__') and len(value) == 1:
        return float(value[0])
    else:
        return float(value)


def save_scenario_shap_visualization(shap_values, feature_names, model_name, original_indices,
                                    X_sample, device_name, output_dir, scenario_num, feature_color_map):
    """
    Save SHAP visualization for a single scenario with improved error handling
    Outputs: JPG, PDF, and CSV files for feature importance
    """
    if shap_values is None or feature_names is None:
        print(f"Skipping visualization for {model_name} (Scenario {scenario_num}) due to missing data")
        return

    try:
        # Process SHAP values
        mean_shap, processed_feature_names = process_shap_values_for_visualization(shap_values, feature_names)

        if mean_shap is None or processed_feature_names is None:
            print(f"Failed to process SHAP values for {model_name}")
            return

        # Limit to top features for better visualization
        n_features_to_show = min(15, len(mean_shap))

        # Sort by importance
        sorted_indices = np.argsort(mean_shap)[::-1][:n_features_to_show]
        sorted_features = [processed_feature_names[idx] for idx in sorted_indices]
        sorted_values = [float(mean_shap[idx]) for idx in sorted_indices]

        # Assign colors
        sorted_colors = [feature_color_map.get(feature, '#87CEEB') for feature in sorted_features]

        # Create DataFrame for CSV
        results_df = pd.DataFrame({
            'Feature_Name': sorted_features,
            'SHAP_Value': sorted_values,
            'Feature_Type': ['Original' if 'BigFeat' not in feature else 'BigFeat_Generated' for feature in sorted_features],
            'Rank': range(1, len(sorted_features) + 1)
        })

        # Save CSV
        csv_file = os.path.join(output_dir,
                                f"{device_name}_{model_name.replace(' ', '_').replace('(', '').replace(')', '')}_scenario{scenario_num}_shap_results.csv")
        results_df.to_csv(csv_file, index=False)
        print(f"SHAP results CSV for {model_name} (Scenario {scenario_num}) saved to {csv_file}")

        # Create plot
        plt.figure(figsize=(12, 10))
        y_pos = np.arange(len(sorted_features))
        bars = plt.barh(y_pos, sorted_values, color=sorted_colors)
        plt.yticks(y_pos, sorted_features, fontsize=10)
        plt.xlabel('Mean |SHAP Value|', fontsize=12)
        plt.title(
            f'{device_name} - {model_name}\nTop {len(sorted_features)} Features by SHAP Importance (Scenario {scenario_num})',
            fontsize=14, pad=20)
        plt.gca().invert_yaxis()

        # Add value labels
        if sorted_values:
            max_value = max(sorted_values)
            for i, v in enumerate(sorted_values):
                plt.text(v + max_value * 0.01, i, f'{v:.4f}', va='center', ha='left', fontsize=9)

        # Add legend if both types of features exist
        if any(color == '#2E8B57' for color in sorted_colors) and any(color == '#87CEEB' for color in sorted_colors):
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#2E8B57', label='Original Features'),
                Patch(facecolor='#87CEEB', label='BigFeat Generated Features')
            ]
            plt.legend(handles=legend_elements, loc='lower right', fontsize=10)

        plt.tight_layout()

        # Save JPG
        jpg_file = os.path.join(output_dir,
                                f"{device_name}_{model_name.replace(' ', '_').replace('(', '').replace(')', '')}_scenario{scenario_num}_shap_importance.jpg")
        plt.savefig(jpg_file, dpi=300, bbox_inches='tight')
        print(f"SHAP importance JPG for {model_name} (Scenario {scenario_num}) saved to {jpg_file}")

        # Save PDF
        from matplotlib.backends.backend_pdf import PdfPages
        pdf_file = os.path.join(output_dir,
                                f"{device_name}_{model_name.replace(' ', '_').replace('(', '').replace(')', '')}_scenario{scenario_num}_shap_importance.pdf")
        with PdfPages(pdf_file) as pdf:
            pdf.savefig(plt.gcf(), bbox_inches='tight')
        print(f"SHAP importance PDF for {model_name} (Scenario {scenario_num}) saved to {pdf_file}")

        plt.close()

    except Exception as e:
        print(f"Error creating SHAP visualization for {model_name}: {str(e)}")

def create_shap_comparison_visualization(shap_data_list, feature_names_list, model_names,
                                        original_indices_list, X_samples_list, device_name, output_dir,
                                        feature_color_map):
    """
    Create comprehensive SHAP comparison visualization with improved error handling
    Outputs: JPG, PDF, and CSV files for feature importance comparison
    """
    try:
        fig, axes = plt.subplots(2, 2, figsize=(24, 20))
        axes = axes.ravel()
        comparison_data = []

        for idx, (shap_values, feature_names, model_name, original_indices) in enumerate(
                zip(shap_data_list, feature_names_list, model_names, original_indices_list)):

            ax = axes[idx]
            if shap_values is None or feature_names is None:
                ax.text(0.5, 0.5, f'SHAP computation failed\nfor {model_name}',
                        ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.set_title(f'{model_name}\n(Failed)', fontsize=12)
                continue

            try:
                # Process SHAP values
                mean_shap, processed_feature_names = process_shap_values_for_visualization(shap_values, feature_names)

                if mean_shap is None or processed_feature_names is None:
                    ax.text(0.5, 0.5, f'SHAP processing failed\nfor {model_name}',
                            ha='center', va='center', transform=ax.transAxes, fontsize=12)
                    ax.set_title(f'{model_name}\n(Processing Failed)', fontsize=12)
                    continue

                # Limit features for visualization
                n_features_to_show = min(15, len(mean_shap))
                sorted_indices = np.argsort(mean_shap)[::-1][:n_features_to_show]
                sorted_features = [processed_feature_names[i] for i in sorted_indices]
                sorted_values = [float(mean_shap[i]) for i in sorted_indices]

                # Store data for CSV
                for feature, value in zip(sorted_features, sorted_values):
                    comparison_data.append({
                        'Model': model_name,
                        'Feature': feature,
                        'SHAP_Value': value,
                        'Feature_Type': 'Original' if 'BigFeat' not in feature else 'BigFeat_Generated',
                        'Rank': sorted_features.index(feature) + 1
                    })

                # Assign colors
                sorted_colors = [feature_color_map.get(feature, '#87CEEB') for feature in sorted_features]

                # Plot
                y_pos = np.arange(len(sorted_features))
                bars = ax.barh(y_pos, sorted_values, color=sorted_colors)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(sorted_features, fontsize=10)
                ax.set_xlabel('Mean |SHAP Value|', fontsize=11)
                ax.set_title(f'{model_name}\nTop {len(sorted_features)} Features', fontsize=12)
                ax.invert_yaxis()

                # Add value labels
                if sorted_values:
                    max_value = max(sorted_values)
                    for i, v in enumerate(sorted_values):
                        ax.text(v + max_value * 0.01, i, f'{v:.4f}',
                                va='center', ha='left', fontsize=8)

            except Exception as e:
                print(f"Error plotting {model_name}: {str(e)}")
                ax.text(0.5, 0.5, f'Plotting error\nfor {model_name}',
                        ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.set_title(f'{model_name}\n(Error)', fontsize=12)

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#2E8B57', label='Original Features'),
            Patch(facecolor='#87CEEB', label='BigFeat Generated Features')
        ]
        fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.02),
                   ncol=2, fontsize=12)

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.05)

        # Save JPG
        jpg_file = os.path.join(output_dir, f"{device_name}_shap_feature_importance_comparison.jpg")
        plt.savefig(jpg_file, dpi=300, bbox_inches='tight')
        print(f"SHAP feature importance comparison JPG saved to {jpg_file}")

        # Save PDF
        from matplotlib.backends.backend_pdf import PdfPages
        pdf_file = os.path.join(output_dir, f"{device_name}_shap_feature_importance_comparison.pdf")
        with PdfPages(pdf_file) as pdf:
            pdf.savefig(plt.gcf(), bbox_inches='tight')
        print(f"SHAP feature importance comparison PDF saved to {pdf_file}")

        plt.close()

        # Save CSV
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            csv_file = os.path.join(output_dir, f"{device_name}_shap_feature_importance_comparison.csv")
            comparison_df.to_csv(csv_file, index=False)
            print(f"SHAP feature importance comparison CSV saved to {csv_file}")

    except Exception as e:
        print(f"Error creating comparison visualization: {str(e)}")


def create_original_features_comparison(shap_data_list, feature_names_list, model_names,
                                        original_indices_list, device_name, output_dir):
    """
    Create comparison focusing on original features with improved error handling
    """
    try:
        # Extract original feature names
        original_feature_names = None
        for feature_names, original_indices in zip(feature_names_list, original_indices_list):
            if original_indices is not None and feature_names is not None:
                original_feature_names = [feature_names[idx] for idx in original_indices if idx < len(feature_names)]
                break
            elif original_indices is None and feature_names is not None:
                original_feature_names = feature_names[:20]  # Assume first 20 are original
                break

        if original_feature_names is None:
            print("Cannot create original features comparison - no original features found")
            return

        original_shap_comparison = {}
        for shap_values, feature_names, model_name, original_indices in zip(
                shap_data_list, feature_names_list, model_names, original_indices_list):

            if shap_values is not None and feature_names is not None:
                try:
                    # Process SHAP values
                    mean_shap, processed_feature_names = process_shap_values_for_visualization(shap_values,
                                                                                               feature_names)

                    if mean_shap is None:
                        continue

                    # Extract original feature SHAP values
                    if original_indices is not None:
                        original_shap = [mean_shap[idx] for idx in original_indices if idx < len(mean_shap)]
                    else:
                        # Assume first features are original
                        n_original = len(original_feature_names)
                        original_shap = mean_shap[:n_original]

                    # Ensure same length as original feature names
                    min_len = min(len(original_shap), len(original_feature_names))
                    original_shap = original_shap[:min_len]
                    original_shap_comparison[model_name] = original_shap

                except Exception as e:
                    print(f"Error processing SHAP data for {model_name}: {str(e)}")

        if not original_shap_comparison:
            print("No valid SHAP values for original features comparison")
            return

        # Create DataFrame with consistent indexing
        min_features = min(len(v) for v in original_shap_comparison.values())
        original_feature_names_truncated = original_feature_names[:min_features]

        for model_name in original_shap_comparison:
            original_shap_comparison[model_name] = original_shap_comparison[model_name][:min_features]

        comparison_df = pd.DataFrame(original_shap_comparison, index=original_feature_names_truncated)

        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

        # Heatmap
        sns.heatmap(comparison_df.T, annot=True, fmt='.4f', cmap='viridis', ax=ax1)
        ax1.set_title(f'{device_name} - Original Features SHAP Importance Heatmap')
        ax1.set_xlabel('Original Features')
        ax1.set_ylabel('Models')
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')

        # Top features bar plot
        avg_importance = comparison_df.mean(axis=1).sort_values(ascending=False)
        top_10_features = avg_importance.head(10)
        comparison_df.loc[top_10_features.index].plot(kind='bar', ax=ax2, width=0.8)
        ax2.set_title(f'{device_name} - Top 10 Original Features Across Models')
        ax2.set_xlabel('Original Features')
        ax2.set_ylabel('Mean |SHAP Value|')
        ax2.legend(title='Models', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        original_comparison_file = os.path.join(output_dir, f"{device_name}_original_features_shap_comparison.jpg")
        plt.savefig(original_comparison_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Original features SHAP comparison saved to {original_comparison_file}")

        # Save CSV
        comparison_csv = os.path.join(output_dir, f"{device_name}_original_features_shap_values.csv")
        comparison_df.to_csv(comparison_csv)
        print(f"Original features SHAP values saved to {comparison_csv}")

        return comparison_df

    except Exception as e:
        print(f"Error creating original features comparison: {str(e)}")
        return None


def save_scenario_performance(performance, model_name, device_name, output_dir, scenario_num):
    """
    Save performance metrics for a single scenario to CSV
    """
    try:
        performance_df = pd.DataFrame({
            'accuracy': [performance['accuracy']],
            'f1_score': [performance['f1_score']]
        }, index=[model_name])
        performance_file = os.path.join(output_dir,
                                        f"{device_name}_{model_name.replace(' ', '_').replace('(', '').replace(')', '')}_scenario{scenario_num}_performance.csv")
        performance_df.to_csv(performance_file)
        print(f"Performance metrics for {model_name} (Scenario {scenario_num}) saved to {performance_file}")
    except Exception as e:
        print(f"Error saving performance for {model_name} (Scenario {scenario_num}): {str(e)}")


def evaluate_model_performance(model, X_test, y_test, model_name):
    """Evaluate model performance"""
    try:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        print(f"{model_name} Performance:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1 Score (weighted): {f1:.4f}")
        return {'accuracy': accuracy, 'f1_score': f1}
    except Exception as e:
        print(f"Error evaluating {model_name}: {str(e)}")
        return {'accuracy': 0, 'f1_score': 0}


def analyze_device_shap(csv_file, device_name):
    """
    Analyze SHAP feature importance for a single device across all four scenarios
    Saves outputs immediately after each scenario with improved error handling
    """
    print(f"\n{'=' * 80}")
    print(f"Analyzing SHAP Feature Importance for {device_name}")
    print(f"{'=' * 80}")

    # Initialize global feature color mapping
    feature_color_map = {}
    data = pd.read_csv(csv_file)
    original_feature_names = data.drop(columns=['label']).columns.tolist()
    for feature in original_feature_names:
        feature_color_map[feature] = '#2E8B57'  # Green for original features
    # Placeholder for generated features (will be updated if BigFeat is used)

    try:
        print(f"Loading dataset {device_name}...")
        data = pd.read_csv(csv_file)
        X = data.drop(columns=['label'])
        y = data['label']
        print(f"Dataset shape: {X.shape}")
        print(f"Original features: {len(original_feature_names)}")
        print(f"Class distribution: {y.value_counts().to_dict()}")

        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=original_feature_names)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled_df, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")

        # Initialize storage for all scenarios
        all_shap_data = []
        all_feature_names = []
        all_model_names = []
        all_original_indices = []
        all_X_samples = []
        all_performance = []

        # ========== SCENARIO 1: ExtraTrees with Original Features ==========
        print(f"\n{'=' * 60}")
        print("SCENARIO 1: ExtraTrees with Original Features")
        print(f"{'=' * 60}")

        try:
            model1 = train_extratrees_model(X_train, y_train)
            performance1 = evaluate_model_performance(model1, X_test, y_test, "ExtraTrees Original")
            save_scenario_performance(performance1, "ExtraTrees_Original", device_name, output_dir, 1)

            shap_values1, explainer1, X_sample1 = compute_shap_values(
                model1, X_test.values, original_feature_names, "ExtraTrees Original"
            )
            save_scenario_shap_visualization(
                shap_values1, original_feature_names, "ExtraTrees Original",
                None, X_sample1, device_name, output_dir, 1, feature_color_map
            )

            all_shap_data.append(shap_values1)
            all_feature_names.append(original_feature_names)
            all_model_names.append("ExtraTrees Original")
            all_original_indices.append(None)
            all_X_samples.append(X_sample1)
            all_performance.append(performance1)

        except Exception as e:
            print(f"Error in Scenario 1: {str(e)}")
            all_shap_data.append(None)
            all_feature_names.append(None)
            all_model_names.append("ExtraTrees Original")
            all_original_indices.append(None)
            all_X_samples.append(None)
            all_performance.append({'accuracy': 0, 'f1_score': 0})

        # ========== SCENARIO 2: ExtraTrees with BigFeat Features ==========
        print(f"\n{'=' * 60}")
        print("SCENARIO 2: ExtraTrees with BigFeat Features")
        print(f"{'=' * 60}")

        if bigfeat_available:
            try:
                X_bigfeat, feature_names_bigfeat, original_indices, bigfeat_model = generate_bigfeat_features_with_mapping(
                    X_train, y_train, original_feature_names
                )

                if X_bigfeat is not None:
                    # Transform test set
                    X_test_bigfeat = bigfeat_model.transform(X_test.values)

                    model2 = train_extratrees_model(X_bigfeat, y_train)
                    performance2 = evaluate_model_performance(model2, X_test_bigfeat, y_test, "ExtraTrees BigFeat")
                    save_scenario_performance(performance2, "ExtraTrees_BigFeat", device_name, output_dir, 2)

                    shap_values2, explainer2, X_sample2 = compute_shap_values(
                        model2, X_test_bigfeat, feature_names_bigfeat, "ExtraTrees BigFeat"
                    )
                    # Update feature_color_map with generated features
                    for feature in feature_names_bigfeat:
                        if feature not in feature_color_map and not any(f in feature for f in original_feature_names):
                            feature_color_map[feature] = '#87CEEB'  # Blue for generated features
                    save_scenario_shap_visualization(
                        shap_values2, feature_names_bigfeat, "ExtraTrees BigFeat",
                        original_indices, X_sample2, device_name, output_dir, 2, feature_color_map
                    )

                    all_shap_data.append(shap_values2)
                    all_feature_names.append(feature_names_bigfeat)
                    all_model_names.append("ExtraTrees BigFeat")
                    all_original_indices.append(original_indices)
                    all_X_samples.append(X_sample2)
                    all_performance.append(performance2)
                else:
                    print("BigFeat feature generation failed")
                    all_shap_data.append(None)
                    all_feature_names.append(None)
                    all_model_names.append("ExtraTrees BigFeat")
                    all_original_indices.append(None)
                    all_X_samples.append(None)
                    all_performance.append({'accuracy': 0, 'f1_score': 0})

            except Exception as e:
                print(f"Error in Scenario 2: {str(e)}")
                all_shap_data.append(None)
                all_feature_names.append(None)
                all_model_names.append("ExtraTrees BigFeat")
                all_original_indices.append(None)
                all_X_samples.append(None)
                all_performance.append({'accuracy': 0, 'f1_score': 0})
        else:
            print("BigFeat not available, skipping Scenario 2")
            all_shap_data.append(None)
            all_feature_names.append(None)
            all_model_names.append("ExtraTrees BigFeat")
            all_original_indices.append(None)
            all_X_samples.append(None)
            all_performance.append({'accuracy': 0, 'f1_score': 0})

        # ========== SCENARIO 3: TPOT with Original Features ==========
        print(f"\n{'=' * 60}")
        print("SCENARIO 3: TPOT with Original Features")
        print(f"{'=' * 60}")

        try:
            model3 = train_tpot_model(X_train, y_train)
            performance3 = evaluate_model_performance(model3, X_test, y_test, "TPOT Original")
            save_scenario_performance(performance3, "TPOT_Original", device_name, output_dir, 3)

            # For TPOT, use the original test data directly (no transformation needed)
            # since TPOT's fitted_pipeline_ is the actual classifier, not a preprocessing pipeline
            shap_values3, explainer3, X_sample3 = compute_shap_values(
                model3, X_test.values, original_feature_names, "TPOT Original", max_samples=200
            )
            save_scenario_shap_visualization(
                shap_values3, original_feature_names, "TPOT Original",
                None, X_sample3, device_name, output_dir, 3, feature_color_map
            )

            all_shap_data.append(shap_values3)
            all_feature_names.append(original_feature_names)
            all_model_names.append("TPOT Original")
            all_original_indices.append(None)
            all_X_samples.append(X_sample3)
            all_performance.append(performance3)

        except Exception as e:
            print(f"Error in Scenario 3: {str(e)}")
            all_shap_data.append(None)
            all_feature_names.append(None)
            all_model_names.append("TPOT Original")
            all_original_indices.append(None)
            all_X_samples.append(None)
            all_performance.append({'accuracy': 0, 'f1_score': 0})

        # ========== SCENARIO 4: TPOT with BigFeat Features ==========
        print(f"\n{'=' * 60}")
        print("SCENARIO 4: TPOT with BigFeat Features")
        print(f"{'=' * 60}")

        if bigfeat_available and 'X_bigfeat' in locals() and X_bigfeat is not None:
            try:
                model4 = train_tpot_model(X_bigfeat, y_train)
                performance4 = evaluate_model_performance(model4, X_test_bigfeat, y_test, "TPOT BigFeat")
                save_scenario_performance(performance4, "TPOT_BigFeat", device_name, output_dir, 4)

                shap_values4, explainer4, X_sample4 = compute_shap_values(
                    model4, X_test_bigfeat, feature_names_bigfeat, "TPOT BigFeat"
                )
                # Update feature_color_map with generated features if not already set
                for feature in feature_names_bigfeat:
                    if feature not in feature_color_map and not any(f in feature for f in original_feature_names):
                        feature_color_map[feature] = '#87CEEB'  # Blue for generated features
                save_scenario_shap_visualization(
                    shap_values4, feature_names_bigfeat, "TPOT BigFeat",
                    original_indices, X_sample4, device_name, output_dir, 4, feature_color_map
                )

                all_shap_data.append(shap_values4)
                all_feature_names.append(feature_names_bigfeat)
                all_model_names.append("TPOT BigFeat")
                all_original_indices.append(original_indices)
                all_X_samples.append(X_sample4)
                all_performance.append(performance4)

            except Exception as e:
                print(f"Error in Scenario 4: {str(e)}")
                all_shap_data.append(None)
                all_feature_names.append(None)
                all_model_names.append("TPOT BigFeat")
                all_original_indices.append(None)
                all_X_samples.append(None)
                all_performance.append({'accuracy': 0, 'f1_score': 0})
        else:
            print("BigFeat not available or failed, skipping Scenario 4")
            all_shap_data.append(None)
            all_feature_names.append(None)
            all_model_names.append("TPOT BigFeat")
            all_original_indices.append(None)
            all_X_samples.append(None)
            all_performance.append({'accuracy': 0, 'f1_score': 0})

        # ========== CREATE COMPREHENSIVE COMPARISON ==========
        print(f"\n{'=' * 60}")
        print("Creating Comprehensive SHAP Comparison")
        print(f"{'=' * 60}")

        try:
            create_shap_comparison_visualization(
                all_shap_data, all_feature_names, all_model_names,
                all_original_indices, all_X_samples, device_name, output_dir, feature_color_map
            )

            # Save comprehensive performance comparison
            performance_comparison_df = pd.DataFrame(all_performance, index=all_model_names)
            performance_comparison_file = os.path.join(output_dir, f"{device_name}_performance_comparison.csv")
            performance_comparison_df.to_csv(performance_comparison_file)
            print(f"Performance comparison saved to {performance_comparison_file}")

        except Exception as e:
            print(f"Error creating comprehensive comparison: {str(e)}")

        print(f"\n{'=' * 60}")
        print(f"SHAP Analysis completed for {device_name}")
        print(f"All outputs saved to: {output_dir}")
        print(f"{'=' * 60}")

    except Exception as e:
        print(f"Critical error analyzing {device_name}: {str(e)}")
        import traceback
        traceback.print_exc()


def main():
    """Main execution function"""
    print("=" * 80)
    print("SHAP Feature Importance Analysis")
    print("Comparing ExtraTrees and TPOT with/without BigFeat")
    print(f"Starting from Device {DEVICE_START_INDEX}")
    print("=" * 80)

    start_time = time.time()

    # Check available CSV files
    available_files = [f for f in csv_files if os.path.exists(f)]
    if not available_files:
        print("ERROR: No CSV files found!")
        print("Expected files:")
        for f in csv_files:
            print(f"  {f}")
        sys.exit(1)

    print(f"Found {len(available_files)} CSV files to process")

    # Process each device
    for i, csv_file in enumerate(available_files, DEVICE_START_INDEX):
        device_name = f"Device{i}"
        print(f"\nProcessing {device_name} ({i-DEVICE_START_INDEX+1}/{len(available_files)})")

        try:
            analyze_device_shap(csv_file, device_name)
        except Exception as e:
            print(f"ERROR processing {device_name}: {str(e)}")
            continue

    total_time = time.time() - start_time
    print(f"\n{'=' * 80}")
    print(f"SHAP Analysis Complete!")
    print(f"Total execution time: {total_time:.2f} seconds")
    print(f"Results saved to: {output_dir}")
    print(f"{'=' * 80}")

if __name__ == "__main__":
    main()