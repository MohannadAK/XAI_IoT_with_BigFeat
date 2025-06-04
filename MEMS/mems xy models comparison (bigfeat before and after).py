import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
import tpot
import time


def evaluate_models_with_tpot(X_data, y_data, title, output_filename):
    """
    Enhanced model evaluation function that includes TPOT optimization
    """
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

    # Create a dictionary to store model names and the corresponding model objects
    models = {
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'AdaBoost': AdaBoostClassifier(random_state=42),
        'SVM': SVC(random_state=42),
        'Deep Neural Network': MLPClassifier(max_iter=1000, random_state=42)
    }

    # Initialize an empty DataFrame to store performance metrics
    metrics_df = pd.DataFrame(columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'Training Time'])

    # Iterate through the traditional models and calculate performance metrics
    for model_name, model in models.items():
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        metrics_df = pd.concat([metrics_df, pd.DataFrame(
            [{'Model': model_name, 'Accuracy': accuracy, 'Precision': precision,
              'Recall': recall, 'F1 Score': f1, 'Training Time': training_time}])],
                               ignore_index=True)

    # Add TPOT optimization
    print(f"\nRunning TPOT optimization for {title}...")

    # TPOT with conservative settings for faster execution
    tpot_classifier = tpot.TPOTClassifier(
        generations=10,  # Reduced for faster execution
        population_size=20,  # Reduced for faster execution
        verbosity=2,  # Show progress
        random_state=42,
        n_jobs=-1,  # Use all available cores
        scoring='accuracy',  # Primary metric to optimize
        cv=3,  # Reduced CV folds for speed
        max_time_mins=10,  # Maximum 10 minutes
        max_eval_time_mins=2,  # Maximum 2 minutes per pipeline
        early_stop=5  # Stop if no improvement for 5 generations
    )

    start_time = time.time()
    tpot_classifier.fit(X_train, y_train)
    tpot_training_time = time.time() - start_time

    # Evaluate TPOT
    y_pred_tpot = tpot_classifier.predict(X_test)

    accuracy_tpot = accuracy_score(y_test, y_pred_tpot)
    precision_tpot = precision_score(y_test, y_pred_tpot, average='weighted')
    recall_tpot = recall_score(y_test, y_pred_tpot, average='weighted')
    f1_tpot = f1_score(y_test, y_pred_tpot, average='weighted')

    # Add TPOT results to metrics
    metrics_df = pd.concat([metrics_df, pd.DataFrame(
        [{'Model': 'TPOT (AutoML)', 'Accuracy': accuracy_tpot, 'Precision': precision_tpot,
          'Recall': recall_tpot, 'F1 Score': f1_tpot, 'Training Time': tpot_training_time}])],
                           ignore_index=True)

    # Export the best TPOT pipeline
    pipeline_filename = f"tpot_best_pipeline_{title.replace(' ', '_').lower()}.py"
    tpot_classifier.export(pipeline_filename)
    print(f"Best TPOT pipeline exported to: {pipeline_filename}")

    # Print the performance metrics table
    print(f"\nPerformance Metrics for {title} (including TPOT):")
    print(metrics_df)

    # Create a table image using matplotlib
    plt.figure(figsize=(12, 8))
    plt.axis('off')  # Turn off axis

    # Format cell text for better readability
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
    table.scale(1.2, 1.8)

    # Highlight TPOT row
    tpot_row_idx = len(metrics_df)  # TPOT is the last row
    for j in range(len(metrics_df.columns)):
        table._cells[(tpot_row_idx, j)].set_facecolor('#e6f3ff')  # Light blue for TPOT

    plt.title(f'MEMS Classification Performance: {title} (with TPOT)', fontsize=14)
    plt.tight_layout()

    # Save the table as a jpg image
    plt.savefig(output_filename, format='jpg', dpi=300, bbox_inches='tight')

    return metrics_df, tpot_classifier


def run_comprehensive_analysis():
    """
    Main function to run the comprehensive analysis with TPOT integration
    """
    if __name__ == "__main__":
        # Load your dataset (relative to \XAI_for_IoT_Systems\MEMS\)
        data = pd.read_csv('../Datasets/mems_dataset.csv')

        # Split the data into features (X) and labels (y)
        X = data[['x', 'y']]  # DataFrame
        y = data['label'].to_numpy()  # Convert Series to numpy array

        print(f"Dataset shape: {X.shape}")
        print(f"Label distribution: {np.unique(y, return_counts=True)}")

        # Initialize MinMaxScaler for original features
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

        # First run: evaluate performance with original features only (including TPOT)
        print("=" * 80)
        print("PHASE 1: Evaluating Original Features with Traditional ML + TPOT")
        print("=" * 80)
        original_metrics, tpot_original = evaluate_models_with_tpot(
            X_scaled_df, y, "Original Features Only",
            "../Results/MEMS/Mems_XY_BigFeat_with_TPOT/mems_xy_original_with_tpot_metrics.jpg"
        )

        # Now run with BigFeat features
        # Prioritize the local BigFeat package
        bigfeat_path = r'/mnt/Projects/BigFeat'  # Parent directory of bigfeat package
        sys.path.insert(0, bigfeat_path)  # Insert at start of sys.path to override site-packages

        # Import BigFeat
        from bigfeat.bigfeat_base import BigFeat

        # Initialize BigFeat for classification
        bigfeat = BigFeat(task_type='classification')

        print("\n" + "=" * 80)
        print("PHASE 2: Generating BigFeat Features")
        print("=" * 80)

        try:
            # First try with standard settings but RF estimator which is more stable
            print("\nGenerating BigFeat features...")
            X_bigfeat = bigfeat.fit(
                X, y,
                gen_size=5,  # Number of features to generate per iteration
                random_state=42,  # For reproducibility
                iterations=5,  # Number of iterations for feature generation
                estimator='rf',  # Use only RandomForest for importance
                feat_imps=True,  # Use feature importances
                check_corr=True,  # Check for correlated features
                selection='stability'  # Use stability selection
            )
        except ValueError as e:
            if "probabilities contain NaN" in str(e):
                print("Caught NaN probabilities error, using simpler approach...")
                # Fallback to a simpler configuration
                bigfeat = BigFeat(task_type='classification')
                X_bigfeat = bigfeat.fit(
                    X, y,
                    gen_size=5,
                    random_state=42,
                    iterations=2,
                    estimator='rf',  # Only use RandomForest
                    feat_imps=False,  # Turn off feature importance weighting
                    check_corr=True,
                    selection=None  # Disable stability selection
                )
            else:
                raise e

        # Transform the test data
        X_bigfeat = bigfeat.transform(X)

        # Convert transformed features to DataFrame for consistency
        X_bigfeat = pd.DataFrame(X_bigfeat, columns=[f'feat_{i}' for i in range(X_bigfeat.shape[1])])

        print(f"BigFeat generated {X_bigfeat.shape[1]} features from {X.shape[1]} original features")

        # Second run: evaluate performance with BigFeat features (including TPOT)
        print("\n" + "=" * 80)
        print("PHASE 3: Evaluating BigFeat Features with Traditional ML + TPOT")
        print("=" * 80)
        bigfeat_metrics, tpot_bigfeat = evaluate_models_with_tpot(
            X_bigfeat, y, "With BigFeat Features",
            "../Results/MEMS/Mems_XY_BigFeat_with_TPOT/mems_xy_bigfeat_with_tpot_metrics.jpg"
        )

        # Compare the results
        print("\n" + "=" * 80)
        print("PHASE 4: Comprehensive Comparison Analysis")
        print("=" * 80)

        mean_improvement = {}
        for metric in ['Accuracy', 'Precision', 'Recall', 'F1 Score']:
            diff = bigfeat_metrics[metric] - original_metrics[metric]
            mean_improvement[metric] = diff.mean()

        # Create a comprehensive comparison table
        comparison_df = pd.DataFrame()
        comparison_df['Model'] = original_metrics['Model']

        for metric in ['Accuracy', 'Precision', 'Recall', 'F1 Score']:
            comparison_df[f'Original {metric}'] = original_metrics[metric]
            comparison_df[f'BigFeat {metric}'] = bigfeat_metrics[metric]
            comparison_df[f'{metric} Diff'] = bigfeat_metrics[metric] - original_metrics[metric]

        # Add a summary row
        summary_row = {'Model': 'AVERAGE'}
        for metric in ['Accuracy', 'Precision', 'Recall', 'F1 Score']:
            summary_row[f'Original {metric}'] = original_metrics[metric].mean()
            summary_row[f'BigFeat {metric}'] = bigfeat_metrics[metric].mean()
            summary_row[f'{metric} Diff'] = mean_improvement[metric]

        comparison_df = pd.concat([comparison_df, pd.DataFrame([summary_row])], ignore_index=True)

        # Print the comparison
        print("\nComprehensive Comparison: Original vs BigFeat features (including TPOT):")
        print(comparison_df)

        # Find the best performing model overall
        best_original_idx = original_metrics['Accuracy'].idxmax()
        best_bigfeat_idx = bigfeat_metrics['Accuracy'].idxmax()

        best_original_model = original_metrics.loc[best_original_idx, 'Model']
        best_original_acc = original_metrics.loc[best_original_idx, 'Accuracy']

        best_bigfeat_model = bigfeat_metrics.loc[best_bigfeat_idx, 'Model']
        best_bigfeat_acc = bigfeat_metrics.loc[best_bigfeat_idx, 'Accuracy']

        print(f"\n" + "=" * 60)
        print("SUMMARY OF BEST RESULTS:")
        print("=" * 60)
        print(f"Best Original Features Model: {best_original_model} (Accuracy: {best_original_acc:.4f})")
        print(f"Best BigFeat Features Model: {best_bigfeat_model} (Accuracy: {best_bigfeat_acc:.4f})")

        if best_bigfeat_acc > best_original_acc:
            improvement = ((best_bigfeat_acc - best_original_acc) / best_original_acc) * 100
            print(f"BigFeat improved performance by {improvement:.2f}%")
        else:
            decline = ((best_original_acc - best_bigfeat_acc) / best_original_acc) * 100
            print(f"BigFeat decreased performance by {decline:.2f}%")

        # Create enhanced comparison visualization
        plt.figure(figsize=(16, 12))
        plt.axis('off')

        # Format cell text for better readability
        cell_text = []
        for _, row in comparison_df.iterrows():
            row_values = []
            for col in comparison_df.columns:
                if isinstance(row[col], float):
                    if 'Diff' in col:
                        val = row[col]
                        value_str = f"{val:+.4f}"  # Show + for positive differences
                    else:
                        value_str = f"{row[col]:.4f}"
                else:
                    value_str = str(row[col])
                row_values.append(value_str)
            cell_text.append(row_values)

        # Create the table
        table = plt.table(
            cellText=cell_text,
            colLabels=comparison_df.columns,
            cellLoc='center',
            loc='center',
            colColours=['#f2f2f2'] * len(comparison_df.columns)
        )

        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.5, 2.0)

        # Color the diff cells and highlight TPOT rows
        for i in range(len(comparison_df)):
            model_name = comparison_df.iloc[i]['Model']
            for j, col in enumerate(comparison_df.columns):
                cell = table._cells[(i + 1, j)]

                # Highlight TPOT rows
                if 'TPOT' in str(model_name):
                    cell.set_facecolor('#ffe6cc')  # Light orange for TPOT

                # Color difference columns
                if 'Diff' in col:
                    val = comparison_df.iloc[i][col]
                    if val > 0:
                        cell.set_facecolor('#d4f7d4')  # Light green for improvement
                    elif val < 0:
                        cell.set_facecolor('#f7d4d4')  # Light red for worse

        plt.title('Comprehensive Comparison: Original vs BigFeat Features (with TPOT AutoML)', fontsize=16)
        plt.tight_layout()

        # Save the comparison table
        plt.savefig('mems_comprehensive_comparison_with_tpot.jpg', format='jpg', dpi=300, bbox_inches='tight')

        print("\nComprehensive analysis complete!")
        print("Results saved to 'mems_comprehensive_comparison_with_tpot.jpg'")

        # Save all results for future reference
        original_metrics.to_csv('mems_original_with_tpot_metrics.csv', index=False)
        bigfeat_metrics.to_csv('mems_bigfeat_with_tpot_metrics.csv', index=False)
        comparison_df.to_csv('mems_comprehensive_comparison_with_tpot.csv', index=False)

        # Optional: save the transformed dataset for inspection
        X_bigfeat_with_labels = X_bigfeat.copy()
        X_bigfeat_with_labels['label'] = y
        X_bigfeat_with_labels.to_csv('mems_bigfeat_transformed_with_tpot.csv', index=False)

        # Generate feature importance visualization
        print("\n" + "=" * 60)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("=" * 60)

        if hasattr(bigfeat, 'tracking_ids') and len(bigfeat.tracking_ids) > 0:
            print("\nFeature importance of BigFeat generated features:")

            # Train a random forest to get feature importance
            rf = RandomForestClassifier(random_state=42)
            rf.fit(X_bigfeat, y)

            # Get feature importance
            importances = rf.feature_importances_
            feature_names = X_bigfeat.columns

            # Sort features by importance
            indices = np.argsort(importances)[::-1]

            # Take top 15 or all if less
            top_n = min(15, len(indices))

            plt.figure(figsize=(12, 8))
            plt.title('Top BigFeat Generated Features by Importance')
            bars = plt.bar(range(top_n), importances[indices[:top_n]], align='center')
            plt.xticks(range(top_n), [feature_names[i] for i in indices[:top_n]], rotation=45, ha='right')
            plt.ylabel('Feature Importance')
            plt.xlabel('Features')

            # Color bars based on importance
            for i, bar in enumerate(bars):
                if i < 5:  # Top 5 features
                    bar.set_color('#2E8B57')  # Dark green
                else:
                    bar.set_color('#87CEEB')  # Light blue

            plt.tight_layout()
            plt.savefig('mems_bigfeat_feature_importance_with_tpot.jpg', dpi=300, bbox_inches='tight')

            # Print out the top features with their importance values
            print("Top features by importance:")
            for i in range(top_n):
                print(f"{feature_names[indices[i]]}: {importances[indices[i]]:.4f}")

        print("\n" + "=" * 60)
        print("TPOT PIPELINE INSIGHTS")
        print("=" * 60)
        print("Check the exported pipeline files to see the optimized ML pipelines:")
        print("- tpot_best_pipeline_original_features_only.py")
        print("- tpot_best_pipeline_with_bigfeat_features.py")
        print("\nThese files contain the exact scikit-learn code for the best pipelines found by TPOT.")


# Run the analysis
run_comprehensive_analysis()