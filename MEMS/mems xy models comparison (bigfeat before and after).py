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

# Load your dataset (relative to D:\XAI_for_IoT_Systems\MEMS\)
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


# Create a function to evaluate models
def evaluate_models(X_data, y_data, title, output_filename):
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
    metrics_df = pd.DataFrame(columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])

    # Iterate through the models and calculate performance metrics
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        metrics_df = pd.concat([metrics_df, pd.DataFrame(
            [{'Model': model_name, 'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1 Score': f1}])],
                               ignore_index=True)

    # Print the performance metrics table
    print(f"\nPerformance Metrics for {title}:")
    print(metrics_df)

    # Create a table image using matplotlib
    plt.figure(figsize=(10, 6))
    plt.axis('off')  # Turn off axis
    table = plt.table(
        cellText=[[f"{val:.4f}" if isinstance(val, float) else val for val in row]
                  for row in metrics_df.values],
        colLabels=metrics_df.columns,
        cellLoc='center',
        loc='center',
        colColours=['#f2f2f2'] * len(metrics_df.columns)
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    plt.title(f'MEMS Classification Performance: {title}')
    plt.tight_layout()

    # Save the table as a jpg image
    plt.savefig(output_filename, format='jpg', dpi=300, bbox_inches='tight')

    return metrics_df


# First run: evaluate performance with original features only
original_metrics = evaluate_models(X_scaled_df, y, "Original Features Only", "mems_xy_original_metrics.jpg")

# Now run with BigFeat features
# Prioritize the local BigFeat package
bigfeat_path = r'D:\BigFeat'  # Parent directory of bigfeat package
sys.path.insert(0, bigfeat_path)  # Insert at start of sys.path to override site-packages

# Import BigFeat
from bigfeat.bigfeat_base import BigFeat

# Initialize BigFeat for classification
bigfeat = BigFeat(task_type='classification')

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

# Second run: evaluate performance with BigFeat features
bigfeat_metrics = evaluate_models(X_bigfeat, y, "With BigFeat Features", "mems_xy_bigfeat_metrics.jpg")

# Compare the results
mean_improvement = {}
for metric in ['Accuracy', 'Precision', 'Recall', 'F1 Score']:
    diff = bigfeat_metrics[metric] - original_metrics[metric]
    mean_improvement[metric] = diff.mean()

# Create a comparison table
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
print("\nComparison between Original and BigFeat features:")
print(comparison_df)

# Create a comparison visualization
plt.figure(figsize=(15, 10))
plt.axis('off')  # Turn off axis

# Format cell text for better readability
cell_text = []
for _, row in comparison_df.iterrows():
    row_values = []
    for col in comparison_df.columns:
        if isinstance(row[col], float):
            # Format differences with color indication in HTML
            if 'Diff' in col:
                val = row[col]
                if val > 0:
                    value_str = f"{val:.4f}"
                elif val < 0:
                    value_str = f"{val:.4f}"
                else:
                    value_str = f"{val:.4f}"
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
table.scale(1.5, 1.5)

# Color the diff cells based on whether they are positive or negative
for i in range(len(comparison_df)):
    for j, col in enumerate(comparison_df.columns):
        if 'Diff' in col:
            cell = table._cells[(i + 1, j)]
            val = comparison_df.iloc[i][col]
            if val > 0:
                cell.set_facecolor('#d4f7d4')  # Light green for improvement
            elif val < 0:
                cell.set_facecolor('#f7d4d4')  # Light red for worse

plt.title('Comparison: Original vs BigFeat Features', fontsize=16)
plt.tight_layout()

# Save the comparison table
plt.savefig('mems_comparison.jpg', format='jpg', dpi=300, bbox_inches='tight')

print("\nComparison complete. Results saved to 'mems_comparison.jpg'")

# Optional: save the transformed dataset for inspection
X_bigfeat['label'] = y
X_bigfeat.to_csv('mems_bigfeat_transformed.csv', index=False)

# Save the original metrics and bigfeat metrics for future reference
original_metrics.to_csv('mems_original_metrics.csv', index=False)
bigfeat_metrics.to_csv('mems_bigfeat_metrics.csv', index=False)
comparison_df.to_csv('mems_comparison_metrics.csv', index=False)

# Generate a visualization showing feature importance (for BigFeat)
if hasattr(bigfeat, 'tracking_ids') and len(bigfeat.tracking_ids) > 0:
    print("\nFeature importance of generated features:")

    # Train a random forest to get feature importance
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_bigfeat.iloc[:, :-1], y)  # Exclude the label column

    # Get feature importance
    importances = rf.feature_importances_
    feature_names = X_bigfeat.columns[:-1]  # Exclude the label column

    # Sort features by importance
    indices = np.argsort(importances)[::-1]

    # Take top 15 or all if less
    top_n = min(15, len(indices))

    plt.figure(figsize=(10, 8))
    plt.title('Feature Importances')
    plt.bar(range(top_n), importances[indices[:top_n]], align='center')
    plt.xticks(range(top_n), [feature_names[i] for i in indices[:top_n]], rotation=90)
    plt.tight_layout()
    plt.savefig('mems_feature_importance.jpg', dpi=300, bbox_inches='tight')

    # Print out the top features with their importance values
    print("Top features by importance:")
    for i in range(top_n):
        print(f"{feature_names[indices[i]]}: {importances[indices[i]]:.4f}")