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

# Prioritize the local BigFeat package
bigfeat_path = r'D:\BigFeat'  # Parent directory of bigfeat package
sys.path.insert(0, bigfeat_path)  # Insert at start of sys.path to override site-packages

# Import BigFeat
from bigfeat.bigfeat_base import BigFeat

# Verify the correct BigFeat is imported
import bigfeat.bigfeat_base

print(f"BigFeat imported from: {bigfeat.bigfeat_base.__file__}")

# Load your dataset (relative to D:\XAI_for_IoT_Systems\MEMS\)
data = pd.read_csv('../Datasets/mems_dataset.csv')

# Split the data into features (X) and labels (y)
X = data[['x', 'y']]  # DataFrame
y = data['label'].to_numpy()  # Convert Series to numpy array

# Initialize BigFeat for classification with more robust parameters
bigfeat = BigFeat(task_type='classification')

try:
    # First try with standard settings but RF estimator which is more stable
    X_bigfeat = bigfeat.fit(
        X, y,
        gen_size=5,  # Number of features to generate per iteration
        random_state=42,  # For reproducibility
        iterations=5,  # Number of iterations for feature generation
        estimator='rf',  # Use only RandomForest for importance (more stable than 'avg')
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

# Transform the test data (already done in fit for training, but needed for consistency)
X_bigfeat = bigfeat.transform(X)

# Convert transformed features to DataFrame for consistency
X_bigfeat = pd.DataFrame(X_bigfeat, columns=[f'feat_{i}' for i in range(X_bigfeat.shape[1])])

# Split the transformed data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_bigfeat, y, test_size=0.2, random_state=42)

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
print("\nPerformance Metrics:")
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
plt.title('MEMS Classification Performance with BigFeat Features')
plt.tight_layout()

# Save the table as a jpg image
plt.savefig('mems_xy_bigfeat_metrics.jpg', format='jpg', dpi=300, bbox_inches='tight')

# Display the table
plt.show()

# Optional: save the transformed dataset for inspection
X_bigfeat_df = pd.DataFrame(X_bigfeat, columns=[f'feat_{i}' for i in range(X_bigfeat.shape[1])])
X_bigfeat_df['label'] = y
X_bigfeat_df.to_csv('mems_bigfeat_transformed.csv', index=False)

print("Analysis complete. Transformed dataset saved to 'mems_bigfeat_transformed.csv'")