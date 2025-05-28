import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neural_network import MLPClassifier
import shap
import os
import warnings

warnings.filterwarnings('ignore')

# Load your dataset
data = pd.read_csv('../../Datasets/mems_dataset.csv')

# Split the data into features (X) and labels (y)
X = data[['x', 'y', 'z']]
y = data['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a dictionary to store model names and the corresponding model objects
models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'AdaBoost': AdaBoostClassifier(random_state=42),
    'SVM': SVC(probability=True, random_state=42),
    'Deep Neural Network': MLPClassifier(random_state=42, max_iter=500)
}

shap_plot_dir = '../../Results/MEMS/Feature_Importance/mems_shap_plots/'

# Ensure the directory exists
os.makedirs(shap_plot_dir, exist_ok=True)

# Process each model individually with error handling
for model_name, model in models.items():
    print(f"\n{'=' * 50}")
    print(f"Processing: {model_name}")
    print(f"{'=' * 50}")

    try:
        # Train the model
        print(f"Training {model_name}...")
        model.fit(X_train, y_train)
        print(f"✓ {model_name} trained successfully")

        # Make predictions for accuracy check
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"✓ {model_name} accuracy: {accuracy:.4f}")

        # Create SHAP explainer based on model type
        print(f"Creating SHAP explainer for {model_name}...")

        # Use different explainer strategies based on model type
        if model_name == 'Decision Tree' or model_name == 'Random Forest':
            # Use TreeExplainer for tree-based models (faster and more accurate)
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)

            if isinstance(shap_values, list):
                if len(shap_values) > 2:  # Multi-class
                    shap_values_plot = np.abs(shap_values).mean(axis=0)
                else:  # Binary classification
                    shap_values_plot = shap_values[1]
            else:
                shap_values_plot = shap_values

        elif hasattr(model, 'predict_proba'):
            # For models with predict_proba (most sklearn models)
            explainer = shap.KernelExplainer(model.predict_proba, X_train)  # Full training set
            shap_values = explainer.shap_values(X_test)

            # For multi-class, shap_values is a list - use the first class or mean
            if isinstance(shap_values, list):
                if len(shap_values) > 2:  # Multi-class
                    shap_values_plot = np.abs(shap_values).mean(axis=0)
                else:  # Binary classification - use positive class
                    shap_values_plot = shap_values[1]
            else:
                shap_values_plot = shap_values
        else:
            # Fallback to KernelExplainer with predict
            explainer = shap.KernelExplainer(model.predict, X_train)
            shap_values_plot = explainer.shap_values(X_test)

        print(f"✓ SHAP values calculated for {model_name}")

        # Create and save the SHAP summary plot
        print(f"Creating SHAP plot for {model_name}...")
        plt.figure(figsize=(10, 6))

        # Create bar plot
        shap.summary_plot(shap_values_plot,
                          X_test,
                          plot_type='bar',
                          show=False)

        plt.title(f'SHAP Feature Importance - {model_name}')
        plt.tight_layout()

        # Save the plot
        plot_filename = os.path.join(shap_plot_dir, f'mems_shap_summary_bar_{model_name.replace(" ", "_")}.jpg')
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ SHAP plot saved: {plot_filename}")
        print(f"✓ {model_name} completed successfully!")

    except Exception as e:
        print(f"✗ Error processing {model_name}: {str(e)}")
        print(f"✗ Skipping {model_name} and continuing with next model...")
        continue

print(f"\n{'=' * 50}")
print("SHAP Analysis Complete!")
print(f"All plots saved in: {shap_plot_dir}")
print(f"{'=' * 50}")