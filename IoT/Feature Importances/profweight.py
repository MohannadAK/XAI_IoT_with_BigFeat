# -*- coding: utf-8 -*-
"""SGD metrics.ipynb"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# List of 9 CSV files
csv_files = [
    '../../Datasets/IOT Top 20 features datasets/device1_top_20_features.csv',
    '../../Datasets/IOT Top 20 features datasets/device2_top_20_features.csv',
    '../../Datasets/IOT Top 20 features datasets/device3_top_20_features.csv',
    '../../Datasets/IOT Top 20 features datasets/device4_top_20_features.csv',
    '../../Datasets/IOT Top 20 features datasets/device5_top_20_features.csv',
    '../../Datasets/IOT Top 20 features datasets/device6_top_20_features.csv',
    '../../Datasets/IOT Top 20 features datasets/device7_top_20_features.csv',
    '../../Datasets/IOT Top 20 features datasets/device8_top_20_features.csv',
    '../../Datasets/IOT Top 20 features datasets/device9_top_20_features.csv'
]

# Initialize an empty DataFrame to store aggregated performance metrics
all_metrics_df = pd.DataFrame(columns=['File', 'Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])

# Loop through each CSV file
for i, csv_file in enumerate(csv_files):
    print(f"Processing file {i + 1} of 9: {csv_file}")

    # Load dataset from the current CSV file
    data = pd.read_csv(csv_file)

    # Split the data into features (X) and labels (y)
    X = data.drop(columns=['label'])
    y = data['label']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a dictionary to store model names and the corresponding model objects
    models = {
        'SGD Classifier': SGDClassifier(random_state=42)
    }

    # Initialize an empty DataFrame to store performance metrics for the current file
    metrics_df = pd.DataFrame(columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])

    # Iterate through the models and calculate performance metrics
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        # Create a new DataFrame for the current metrics
        new_metrics = pd.DataFrame([{
            'Model': model_name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1
        }])

        # Concatenate the new metrics to metrics_df
        metrics_df = pd.concat([metrics_df, new_metrics], ignore_index=True)

    # Add the file information to the metrics DataFrame
    metrics_df['File'] = csv_file
    all_metrics_df = pd.concat([all_metrics_df, metrics_df], ignore_index=True)

# Create a table image using matplotlib
plt.figure(figsize=(12, 8))
plt.axis('off')  # Turn off axis
plt.table(cellText=all_metrics_df.values, colLabels=all_metrics_df.columns, cellLoc='center', loc='center', colColours=['#f2f2f2']*len(all_metrics_df.columns))
plt.tight_layout()

# Save the table as a JPG image
plt.savefig('sgd_metrics.jpg', format='jpg')

# Display the table
plt.show()

# Feature Importance Section
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import permutation_importance

# List of 9 CSV files (adjusted paths if needed)
csv_files = [
    '../../Datasets/IOT Top 20 features datasets/device1_top_20_features.csv',
    '../../Datasets/IOT Top 20 features datasets/device2_top_20_features.csv',
    '../../Datasets/IOT Top 20 features datasets/device3_top_20_features.csv',
    '../../Datasets/IOT Top 20 features datasets/device4_top_20_features.csv',
    '../../Datasets/IOT Top 20 features datasets/device5_top_20_features.csv',
    '../../Datasets/IOT Top 20 features datasets/device6_top_20_features.csv',
    '../../Datasets/IOT Top 20 features datasets/device7_top_20_features.csv',
    '../../Datasets/IOT Top 20 features datasets/device8_top_20_features.csv',
    '../../Datasets/IOT Top 20 features datasets/device9_top_20_features.csv'
]

# Loop through each CSV file
for i, csv_file in enumerate(csv_files):
    print(f"Processing file {i + 1} of 9: {csv_file}")

    # Load dataset from the current CSV file
    data = pd.read_csv(csv_file)

    # Separate features and labels
    X = data.drop(columns=['label'])
    y = data['label']

    # Define models
    random_forest_model = RandomForestClassifier()
    decision_tree_model = DecisionTreeClassifier()
    ada_model = AdaBoostClassifier(estimator=DecisionTreeClassifier(), n_estimators=100)

    # Fit models to the data
    random_forest_model.fit(X, y)
    decision_tree_model.fit(X, y)
    ada_model.fit(X, y)

    # Calculate ProfWeight-based feature importances for Random Forest
    pfi_result_rf = permutation_importance(random_forest_model, X, y, scoring='neg_mean_squared_error', n_repeats=30, random_state=42)
    rf_feature_importances = pfi_result_rf.importances_mean

    # Calculate ProfWeight-based feature importances for Decision Tree
    pfi_result_dt = permutation_importance(decision_tree_model, X, y, scoring='neg_mean_squared_error', n_repeats=30, random_state=42)
    dt_feature_importances = pfi_result_dt.importances_mean

    # Calculate ProfWeight-based feature importances for AdaBoost
    pfi_result_ada = permutation_importance(ada_model, X, y, scoring='neg_mean_squared_error', n_repeats=30, random_state=42)
    ada_feature_importances = pfi_result_ada.importances_mean

    # Plot and save ProfWeight-based feature importances for all models
    def save_feature_importance_plot(feature_importances, title, filename):
        plt.figure(figsize=(8, 6))
        plt.bar(range(len(X.columns)), feature_importances)
        plt.xticks(range(len(X.columns)), X.columns, rotation=90)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    save_feature_importance_plot(rf_feature_importances, f'd{i + 1} Random Forest ProfWeight Feature Importances', f'd{i + 1}_random_forest_profweight_feature_importance.jpg')
    save_feature_importance_plot(dt_feature_importances, f'd{i + 1} Decision Tree ProfWeight Feature Importances', f'd{i + 1}_decision_tree_profweight_feature_importance.jpg')
    save_feature_importance_plot(ada_feature_importances, f'd{i + 1} AdaBoost ProfWeight Feature Importances', f'd{i + 1}_ada_profweight_feature_importance.jpg')