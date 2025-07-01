import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import shap
import os
import warnings
import time
from tqdm import tqdm

warnings.filterwarnings('ignore')

data = pd.read_csv('../../Datasets/IOT Top 20 features datasets/device3_top_20_features.csv')

# Display basic information about the dataset
print("Dataset Information:")
print(f"Shape: {data.shape}")
print(f"Columns: {list(data.columns)}")
print(f"Label distribution:\n{data['label'].value_counts()}")

# Split the data into features (X) and labels (y)
# Assuming 'label' is the target column - adjust based on your dataset
feature_columns = [col for col in data.columns if col != 'label']
X = data[feature_columns]
y = data['label']

print(f"\nFeature columns ({len(feature_columns)}): {feature_columns}")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nData split:")
print(f"Training set: {X_train.shape}")
print(f"Testing set: {X_test.shape}")

# Create the ExtraTrees model
model = ExtraTreesClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=4,  # Use 4 cores for parallel processing
    bootstrap=False,  # ExtraTrees typically doesn't use bootstrap
    max_features='sqrt'  # Good default for classification
)

# Create directory for saving plots
shap_plot_dir = '../../Results/IoT/Feature_Importance/baiot_shap_plots/'
os.makedirs(shap_plot_dir, exist_ok=True)

print(f"\n{'=' * 50}")
print("Processing: ExtraTrees")
print(f"{'=' * 50}")

try:
    # Train the model
    print("Training ExtraTrees...")
    model.fit(X_train, y_train)
    print("✓ ExtraTrees trained successfully")

    # Make predictions and calculate metrics
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"✓ Model Performance:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")

    # Create SHAP explainer for ExtraTrees
    print("Creating SHAP explainer for ExtraTrees...")

    # Use TreeExplainer for ExtraTrees (faster and more accurate)
    explainer = shap.TreeExplainer(model)

    # Use the full test dataset for SHAP analysis (for research purposes)
    X_sample = X_test
    print(f"Using full test dataset for SHAP analysis: {len(X_sample)} samples")

    # Calculate SHAP values with progress tracking
    print("Calculating SHAP values... (this may take several minutes)")
    start_time = time.time()

    # For very large datasets, consider processing in batches
    batch_size = 500  # Process 500 samples at a time
    if len(X_sample) > batch_size:
        print(f"Processing in batches of {batch_size} samples...")
        shap_values_list = []

        # Create progress bar for batches
        num_batches = (len(X_sample) + batch_size - 1) // batch_size

        with tqdm(total=len(X_sample), desc="SHAP Progress", unit="samples") as pbar:
            for i in range(0, len(X_sample), batch_size):
                batch_end = min(i + batch_size, len(X_sample))
                batch_data = X_sample.iloc[i:batch_end]

                # Calculate SHAP values for this batch
                batch_shap_values = explainer.shap_values(batch_data)
                shap_values_list.append(batch_shap_values)

                # Update progress bar
                pbar.set_postfix({
                    'Batch': f"{i // batch_size + 1}/{num_batches}",
                    'Elapsed': f"{time.time() - start_time:.1f}s"
                })
                pbar.update(len(batch_data))

        # Combine all batch results
        if isinstance(shap_values_list[0], list):
            # Multi-class case
            shap_values = []
            for class_idx in range(len(shap_values_list[0])):
                class_values = np.vstack([batch[class_idx] for batch in shap_values_list])
                shap_values.append(class_values)
        else:
            # Binary or regression case
            shap_values = np.vstack(shap_values_list)
    else:
        # For smaller datasets, process all at once with simple progress indication
        print("Processing all samples at once...")
        with tqdm(total=1, desc="SHAP Calculation") as pbar:
            shap_values = explainer.shap_values(X_sample)
            pbar.update(1)

    elapsed_time = time.time() - start_time
    print(f"✓ SHAP values calculated in {elapsed_time:.1f} seconds")

    # Handle multi-class case
    if isinstance(shap_values, list):
        if len(shap_values) > 2:  # Multi-class
            shap_values_plot = np.abs(shap_values).mean(axis=0)
        else:  # Binary classification
            shap_values_plot = shap_values[1]
    else:
        shap_values_plot = shap_values

    # Create and save the SHAP summary plot (bar plot)
    print("Creating SHAP bar plot...")
    plt.figure(figsize=(12, 8))

    shap.summary_plot(shap_values_plot,
                      X_sample,
                      plot_type='bar',
                      show=False)

    plt.title('SHAP Feature Importance - ExtraTrees (BaIoT Dataset)')
    plt.tight_layout()

    # Save the bar plot
    bar_plot_filename = os.path.join(shap_plot_dir, 'baiot_shap_summary_bar_extratrees.jpg')
    plt.savefig(bar_plot_filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ SHAP bar plot saved: {bar_plot_filename}")

    # Create and save the SHAP summary plot (dot plot)
    print("Creating SHAP dot plot...")
    plt.figure(figsize=(12, 8))

    if isinstance(shap_values, list) and len(shap_values) == 2:
        # Binary classification - use positive class
        shap.summary_plot(shap_values[1], X_sample, show=False)
    elif isinstance(shap_values, list):
        # Multi-class - use first class or create combined plot
        shap.summary_plot(shap_values[0], X_sample, show=False)
    else:
        shap.summary_plot(shap_values, X_sample, show=False)

    plt.title('SHAP Feature Impact - ExtraTrees (BaIoT Dataset)')
    plt.tight_layout()

    # Save the dot plot
    dot_plot_filename = os.path.join(shap_plot_dir, 'baiot_shap_summary_dot_extratrees.jpg')
    plt.savefig(dot_plot_filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ SHAP dot plot saved: {dot_plot_filename}")

    # Display feature importance from the model itself
    print("\nTop 10 Feature Importances (from ExtraTrees):")
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print(feature_importance.head(10).to_string(index=False))

    # Save feature importance to CSV
    importance_filename = os.path.join(shap_plot_dir, 'baiot_feature_importance_extratrees.csv')
    feature_importance.to_csv(importance_filename, index=False)
    print(f"✓ Feature importance saved: {importance_filename}")

    print("✓ ExtraTrees analysis completed successfully!")

except Exception as e:
    print(f"✗ Error processing ExtraTrees: {str(e)}")
    import traceback

    traceback.print_exc()

print(f"\n{'=' * 50}")
print("BaIoT ExtraTrees SHAP Analysis Complete!")
print(f"All plots saved in: {shap_plot_dir}")
print(f"{'=' * 50}")