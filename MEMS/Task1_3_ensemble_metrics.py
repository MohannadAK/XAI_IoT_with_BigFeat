import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from mlxtend.classifier import StackingClassifier
import matplotlib.pyplot as plt
from tabulate import tabulate

# Load your dataset
dataset = pd.read_csv('../Datasets/mems_dataset.csv')

# Handle missing values
dataset[['x', 'y', 'z']] = dataset[['x', 'y', 'z']].fillna(dataset[['x', 'y', 'z']].mean())

# Extract features (X) and labels (y)
X = dataset[['x', 'y', 'z']].values
y = dataset['label'].values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Bagging
bagging_classifier = BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=100, random_state=42)
bagging_classifier.fit(X_train, y_train)
bagging_pred = bagging_classifier.predict(X_test)

# AdaBoost
adaboost_classifier = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1), n_estimators=100, random_state=42)
adaboost_classifier.fit(X_train, y_train)
adaboost_pred = adaboost_classifier.predict(X_test)

# Random Forest
random_forest_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest_classifier.fit(X_train, y_train)
random_forest_pred = random_forest_classifier.predict(X_test)

# Stacking
stacking_classifier = StackingClassifier(classifiers=[bagging_classifier, adaboost_classifier, random_forest_classifier],
                                         meta_classifier=LogisticRegression(), use_probas=True)
stacking_classifier.fit(X_train, y_train)
stacking_pred = stacking_classifier.predict(X_test)

# Blending (Manual Implementation)
base_learners = [bagging_classifier, adaboost_classifier, random_forest_classifier]
blend_pred = np.zeros((len(X_test), len(np.unique(y))))  # Dynamic number of classes
for base_learner in base_learners:
    base_learner.fit(X_train, y_train)
    blend_pred += base_learner.predict_proba(X_test)
blend_pred = np.argmax(blend_pred, axis=1)

# Voting
voting_classifier = VotingClassifier(estimators=[('bagging', bagging_classifier),
                                                 ('adaboost', adaboost_classifier),
                                                 ('random_forest', random_forest_classifier)],
                                     voting='hard')
voting_classifier.fit(X_train, y_train)
voting_pred = voting_classifier.predict(X_test)

# Evaluate and report metrics
def evaluate_and_report(y_true, y_pred, ensemble_name):
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)

    # Convert conf_matrix to a string for tabulate
    conf_matrix_str = np.array2string(conf_matrix, separator=', ')

    # Create a table from metrics
    table = [
        ["Accuracy", f"{accuracy:.4f}"],
        ["Classification Report", report],
        ["Confusion Matrix", conf_matrix_str]
    ]

    # Save the table as a text file
    with open(f'{ensemble_name}_metrics.txt', 'w') as f:
        f.write(tabulate(table, headers=["Metric", "Value"], tablefmt="grid"))

    # Create a figure for the table and save it as an image
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=table, colLabels=["Metric", "Value"], cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.5, 2)  # Adjust the table size as needed

    # Save the table as a JPG file
    plt.savefig(f'{ensemble_name}_metrics.jpg', bbox_inches='tight', pad_inches=0.1)
    plt.close()

    # Save the confusion matrix as a JPG file
    plt.figure(figsize=(8, 6))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - {ensemble_name}')
    plt.colorbar()
    num_classes = len(np.unique(y_true))
    plt.xticks(np.arange(num_classes), [f'Class {i+1}' for i in range(num_classes)])
    plt.yticks(np.arange(num_classes), [f'Class {i+1}' for i in range(num_classes)])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(f'{ensemble_name}_confusion_matrix.jpg')
    plt.close()

# Evaluate and report metrics for each ensemble method
evaluate_and_report(y_test, bagging_pred, "Bagging")
evaluate_and_report(y_test, adaboost_pred, "AdaBoost")
evaluate_and_report(y_test, random_forest_pred, "Random Forest")
evaluate_and_report(y_test, stacking_pred, "Stacking")
evaluate_and_report(y_test, blend_pred, "Blending")
evaluate_and_report(y_test, voting_pred, "Voting")