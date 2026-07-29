import os
import sys
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tpot import TPOTClassifier
from tpot.old_config_utils import convert_config_dict_to_choicepipeline

sys.path.insert(0, '/mnt/Projects/BigFeat')
from bigfeat.bigfeat_base import BigFeat

csv_file = '../Datasets/IOT Top 20 features datasets/device1_top_20_features.csv'
data = pd.read_csv(csv_file)
X = data.drop(columns=['label'])
y = data['label'].values

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

bigfeat = BigFeat(task_type='classification')
X_bigfeat = bigfeat.fit(
    X_train, y_train,
    gen_size=10, random_state=42, iterations=10, estimator='rf',
    feat_imps=True, check_corr=True, selection='stability'
)

tpot_config = {
    'sklearn.ensemble.ExtraTreesClassifier': {
        'n_estimators': [100, 200, 300],
        'criterion': ['gini', 'entropy'],
        'max_features': ['sqrt', 'log2'],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 5],
        'bootstrap': [True, False],
        'max_depth': [None, 10, 20]
    }
}

tpot_classifier = TPOTClassifier(
    search_space=convert_config_dict_to_choicepipeline(tpot_config),
    generations=8,
    population_size=15,
    verbose=1,
    random_state=42,
    n_jobs=1,
    scorers=['f1_weighted'],
    cv=3,
    max_time_mins=None,
    max_eval_time_mins=5,
    early_stop=5
)

print("Starting TPOT...")
start = time.time()
tpot_classifier.fit(X_bigfeat, y_train)
end = time.time()

print(f"Total TPOT time: {end - start:.2f} seconds ({(end - start)/60:.2f} minutes)")
