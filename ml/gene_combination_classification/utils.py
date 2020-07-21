import os
import numpy as np
from sklearn.metrics import roc_auc_score
import pandas as pd

def convert_surgery(x):
    if (x == "mastectomy"):
        return 1
    if (x == "breast preserving"):
        return 2
    if (x == 'NA'):
        return 0
    raise Exception("bad surgery")


def convert_posOutcome(x):
    if (x == '2'):
        return 1
    return int(x)


def read_treat_dataset(path="./"):
    return pd.read_csv(os.path.join(path, 'bmc15mldata1.csv'), converters=dict(surgery = convert_surgery, posOutcome = convert_posOutcome))


def prepare_full_dataset(full_dataset):
    X = full_dataset.drop(columns=['study', 'patient_ID','pCR', 'RFS', 'DFS', 'posOutcome']).to_numpy()
    y_posOutcome = full_dataset['posOutcome'].to_numpy()
    return X, y_posOutcome


def calc_results_for_fold(X, y, train_index, test_index, clf):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # clf = XGBClassifier()
    clf.fit(X_train, y_train)

    y_true_prob = clf.predict_proba(X_test)[:, 1]
    y_true = clf.predict(X_test)

    acc = np.mean(y_true == y_test)
    auc_1 = roc_auc_score(y_test, y_true_prob)
    auc_2 = roc_auc_score(y_test, y_true)

    return acc, auc_1, auc_2
