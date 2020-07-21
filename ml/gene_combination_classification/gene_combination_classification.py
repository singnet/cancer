import os
import numpy as np
import random
import math
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import pandas as pd
from gene_combination_generator import CombinationsGenerator


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

labels = read_treat_dataset("/home/oleg/Work/bc_data_procesor/git/bc_data_procesor/")

X_lab, y_pos_out = prepare_full_dataset(labels)

comb_gen = CombinationsGenerator("ex15bmcMerged70genes.csv")
datasets, gene_names_comb = comb_gen.generate(3)
print(datasets[0].shape)

clf1 = XGBClassifier()
clf2 = LogisticRegression()
clf_map = {"xgboost" : clf1, "logistic_regression" : clf2}
kf = KFold(n_splits=5, shuffle=True)
results_folder = "3_genes"
for k, v in clf_map.items():
    print("::::::::::::::::::::"+k+"::::::::::::::::::::")
    with open(results_folder+k+"_results.txt", 'a') as out:
        for X_full, gene_pair in zip(datasets, gene_names_comb):
            print('========================', gene_pair, '==============================')
            out.write(str(gene_pair))
            out.write('\n')
            for i, (train_index, test_index) in enumerate(kf.split(X_full)):
                acc_f, auc_1_f, auc_2_f = calc_results_for_fold(X_full, y_pos_out, train_index, test_index, v)
                print("full: ", acc_f, auc_1_f, auc_2_f)
                out.write(str([acc_f, auc_1_f, auc_2_f]))
                out.write('\n')
        out.close()

print(X_lab.shape)
print(y_pos_out.shape)