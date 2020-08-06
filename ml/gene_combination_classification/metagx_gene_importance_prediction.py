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
from utils import *
import json


def convert_study(val):
    d = json.loads(open('metagx_study_labels.json').read())
    return d[val]


metagx_covar_data_converted = pd.read_csv('/mnt/fileserver/shared/datasets/biodata/MetaGX/metaGXcovarTable.csv', converters=dict(study = convert_study))
studies = metagx_covar_data_converted['study']
print(studies[8000:9000])

metagx_expr_data = pd.read_csv('/mnt/fileserver/shared/datasets/biodata/MetaGX/merged/noNormMergedAnd10k.csv')
print(metagx_expr_data.shape)

metagx_result_data = pd.merge(metagx_covar_data_converted, metagx_expr_data, on='sample_name')
print(metagx_result_data.shape)

y_ = metagx_result_data['study'].to_numpy()
print(y_)
cols_to_drop = metagx_result_data.iloc[:, 0:29]
X_ = metagx_result_data.drop(columns=cols_to_drop).to_numpy()
gene_names = metagx_result_data.drop(columns=cols_to_drop).columns.values.tolist()
print(metagx_result_data.drop(columns=cols_to_drop).shape)
print(metagx_result_data.drop(columns=cols_to_drop).iloc[:, 0])
print(gene_names)

def calc_results_for_fold(X, y, train_index, test_index, clf):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # clf = XGBClassifier()
    clf.fit(X_train, y_train)

    y_true_prob = clf.predict_proba(X_test)[:, 1]
    y_true = clf.predict(X_test)

    acc = np.mean(y_true == y_test)
    # auc_1 = roc_auc_score(y_test, y_true_prob)
    # auc_2 = roc_auc_score(y_test, y_true)

    return acc


clf1 = XGBClassifier()
n_splits = 7
kf = KFold(n_splits=n_splits, shuffle=True)
gene_importance_dict = {}
for col in range(X_.shape[1]):
    print("===========", gene_names[col], "============")
    fold_acc = []
    for i, (train_index, test_index) in enumerate(kf.split(X_)):
        X_f = np.reshape(X_[:, col], (X_.shape[0], 1))
        acc_f = calc_results_for_fold(X_f, y_, train_index, test_index, clf1)
        fold_acc.append(acc_f)
        print("full: ", acc_f)
    fold_acc = np.array(fold_acc)
    mean_acc = np.mean(fold_acc)
    gene_importance_dict[gene_names[col]] = mean_acc
    print("__________________")
    print("mean acc: ", mean_acc)

sorted_genes_dict = {k: v for k, v in sorted(gene_importance_dict.items(), key=lambda item: item[1])}
sorted_genes_list = [[k, v] for k, v in sorted(gene_importance_dict.items(), key=lambda item: item[1])]
with open("gene_importance_4_study_prediction.json", "w") as fp:
    json.dump(sorted_genes_dict, fp)
