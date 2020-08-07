import os
import numpy as np
import random
import math
from xgboost import XGBClassifier
from xgboost import plot_importance
from matplotlib import pyplot
from sklearn.model_selection import KFold
import pandas as pd
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


clf1 = XGBClassifier(tree_method='gpu_hist')
n_splits = 2
kf = KFold(n_splits=n_splits, shuffle=True)
gene_importance_dict = {}

res_gene_imp = [[], []]
for i, (train_index, test_index) in enumerate(kf.split(X_)):
    acc_f = calc_results_for_fold(X_, y_, train_index, test_index, clf1)
    print("full: ", acc_f)
    print(clf1.feature_importances_)
    res_gene_imp[i] = list(zip(gene_names, clf1.feature_importances_))
    # plot_importance(clf1)
    # pyplot.show()

sorted_gene_imp = sorted(res_gene_imp[0], key=lambda x: x[1], reverse=True)
print(sorted_gene_imp[:20])

features = metagx_result_data.drop(columns=cols_to_drop)
important_genes = [x[0] for x in sorted_gene_imp]
features_to_drop = features.loc[:, important_genes[:2000]]
less_imp_features = features.drop(columns=features_to_drop).to_numpy()
X_ = less_imp_features

print("________________________________________")

clf2 = XGBClassifier(tree_method='gpu_hist')
n_splits = 2
kf = KFold(n_splits=n_splits, shuffle=True)
gene_importance_dict = {}

res_gene_imp = []
for i, (train_index, test_index) in enumerate(kf.split(X_)):
    acc_f = calc_results_for_fold(X_, y_, train_index, test_index, clf2)
    print("full: ", acc_f)
# with open("gene_importance_4_study_prediction.json", "w") as fp:
#     json.dump(sorted_genes_dict, fp)
