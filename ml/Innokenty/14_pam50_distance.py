import pandas as pd
import os
import numpy as np
import random
import math 
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, recall_score
from sklearn.model_selection import StratifiedKFold
from funs_balance import random_upsample_balance
from funs_common_balanced import get_balanced_split
from funs_common import *
from collections import defaultdict
import sys


def merge_pam50_with_other_datasets(path="./"):
    dataset = pd.read_csv(os.path.join(path, 'metaGXcovarTable.csv'))[['sample_name', 'tumor_size', 'N', 'age_at_initial_pathologic_diagnosis', 'grade']]
    pam50_distances = pd.read_csv(os.path.join(path, 'pam50labelProbabilities.csv'))
    treat_dataset = read_treat_dataset(path)
    return pd.merge(pd.merge(treat_dataset, dataset, left_on="sample_name", right_on="sample_name"), pam50_distances, left_on="sample_name", right_on="sample_name")


def read_pam50_distance_dataset(path, keep_cols):
    pam50_distances_dataset = merge_pam50_with_other_datasets(path)
    keep = pam50_distances_dataset[keep_cols]
    return keep


def read_treat_dataset(path="./"):
    return pd.read_csv(os.path.join(path, 'mldata.csv'))


def calc_results_simple(X_train, X_test, y_train, y_test, clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = np.mean(y_test == y_pred)
    recall_0 = recall_score(y_test, y_pred, pos_label=0)
    recall_1 = recall_score(y_test, y_pred, pos_label=1)
    assert clf.classes_[1] == 1

    y_pred_prob = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_prob)

    return acc, recall_0, recall_1, auc

def calc_results_simple_fold(X, y, train_index, test_index, clf):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    return calc_results_simple(X_train, X_test, y_train, y_test, clf)

output = open("14_results.txt", "w")



# path = "/mnt/fileserver/shared/datasets/biodata/MetaGX/"
path = "/home/daddywesker/bc_data_processor/MetaGX/"
datasets = [ "noNormBatchedMergedAnd5k.csv", "noNormBatchedMergedAnd10k.csv", "noNormMergedAnd5k.csv", "noNormMergedAnd10k.csv",
            "rlNormBatchedMergedAnd5k.csv", "rlNormBatchedMergedAnd10k.csv", "rlNormMergedAnd5k.csv", "rlNormMergedAnd10k.csv",
            "noNormBatchedMergedOr5k.csv", "noNormBatchedMergedOr10k.csv", "noNormMergedOr5k.csv", "noNormMergedOr10k.csv",
            "rlNormBatchedMergedOr5k.csv", "rlNormBatchedMergedOr10k.csv", "rlNormMergedOr5k.csv", "rlNormMergedOr10k.csv"]


print_order = ["full", "full_notrea"]
for dataset in datasets:
    if(dataset[6:13] == "Batched"):
        pam_types_cat_dataset = read_pam50_distance_dataset(path, ['sample_name', 'alt_sample_name', 'recurrence', 'study', 'chemo','hormone', 'posOutcome',  'Basal', 'Her2', 'LumA',	'LumB',	'Normal'])
    else:
        pam_types_cat_dataset = read_pam50_distance_dataset(path, ['sample_name', 'alt_sample_name', 'recurrence', 'study', 'chemo', 'hormone', 'posOutcome',
                                                                   'Basal_batched', 'Her2_batched', 'LumA_batched', 'LumB_batched', 'Normal_batched'])
    metagx_dataset = read_metagx_dataset(path+"merged/"+dataset)
    pam_types_cat_dataset = pd.merge(metagx_dataset[["sample_name"]], pam_types_cat_dataset, left_on="sample_name", right_on="sample_name")
    pam_types_cat_dataset = drop_na(pam_types_cat_dataset, "posOutcome")
    pam_types_cat_dataset_notrea_dataset = drop_trea(pam_types_cat_dataset)
    X_full, y_full = prepare_full_dataset(pam_types_cat_dataset)
    X_notrea, _ = prepare_full_dataset(pam_types_cat_dataset_notrea_dataset)

    kf = StratifiedKFold(n_splits=5, shuffle=True)
    max_len_order = max(map(len, print_order))
    print("==> posOutcome "+str(dataset))
    output.write("==> posOutcome "+str(dataset) + "\n")
    rez = defaultdict(list)
    for i, (train_index, test_index) in enumerate(kf.split(X_full, y_full)):
        print("split ", i)
        rez["full"].append(calc_results_simple_fold(X_full, y_full, train_index, test_index, XGBClassifier()))
        rez["full_notrea"].append(
            calc_results_simple_fold(X_notrea, y_full, train_index, test_index, XGBClassifier()))

        for order in print_order:
            print(order, " " * (max_len_order - len(order)), ": ", list_to_4g_str(rez[order][-1]))
        print("")

    for order in print_order:
        output.write("==> " + str(order) + " " * (max_len_order - len(order)) + ": " + str(list_to_4g_str(np.mean(rez[order], axis=0))))
        output.write("\n")
        print("==> ", order, " " * (max_len_order - len(order)), ": ", list_to_4g_str(np.mean(rez[order], axis=0)))

output.close()

output.close()
