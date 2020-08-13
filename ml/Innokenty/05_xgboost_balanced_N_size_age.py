import pandas as pd
import os
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, recall_score
from funs_common_balanced import get_balanced_split
from funs_common import *
from collections import defaultdict
import sys

def calc_results_simple(X_train, X_test, y_train, y_test, clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = np.mean(y_test == y_pred)

    y_pred_prob = clf.predict_proba(X_test)[:, 1]
    assert clf.classes_[1] == 1
    recall_0 = recall_score(y_test, y_pred, pos_label=0)
    recall_1 = recall_score(y_test, y_pred, pos_label=1)
    auc = roc_auc_score(y_test, y_pred_prob)

    return acc, recall_0, recall_1, auc

output = open("05_results.txt", "w")

def print_results_for_field(dataset, field):
    dataset = drop_na(dataset, field)

    notrea_dataset = drop_trea(dataset)

    print_order = ["full_xgboost", "full_xgboost_notrea"]
    max_len_order = max(map(len, print_order))

    rez = defaultdict(list)
    for i in range(10):
        print("run ", i)
        ds = get_balanced_split(dataset)
        ds_notrea = get_balanced_split(notrea_dataset)
        rez["full_xgboost"].append(calc_results_simple(*ds, XGBClassifier()))
        rez["full_xgboost_notrea"].append(calc_results_simple(*ds_notrea, XGBClassifier()))

        for order in print_order:
            print(order, " " * (max_len_order - len(order)), ": ", list_to_4g_str(rez[order][-1]))
        print("")
        sys.stdout.flush()

    for order in print_order:
        output.write("==> " + str(order) + " " * (max_len_order - len(order)) + ": " + str(list2d_to_4g_str_pm(rez[order])))
        print("==> ", order, " " * (max_len_order - len(order)), ": ", list2d_to_4g_str_pm(rez[order]))



# path = "/mnt/fileserver/shared/datasets/biodata/MetaGX/merged/"
path = "/home/daddywesker/bc_data_processor/MetaGX/merged/"
datasets = [ "noNormBatchedMergedAnd5k.csv", "noNormBatchedMergedAnd10k.csv", "noNormMergedAnd5k.csv", "noNormMergedAnd10k.csv",
            "rlNormBatchedMergedAnd5k.csv", "rlNormBatchedMergedAnd10k.csv", "rlNormMergedAnd5k.csv", "rlNormMergedAnd10k.csv",
            "noNormBatchedMergedOr5k.csv", "noNormBatchedMergedOr10k.csv", "noNormMergedOr5k.csv", "noNormMergedOr10k.csv",
            "rlNormBatchedMergedOr5k.csv", "rlNormBatchedMergedOr10k.csv", "rlNormMergedOr5k.csv", "rlNormMergedOr10k.csv"]


for dataset in datasets:
    metagx_dataset = read_metagx_dataset_with_stuff(path+dataset, ['study', 'sample_name', 'tumor_size', 'N', 'age_at_initial_pathologic_diagnosis', 'grade'])
    print(metagx_dataset.columns)
    output.write(str(metagx_dataset.columns) + "\n")

    print("==> posOutcome "+str(dataset))
    output.write("==> posOutcome "+str(dataset) + "\n")
    print_results_for_field(metagx_dataset, "posOutcome")
    output.write("" + "\n")
    output.write("" + "\n")
    print("")
    print("")

output.close()
