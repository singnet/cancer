import pandas as pd
import os
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, recall_score
from funs_common_balanced import get_balanced_split
from funs_common import *
from collections import defaultdict
import sys

output = open("08_results.txt", "w")


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

def calc_results_for_test_study(full_dataset, test_study, clf):
    all_train_studies = list(set(full_dataset['study']))
    all_train_studies.remove(test_study)

    (X_test, y_test) = prepare_dataset(full_dataset, test_study)
    (X_train, y_train) = prepare_datasets(full_dataset, all_train_studies)
    return calc_results_simple(X_train, X_test, y_train, y_test, clf)

def print_results_for_field(dataset, field):
    dataset = drop_na(dataset, field)
    notrea_dataset = drop_trea(dataset)

    all_studies = list(set(dataset['study']))

    print_order = ["full_xgboost", "notrea_xgboost"]
    max_len_order = max(map(len, print_order))

    for test_study in sorted(all_studies):
        print("==> %s study to test:" % field, test_study)

        rez = defaultdict(list)

        rez["full_xgboost"] = calc_results_for_test_study(dataset, test_study, XGBClassifier())

        rez["notrea_xgboost"] = calc_results_for_test_study(notrea_dataset, test_study, XGBClassifier())

        for order in print_order:
            output.write(str(field) + " " +  str(order) + " " * (max_len_order - len(order)) + ": " + str(list_to_4g_str(rez[order])))
            print(field, order, " " * (max_len_order - len(order)), ": ", list_to_4g_str(rez[order]))

# path = "/mnt/fileserver/shared/datasets/biodata/MetaGX/merged/"
path = "/home/daddywesker/bc_data_processor/MetaGX/merged/"
datasets = [ "noNormBatchedMergedAnd5k.csv", "noNormBatchedMergedAnd10k.csv", "noNormMergedAnd5k.csv", "noNormMergedAnd10k.csv",
            "rlNormBatchedMergedAnd5k.csv", "rlNormBatchedMergedAnd10k.csv", "rlNormMergedAnd5k.csv", "rlNormMergedAnd10k.csv",
            "noNormBatchedMergedOr5k.csv", "noNormBatchedMergedOr10k.csv", "noNormMergedOr5k.csv", "noNormMergedOr10k.csv",
            "rlNormBatchedMergedOr5k.csv", "rlNormBatchedMergedOr10k.csv", "rlNormMergedOr5k.csv", "rlNormMergedOr10k.csv"]




print_order = ["full", "full_notrea"]
for dataset in datasets:
    metagx_dataset = read_metagx_dataset_with_stuff(path+dataset, ['study', 'sample_name'])
    output.write(str(dataset) + "\n")
    print_results_for_field(metagx_dataset, "posOutcome")

output.close()
