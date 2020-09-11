import pandas as pd
import os
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score,recall_score
from funs_balance import random_upsample_balance
from funs_common_balanced import get_balanced_split
from funs_common import *
from collections import defaultdict
import sys


def read_coincide_types_dataset(path="./"):
    dataset = pd.read_csv(os.path.join(path, 'metaGXcovarTable_withpam50.csv'))[["pam50_batch", "pam50", "sample_name", "study", 'tumor_size', 'N', 'age_at_initial_pathologic_diagnosis', 'grade']]
    treat_dataset = read_treat_dataset(path)
    return pd.merge(treat_dataset, dataset, left_on="sample_name", right_on="sample_name")


def read_pam_types_cat_dataset(path, keep_cols, batched=False):
    coincide_types_dataset = read_coincide_types_dataset(path)
    print(list(coincide_types_dataset))
    # keep = coincide_types_dataset[["sample_name", "study",'chemo','hormone', 'posOutcome']]
    keep = coincide_types_dataset[keep_cols]
    if batched:
        return pd.concat([keep, pd.get_dummies(coincide_types_dataset["pam50_batch"])], axis=1)
    else:
        return pd.concat([keep, pd.get_dummies(coincide_types_dataset["pam50"])], axis=1)


def read_treat_dataset(path="./"):
    return pd.read_csv(os.path.join(path, 'mldata_new.csv'))


def calc_results_simple(X_train, X_test, y_train, y_test, clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = np.mean(y_test == y_pred)
    recall_0 = recall_score(y_test, y_pred, pos_label=0)
    recall_1 = recall_score(y_test, y_pred, pos_label=1)
    assert clf.classes_[1] == 1

    y_pred_prob = clf.predict_proba(X_test)[:, 1]
    if (np.sum(y_test) == len(y_test)):
        auc = 0
    else:
        auc = roc_auc_score(y_test, y_pred_prob)

    return acc, recall_0, recall_1, auc

output = open("39_results.txt", "w")

def print_results_for_field(dataset, field):
    dataset = drop_na(dataset, field)
    dataset_notrea_dataset = drop_trea(dataset)

    print_order = ["full_xgboost", "notrea_xgboost"]
    max_len_order = max(map(len, print_order))
    rez = defaultdict(list)
    for i in range(20):
        ds = get_balanced_split(dataset, field)
        if(len(ds) == 0):
            continue
        rez["full_xgboost"].append(calc_results_simple(*ds, XGBClassifier()))

        ds = get_balanced_split(dataset_notrea_dataset, field)
        if (len(ds) == 0):
            continue
        rez["notrea_xgboost"].append(calc_results_simple(*ds, XGBClassifier()))

        for order in print_order:
            print(order, " " * (max_len_order - len(order)), ": ", list_to_4g_str(rez[order][-1]))
        print("")

    for order in print_order:
        print("==> ", order, " " * (max_len_order - len(order)), ": ", list2d_to_4g_str_pm(rez[order]))
        output.write("==> " + str(order) + " " * (max_len_order - len(order)) + ": " + str(
            list_to_4g_str(np.mean(rez[order], axis=0))))
        output.write("\n")


path = "/mnt/fileserver/shared/datasets/biodata/MetaGX/"
# path = "/home/daddywesker/bc_data_processor/MetaGX/"
datasets = ["noNormBatchedOrFull.csv", "noNormOrFull.csv", "rlNormBatchedOrFull.csv", "rlNormOrFull.csv"]


print_order = ["full", "full_notrea"]
for dataset in datasets:
    metagx_dataset = read_metagx_dataset(path + "merged/" + dataset)
    covar_table = pd.read_csv(os.path.join(path, 'metaGXcovarTable_withpam50.csv'))[
        ['sample_name', 'channel_count', 'study', 'tumor_size', 'N', 'age_at_initial_pathologic_diagnosis', 'grade']]
    metagx_dataset = pd.merge(metagx_dataset, covar_table, left_on="sample_name", right_on="sample_name")
    print("==> posOutcome " + str(dataset))
    output.write("==> posOutcome " + str(dataset) + "\n")
    output.write("channel_count == 1\n")
    print_results_for_field(metagx_dataset[metagx_dataset['channel_count'] == 1].drop(columns=['channel_count']), "posOutcome")
    output.write("channel_count == 2\n")
    print_results_for_field(metagx_dataset[metagx_dataset['channel_count'] == 2].drop(columns=['channel_count']),
                            "posOutcome")
    output.write("No split \n")
    print_results_for_field(metagx_dataset, "posOutcome")
    output.write("\n")
    output.write("\n")

output.close()
