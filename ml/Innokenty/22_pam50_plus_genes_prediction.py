import pandas as pd
import os
import numpy as np
import random
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score,recall_score
from sklearn.model_selection import  StratifiedKFold
from funs_balance import random_upsample_balance
from funs_common import *
from collections import defaultdict

def read_treat_dataset(path="./"):
    return pd.read_csv(os.path.join(path, 'mldata_new.csv'))

def read_coincide_types_dataset(path="./"):
    dataset = pd.read_csv(os.path.join(path, 'metaGXcovarTable_withpam50.csv'))[["pam50_batch", "pam50", "sample_name", "study", 'tumor_size', 'N', 'age_at_initial_pathologic_diagnosis', 'grade']]
    treat_dataset = read_treat_dataset(path)
    return pd.merge(treat_dataset, dataset, left_on="sample_name", right_on="sample_name")

def read_pam_types_cat_dataset(path, keep_cols, batched=False):
    coincide_types_dataset = read_coincide_types_dataset(path)
    print(list(coincide_types_dataset))
    keep = coincide_types_dataset[keep_cols]
    if batched:
        return pd.concat([keep, pd.get_dummies(coincide_types_dataset["pam50_batch"])], axis=1)
    else:
        return pd.concat([keep, pd.get_dummies(coincide_types_dataset["pam50"])], axis=1)


def get_balanced_study(full_dataset, study):
    X,y = prepare_dataset(full_dataset, study)
    return random_upsample_balance(X, y)

def get_balanced_studies_except_test_study(full_dataset, test_study):
    all_studies = list(set(full_dataset['study']))
    all_Xy = [get_balanced_study(full_dataset, study) for study in all_studies if (study != test_study)]
    all_X, all_y = zip(*all_Xy)
    return np.concatenate(all_X),  np.concatenate(all_y)


def calc_results_simple(X_train, X_test, y_train, y_test, clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = np.mean(y_test == y_pred)
    recall_0 = recall_score(y_test, y_pred, pos_label=0)
    recall_1 = recall_score(y_test, y_pred, pos_label=1)
    assert clf.classes_[1] == 1

    y_pred_prob = clf.predict_proba(X_test)[:, 1]
    if(np.sum(y_test) == len(y_test)):
        auc = 0
    else:
        auc = roc_auc_score(y_test, y_pred_prob)

    return acc, recall_0, recall_1, auc

def calc_results_simple_fold(X, y, train_index, test_index, clf):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    return calc_results_simple(X_train, X_test, y_train, y_test, clf)

# path = "/mnt/fileserver/shared/datasets/biodata/MetaGX/"
path = "/home/daddywesker/bc_data_processor/MetaGX/"
datasets = [ "noNormBatchedMergedAnd5k.csv", "noNormBatchedMergedAnd10k.csv", "noNormMergedAnd5k.csv", "noNormMergedAnd10k.csv",
            "rlNormBatchedMergedAnd5k.csv", "rlNormBatchedMergedAnd10k.csv", "rlNormMergedAnd5k.csv", "rlNormMergedAnd10k.csv",
            "noNormBatchedMergedOr5k.csv", "noNormBatchedMergedOr10k.csv", "noNormMergedOr5k.csv", "noNormMergedOr10k.csv",
            "rlNormBatchedMergedOr5k.csv", "rlNormBatchedMergedOr10k.csv", "rlNormMergedOr5k.csv", "rlNormMergedOr10k.csv"]

output = open("22_results.txt", "w")

print_order = ["full", "full_notrea"]
for dataset in datasets:
    metagx_dataset = read_metagx_dataset(path+"merged/"+dataset, True)
    if(dataset[6:13] == "Batched"):
        pam_types_cat_dataset = read_pam_types_cat_dataset(path, ["sample_name", "study",'chemo','hormone', 'posOutcome', 'alt_sample_name', 'recurrence'], True)
    else:
        pam_types_cat_dataset = read_pam_types_cat_dataset(path, ["sample_name", "study", 'chemo', 'hormone', 'posOutcome', 'alt_sample_name', 'recurrence'])
    pam_types_cat_dataset = pd.merge(metagx_dataset, pam_types_cat_dataset, left_on="sample_name", right_on="sample_name")
    pam_types_cat_dataset = drop_na(pam_types_cat_dataset, "posOutcome")
    pam_types_cat_dataset_notrea_dataset = drop_trea(pam_types_cat_dataset)
    X_full, y_full = prepare_full_dataset(pam_types_cat_dataset)
    X_notrea, _ = prepare_full_dataset(pam_types_cat_dataset_notrea_dataset)

    kf = StratifiedKFold(n_splits=5, shuffle=True)
    max_len_order = max(map(len, print_order))
    print("==> posOutcome "+str(dataset))
    output.write("==> posOutcome "+str(dataset) + "\n")
    rez = defaultdict(list)
    if (len(X_full) < 5):
        print("too little patients")
        output.write("too little patients\n")
        continue
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
