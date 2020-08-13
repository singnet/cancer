import pandas as pd
import os
import numpy as np
import random
import math 
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, f1_score, recall_score
from sklearn.model_selection import KFold, RepeatedKFold, RepeatedStratifiedKFold, StratifiedKFold
from collections import Counter
from funs_balance import random_upsample_balance
from funs_common_balanced import get_balanced_split
from funs_common import *
from collections import Counter,defaultdict
from sklearn import svm
from sklearn.linear_model import LogisticRegression
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
    return pd.read_csv(os.path.join(path, 'mldata_new.csv'))

def prepare_datasets(full_dataset, studies):
    Xys = [prepare_dataset(full_dataset, study ) for study in studies] 
    Xs, ys = zip(*Xys)
    return np.concatenate(Xs), np.concatenate(ys)

def _get_shuffled_study_list(full_dataset):
    all_studies = list(set(full_dataset['study']))
    random.shuffle(all_studies)
    return all_studies

def study_fold(full_dataset, shuffled_study_list, nval):
    return prepare_datasets(full_dataset, shuffled_study_list[nval:]), prepare_datasets(full_dataset, shuffled_study_list[:nval])

def calc_results_for_fold(X, y, train_index, test_index, clf):        
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    X_train, y_train = random_upsample_balance(X_train, y_train)
    X_test, y_test   = random_upsample_balance(X_test, y_test)
    
    clf.fit(X_train,y_train)    
    y_pred  = clf.predict(X_test)
    
    acc = np.mean(y_test == y_pred)
    recall_0 =  recall_score(y_test, y_pred, pos_label=0)
    recall_1 =  recall_score(y_test, y_pred, pos_label=1)
    
    return acc, recall_0, recall_1

def count_to_str(y):
    c = Counter(y)
    return "count_01=%i/%i"%(c[0], c[1])

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
    auc = roc_auc_score(y_test, y_pred_prob)

    return acc, recall_0, recall_1, auc

def calc_results_simple_fold(X, y, train_index, test_index, clf):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    return calc_results_simple(X_train, X_test, y_train, y_test, clf)

def calc_results_for_test_study(full_dataset, test_study, clf):
    all_train_studies = list(set(full_dataset['study']))
    all_train_studies.remove(test_study)

    (X_test, y_test) = prepare_dataset(full_dataset, test_study)
    (X_train, y_train) = prepare_datasets(full_dataset, all_train_studies)
    return calc_results_simple(X_train, X_test, y_train, y_test, clf)

output = open("19_results.txt", "w")

def print_results_for_field(dataset, field):
    dataset = drop_na(dataset, field)
    notrea_dataset = drop_trea(dataset)

    all_studies = list(set(dataset['study']))

    print_order = ["full_xgboost", "notrea_xgboost"]
    max_len_order = max(map(len, print_order))

    for test_study in sorted(all_studies):
        print("==> %s study to test:" % field, test_study)
        output.write("==> %s study to test:" % field + str(test_study)+"\n")

        rez = defaultdict(list)

        rez["full_xgboost"] = calc_results_for_test_study(dataset, test_study, XGBClassifier())

        rez["notrea_xgboost"] = calc_results_for_test_study(notrea_dataset, test_study, XGBClassifier())

        for order in print_order:
            output.write(str(field) + " " +  str(order) + " " * (max_len_order - len(order)) + ": " + str(list_to_4g_str(rez[order]))+"\n")
            print(field, order, " " * (max_len_order - len(order)), ": ", list_to_4g_str(rez[order]))

# path = "/mnt/fileserver/shared/datasets/biodata/MetaGX/"
path = "/home/daddywesker/bc_data_processor/MetaGX/"
datasets = [ "noNormBatchedMergedAnd5k.csv", "noNormBatchedMergedAnd10k.csv", "noNormMergedAnd5k.csv", "noNormMergedAnd10k.csv",
            "rlNormBatchedMergedAnd5k.csv", "rlNormBatchedMergedAnd10k.csv", "rlNormMergedAnd5k.csv", "rlNormMergedAnd10k.csv",
            "noNormBatchedMergedOr5k.csv", "noNormBatchedMergedOr10k.csv", "noNormMergedOr5k.csv", "noNormMergedOr10k.csv",
            "rlNormBatchedMergedOr5k.csv", "rlNormBatchedMergedOr10k.csv", "rlNormMergedOr5k.csv", "rlNormMergedOr10k.csv"]


print_order = ["full", "full_notrea"]
for dataset in datasets:
    output.write("dataset: "+str(dataset)+"\n")
    if(dataset[6:13] == "Batched"):
        pam_types_cat_dataset = read_pam50_distance_dataset(path, ['sample_name', 'alt_sample_name', 'recurrence', 'study', 'chemo','hormone', 'posOutcome', 'tumor_size', 'N', 'age_at_initial_pathologic_diagnosis', 'grade',  'Basal', 'Her2', 'LumA',	'LumB',	'Normal'])
    else:
        pam_types_cat_dataset = read_pam50_distance_dataset(path, ['sample_name', 'alt_sample_name', 'recurrence', 'study', 'chemo', 'hormone', 'posOutcome', 'tumor_size', 'N', 'age_at_initial_pathologic_diagnosis', 'grade',
                                                                   'Basal_batched', 'Her2_batched', 'LumA_batched', 'LumB_batched', 'Normal_batched'])
    metagx_dataset = read_metagx_dataset(path + "merged/" + dataset)[["sample_name"]]
    pam_types_cat_dataset = pd.merge(pam_types_cat_dataset, metagx_dataset)
    print_results_for_field(pam_types_cat_dataset, "posOutcome")
    output.write("\n")

output.close()

output.close()
