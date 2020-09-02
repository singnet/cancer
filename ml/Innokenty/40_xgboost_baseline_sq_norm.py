import pandas as pd
import os
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, recall_score
from sklearn.model_selection import StratifiedKFold
from collections import defaultdict
from funs_common import *
import sys

def drop_na(d, field):
    return d.loc[d[field].notna()]


def calc_results_simple(X_train, X_test, y_train, y_test, clf):
    clf.fit(X_train,y_train)
    y_pred  = clf.predict(X_test)
    acc = np.mean(y_test == y_pred)
    
    y_pred_prob = clf.predict_proba(X_test)[:,1]
    assert clf.classes_[1] == 1
    recall_0 =  recall_score(y_test, y_pred, pos_label=0)
    recall_1 =  recall_score(y_test, y_pred, pos_label=1)    
    auc = roc_auc_score(y_test, y_pred_prob)
    
    return acc, recall_0, recall_1, auc

def calc_results_simple_fold(X, y, train_index, test_index, clf):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    return calc_results_simple(X_train, X_test, y_train, y_test, clf)

output = open("results.txt", "w")

def print_results_for_field(dataset, field):
    
    dataset = drop_na(dataset, field)
    notrea_dataset = drop_trea(dataset)

    X_notrea,_     = prepare_full_dataset(notrea_dataset, y_field = field)
    X_full, y_full = prepare_full_dataset(dataset, y_field = field)


    kf = StratifiedKFold(n_splits=5, shuffle=True)
    print_order = ["full_xgboost", "notrea_xgboost"]
    max_len_order = max(map(len,print_order))
    
    rez = defaultdict(list)
    for i, (train_index, test_index) in enumerate(kf.split(X_full, y_full)):
        print("split ", i)
        rez["full_xgboost"].append(calc_results_simple_fold(X_full, y_full, train_index, test_index, XGBClassifier()))
        rez["notrea_xgboost"].append(calc_results_simple_fold(X_notrea, y_full, train_index, test_index, XGBClassifier()))
        
        for order in print_order:
            print(order, " "*(max_len_order - len(order)), ": ", list_to_4g_str(rez[order][-1]))
        print ("")
        sys.stdout.flush()
        
    for order in print_order:
        output.write("==> " + str(order) + " "*(max_len_order - len(order)) + ": " + str(list_to_4g_str(np.mean(rez[order], axis=0))))
        output.write("\n")
        print("==> ", order, " "*(max_len_order - len(order)), ": ", list_to_4g_str(np.mean(rez[order], axis=0)))
                

# path = "/mnt/fileserver/shared/datasets/biodata/MetaGX/merged/"
path = "/home/daddywesker/bc_data_processor/MetaGX/merged/"
datasets = [ "sqrecurNormBatchedAnd5k.csv", "sqdeathNormBatchedOrFull.csv", "sqrecurNormBatchedOrFull.csv" ]


for dataset in datasets:
    metagx_dataset = read_metagx_dataset(path+dataset)
    print(metagx_dataset.columns)

    print("==> posOutcome "+str(dataset))
    output.write("==> posOutcome " + str(dataset) + "\n")
    print_results_for_field(metagx_dataset, "posOutcome")
    print("")
    print("")
    output.write("\n")
    output.write("\n")
output.close()
