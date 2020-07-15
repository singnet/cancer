import pandas as pd
import os
import numpy as np
import random
import math 
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score,f1_score,recall_score
from sklearn.model_selection import KFold, RepeatedKFold, RepeatedStratifiedKFold, StratifiedKFold
from collections import Counter,defaultdict
from funs_balance import random_upsample_balance
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from funs_common import read_alltreat_dataset, read_combat_dataset, prepare_full_dataset, list2d_to_4g_str_pm, drop_trea, list_to_4g_str, read_pam_types_cat_dataset
import sys
import itertools    
from class_experement32_BiasedXgboost import BiasedXgboost
from class_experement32_DoubleXgboost import DoubleXgboost

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

def calc_results_for_fold(X, y, train_index, test_index, clf):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]    
    return calc_results_simple(X_train, X_test, y_train, y_test, clf)               

def add_one_features(X, f):
    return np.concatenate([np.ones((X.shape[0], 1)) * f, X ], axis=1)

def print_results(dataset, set1, set2):
    X_set1, y_set1 = prepare_full_dataset(dataset.loc[dataset['patient_ID'].isin(set1)])
    X_set2, y_set2 = prepare_full_dataset(dataset.loc[dataset['patient_ID'].isin(set2)])

#    X_set1 = np.random.rand(*X_set1.shape)
#    X_set2 = np.random.rand(*X_set2.shape)


    X_set1_wf = add_one_features(X_set1, 0)
    X_set2_wf = add_one_features(X_set2, 1)
    
    
    X_genes    = np.concatenate([X_set1,    X_set2])
    X_genes_wf = np.concatenate([X_set1_wf, X_set2_wf])
    y_all    = np.concatenate([y_set1,    y_set2])

    Xs = X_genes_wf[:,:1]

    
    kf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10)
    print_order = ["genes", "genes_set", "genes_biased", "genes_double", "study"]
        
    max_len_order = max(map(len,print_order))
        
    rez = defaultdict(list)
    
    for i, (train_index, test_index) in enumerate(kf.split(X_genes, y_all)):
        rez["genes"].append(   calc_results_for_fold(X_genes,    y_all, train_index, test_index, XGBClassifier()))
        rez["genes_set"].append(calc_results_for_fold(X_genes_wf, y_all, train_index, test_index, XGBClassifier()))
        rez["genes_biased"].append(calc_results_for_fold(X_genes_wf, y_all, train_index, test_index, BiasedXgboost()))
        rez["genes_double"].append(calc_results_for_fold(X_genes_wf, y_all, train_index, test_index, DoubleXgboost()))
        rez["study"].append( calc_results_for_fold(Xs       , y_all, train_index, test_index, XGBClassifier()))
        
        
        for order in print_order:
            print(order, " "*(max_len_order - len(order)), ": ", list_to_4g_str(rez[order][-1]))
        print ("")
                                        
        
    for order in print_order:
        print("==> ", order, " "*(max_len_order - len(order)), ": ", list2d_to_4g_str_pm(rez[order]))
                
atreat_dataset = read_alltreat_dataset()

pam_types_cat_dataset = read_pam_types_cat_dataset()
notrea_dataset = drop_trea(pam_types_cat_dataset)


#Variant 1  study_20194_GPL96_all-bmc15 protocol 1 vs protocol 5

set1 = atreat_dataset.loc[(atreat_dataset['study'] == "study_20194_GPL96_all-bmc15") & (atreat_dataset['treatment_protocol_number'] == '1')]['patient_ID']
set2 = atreat_dataset.loc[(atreat_dataset['study'] == "study_20194_GPL96_all-bmc15") & (atreat_dataset['treatment_protocol_number'] == '5')]['patient_ID']

print("==> study_20194_GPL96_all-bmc15 protocol 1  vs protocol 5")
print_results(notrea_dataset, set1, set2)
print("==>")


#Variant 2  study_9893_GPL5049_all-bmc15 protocol 1 vs protocol 2

set1 = atreat_dataset.loc[(atreat_dataset['study'] == "study_9893_GPL5049_all-bmc15") & (atreat_dataset['treatment_protocol_number'] == '1')]['patient_ID']
set2 = atreat_dataset.loc[(atreat_dataset['study'] == "study_9893_GPL5049_all-bmc15") & (atreat_dataset['treatment_protocol_number'] == '2')]['patient_ID']

print("==> study_9893_GPL5049_all-bmc15 protocol 1 vs protocol 2")
print_results(notrea_dataset, set1, set2)
print("==> ")


