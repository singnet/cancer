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
from funs_common import read_alltreat_dataset, read_combat_dataset, prepare_full_dataset, list2d_to_4g_str_pm, drop_trea
import sys
import itertools    


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


def print_results(dataset, set1, set2):
    
    
    bias_acc = max(len(set1), len(set2)) / (len(set1) + len(set2))
    print(len(set1), len(set2))
    print(f"bias_accuracy: {bias_acc : .3f}")
    X_set1, _ = prepare_full_dataset(dataset.loc[dataset['patient_ID'].isin(set1)])
    X_set2, _ = prepare_full_dataset(dataset.loc[dataset['patient_ID'].isin(set2)])
    
    y_set1 = np.zeros(len(set1))
    y_set2 = np.ones(len(set2))
    
    X_full = np.concatenate([X_set1, X_set2])
    y_full = np.concatenate([y_set1, y_set2])
    
    
    kf = RepeatedStratifiedKFold(n_splits=5, n_repeats=1)
    rez = []
    for i, (train_index, test_index) in enumerate(kf.split(X_full, y_full)):
        X_train, X_test = X_full[train_index], X_full[test_index]
        y_train, y_test = y_full[train_index], y_full[test_index]
        clf = XGBClassifier()
        rez.append(calc_results_simple(X_train, X_test, y_train, y_test, clf))
        clf.fit(X_train,y_train)
        print(np.argsort(-1 * clf.feature_importances_)[:20])
        print(np.sort(-1 * clf.feature_importances_)[:20])
        
        for i in np.argsort(-1 * clf.feature_importances_)[:20]:
            print(list(dataset.drop(columns=['study', 'patient_ID','pCR', 'RFS', 'DFS', 'posOutcome']))[i])
        
    print(list2d_to_4g_str_pm(rez))
                
atreat_dataset = read_alltreat_dataset()

combat_dataset = read_combat_dataset()
notrea_dataset = drop_trea(combat_dataset)


'''
#Variant 1  study_20194_GPL96_all-bmc15 protocol 1 vs protocol 5

set1 = atreat_dataset.loc[(atreat_dataset['study'] == "study_20194_GPL96_all-bmc15") & (atreat_dataset['treatment_protocol_number'] == '1')]['patient_ID']
set2 = atreat_dataset.loc[(atreat_dataset['study'] == "study_20194_GPL96_all-bmc15") & (atreat_dataset['treatment_protocol_number'] == '5')]['patient_ID']

print("study_20194_GPL96_all-bmc15 protocol 1  vs protocol 5")
print_results(notrea_dataset, set1, set2)
print("")
#Variant 2  study_9893_GPL5049_all-bmc15 protocol 1 vs protocol 2

set1 = atreat_dataset.loc[(atreat_dataset['study'] == "study_9893_GPL5049_all-bmc15") & (atreat_dataset['treatment_protocol_number'] == '1')]['patient_ID']
set2 = atreat_dataset.loc[(atreat_dataset['study'] == "study_9893_GPL5049_all-bmc15") & (atreat_dataset['treatment_protocol_number'] == '2')]['patient_ID']

print("study_9893_GPL5049_all-bmc15 protocol 1 vs protocol 2")
print_results(notrea_dataset, set1, set2)
print("")
'''

#Variant 3 study_25065_GPL96_MDACC-bmc15 protocol 1 vs protocol 2

set1 = atreat_dataset.loc[(atreat_dataset['study'] == "study_25065_GPL96_MDACC-bmc15") & (atreat_dataset['treatment_protocol_number'] == '1')]['patient_ID']
set2 = atreat_dataset.loc[(atreat_dataset['study'] == "study_25065_GPL96_MDACC-bmc15") & (atreat_dataset['treatment_protocol_number'] == '2')]['patient_ID']

print("study_25065_GPL96_MDACC-bmc15 protocol 1 vs protocol 2")
print_results(notrea_dataset, set1, set2)
