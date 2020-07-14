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


def print_results_get_mean_acc(ds1, ds2):
    l1 = len(ds1)
    l2 = len(ds2)
    bias_acc = max(l1,l2) / (l1 + l2)
    print(f"bias_accuracy: {bias_acc : .3f}")
    X_set1, _ = prepare_full_dataset(ds1)
    X_set2, _ = prepare_full_dataset(ds2)
    
    y_set1 = np.zeros(l1)
    y_set2 = np.ones(l2)
    
    X_full = np.concatenate([X_set1, X_set2])
    y_full = np.concatenate([y_set1, y_set2])
    
    
    kf = RepeatedStratifiedKFold(n_splits=5, n_repeats=1)
    rez = []
    for i, (train_index, test_index) in enumerate(kf.split(X_full, y_full)):
        X_train, X_test = X_full[train_index], X_full[test_index]
        y_train, y_test = y_full[train_index], y_full[test_index]
        rez.append(calc_results_simple(X_train, X_test, y_train, y_test, XGBClassifier()))
        
    print(list2d_to_4g_str_pm(rez))
    
    return np.mean(np.array(rez)[:,0])
                

dataset = read_combat_dataset()
dataset = drop_trea(dataset)


all_studies = list(set(dataset['study']))

all_rez = []
for s1,s2 in itertools.combinations(sorted(all_studies), 2):
    ds1 = dataset.loc[dataset['study'] == s1]
    ds2 = dataset.loc[dataset['study'] == s2]
    print("studies: ", s1, s2, len(ds1), len(ds2))
    rez = print_results_get_mean_acc(ds1, ds2)
    all_rez.append((rez, s1, s2))
    
print("\n--------------------------------------\n")
for rez,s1,s2 in sorted(all_rez):
    if (rez < 0.99):
        print(rez, s1, s2)
    
