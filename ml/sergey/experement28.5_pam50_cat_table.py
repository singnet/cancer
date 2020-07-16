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
from funs_common import *
import sys
from sklearn.preprocessing import OneHotEncoder
import itertools


def read_coincide_types_dataset(path="./"):
    dataset = pd.read_csv(os.path.join(path, 'coincideTypes.csv'))
    treat_dataset = read_treat_dataset()
    return pd.merge(treat_dataset, dataset)

def drop_na(d, field):
    return d.loc[d[field].notna()]

def read_pam_types_cat_dataset():    
    coincide_types_dataset = read_coincide_types_dataset()
    print(list(coincide_types_dataset))
    keep = coincide_types_dataset[["patient_ID", "study", 'pCR', 'RFS', 'DFS','radio','surgery','chemo','hormone', 'posOutcome']]
    return pd.concat([keep, pd.get_dummies(coincide_types_dataset["pam_name"])], axis=1) 

def read_pam_types_num_dataset():    
    coincide_types_dataset = read_coincide_types_dataset()
    print(list(coincide_types_dataset))
    return coincide_types_dataset[["patient_ID", "study", 'pCR', 'RFS', 'DFS','radio','surgery','chemo','hormone', 'posOutcome', 'pam_cat']]


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


def calc_results_with_ys(y_test, y_pred, y_pred_prob):
    acc = np.mean(y_test == y_pred)    
    recall_0 =  recall_score(y_test, y_pred, pos_label=0)
    recall_1 =  recall_score(y_test, y_pred, pos_label=1)    
    auc = roc_auc_score(y_test, y_pred_prob)
    return acc, recall_0, recall_1, auc


def calc_results_simple_fold(X, y, train_index, test_index, clf):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    return calc_results_simple(X_train, X_test, y_train, y_test, clf)

def prepare_full_dataset_septrea(full_dataset, y_field = 'posOutcome'):
    list_trea = ['radio','surgery','chemo','hormone']
    X_notrea = full_dataset.drop(columns=list_trea + ['study', 'patient_ID','pCR', 'RFS', 'DFS', 'posOutcome']).to_numpy()
    X_trea   = full_dataset[list_trea].to_numpy()
    y_posOutcome = full_dataset[y_field].to_numpy()
    return X_trea, X_notrea, y_posOutcome

def get_possible_treatments_combinations(X_trea):
    possible_treats = [list(set(Xt)) for Xt in np.transpose(X_trea)]
    return np.array(list(itertools.product(*possible_treats)))

def calc_results_max_treatments(clf, X_test, y_test, possible_treat):
    Ndiff = 0
    y_maxi = []
    y_prob_maxi = []
    for Xi,yi in zip(X_test, y_test):
        Xir = np.tile(Xi, (possible_treat.shape[0], 1))
        Xir[:, :possible_treat.shape[1]] = possible_treat
        
        y_maxi.append(np.max(clf.predict(Xir)))
        y_prob_maxi.append(np.max(clf.predict_proba(Xir)[:,1]))
    return np.sum(y_maxi), np.sum(y_prob_maxi)
        
def print_results_for_field(dataset, field):
    
    dataset = drop_na(dataset, field)
    
    X_full, y_full = prepare_full_dataset(dataset)
    
    rez = defaultdict(int)
    for xi,yi in zip(X_full, y_full):
        rez[(xi[4], tuple(xi[:4]),yi)] += 1
        
    used = set()
    for pam,treat,yi in sorted(rez.keys()):
        if ((pam,treat) not in used):
            used.add((pam,treat))
            
            out0 = rez[(pam,treat, 0)]
            out1 = rez[(pam,treat, 1)]
            survival_rate = out1 / (out0 + out1)
            
            
            print(f"{pam}, {treat}, {out0+out1:5},   {survival_rate:.3g}")
            
    
    

#full_dataset = read_full_dataset()
pam_types_cat_dataset = read_pam_types_num_dataset()



print("==> pCR")
print_results_for_field(pam_types_cat_dataset, "pCR")
print("")
print("")


print("==> RFS")
print_results_for_field(pam_types_cat_dataset, "RFS")
print("")
print("")


print("==> DFS")
print_results_for_field(pam_types_cat_dataset, "DFS")
print("")
print("")


print("==> posOutcome")
print_results_for_field(pam_types_cat_dataset, "posOutcome")
print("")
print("")

