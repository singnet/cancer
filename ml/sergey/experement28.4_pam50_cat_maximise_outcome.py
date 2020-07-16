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
    
    X_trea, X_notrea, y_full = prepare_full_dataset_septrea(dataset)
    
    possible_treat = get_possible_treatments_combinations(X_trea)
    
    X_full =np.concatenate([X_trea, X_notrea], axis=1)
    kf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10)
    
    
    rez = []
    for i, (train_index, test_index) in enumerate(kf.split(X_full, y_full)):        
        X_train, X_test = X_full[train_index], X_full[test_index]
        y_train, y_test = y_full[train_index], y_full[test_index]
        clf =  XGBClassifier()
        clf.fit(X_train,y_train)
        assert clf.classes_[1] == 1
        
        y_sum = np.sum(clf.predict(X_test))
        y_prob_sum = np.sum(clf.predict_proba(X_test)[:,1])        
        y_maxi_sum, y_prob_maxi_sum = calc_results_max_treatments(clf, X_test, y_test, possible_treat)
        N = len(y_test)
        rez.append([y_sum / N, y_maxi_sum / N, y_prob_sum / N, y_prob_maxi_sum / N])
        print("fold ", i, list_to_4g_str(rez[-1]))
    print("==> ", list_to_4g_str(np.mean(rez,axis=0)))
                
    

#full_dataset = read_full_dataset()
pam_types_cat_dataset = read_pam_types_cat_dataset()



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


