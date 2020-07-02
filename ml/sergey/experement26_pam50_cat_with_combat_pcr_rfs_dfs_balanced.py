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
    

def get_balanced_split_for_study(full_dataset, study, y_field):
    X,y = prepare_dataset(full_dataset, study, y_field = y_field)
    
        
    kf = RepeatedStratifiedKFold(n_splits=5)
    (train_index, test_index) = next(kf.split(X,y))
    
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    X_train, y_train = random_upsample_balance(X_train, y_train)
    X_test, y_test   = random_upsample_balance(X_test, y_test)
    
    return X_train, X_test, y_train, y_test

def get_balanced_split(full_dataset, y_field):
    
    all_X_train, all_X_test, all_y_train, all_y_test = [],[],[],[]
            
    for study in list(set(full_dataset['study'])):
        X_train, X_test, y_train, y_test = get_balanced_split_for_study(full_dataset, study, y_field)
        all_X_train.append(X_train)
        all_X_test.append(X_test)
        all_y_train.append(y_train)
        all_y_test.append(y_test)
    return np.concatenate(all_X_train),  np.concatenate(all_X_test), np.concatenate(all_y_train), np.concatenate(all_y_test)
                                                        


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

def print_results_for_field(dataset, field):
    
    dataset = drop_na(dataset, field)
    dataset_notrea_dataset =drop_trea(dataset)
    
    print_order = ["full_xgboost", "notrea_xgboost"]
    max_len_order = max(map(len,print_order))
    rez = defaultdict(list)
    for i in range(10):
        ds = get_balanced_split(dataset, field)
        rez["full_xgboost"].append(calc_results_simple(*ds, XGBClassifier()))
        
        ds = get_balanced_split(dataset_notrea_dataset, field)
        rez["notrea_xgboost"].append(calc_results_simple(*ds, XGBClassifier()))
        
        for order in print_order:
            print(order, " "*(max_len_order - len(order)), ": ", list_to_4g_str(rez[order][-1]))
        print ("")
    
    for order in print_order:
        print("==> ", order, " "*(max_len_order - len(order)), ": ", list2d_to_4g_str_pm(rez[order]))
                

pam_types_cat_dataset = read_pam_types_cat_dataset()
combat_dataset = read_combat_dataset()
dataset = pd.merge(pam_types_cat_dataset, combat_dataset)


print("==> pCR")
print_results_for_field(dataset, "pCR")
print("")
print("")

print("==> RFS")
print_results_for_field(dataset, "RFS")
print("")
print("")

print("==> DFS")
print_results_for_field(dataset, "DFS")
print("")
print("")

print("==> posOutcome")
print_results_for_field(dataset, "posOutcome")
print("")
print("")


