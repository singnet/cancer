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

def calc_results_smart_bias(X_train, X_test, y_train, y_test):
    y_split = defaultdict(list)
    for x,y in zip(X_train[:,0], y_train):
        y_split[x].append(y)
        
    most_common, prob_1 = {},{} 
    for x, ys_for_x in y_split.items():
        c = Counter(ys_for_x)
        prob_1[x]      = c[1] / (c[0] + c[1])
        most_common[x] = c.most_common(1)[0][0]
        
    y_pred = [most_common[x]   for x in X_test[:,0]]
    y_prob = [prob_1[x]        for x in X_test[:,0]]
    
    acc = np.mean(y_test == y_pred)
    recall_0 =  recall_score(y_test, y_pred, pos_label=0)
    recall_1 =  recall_score(y_test, y_pred, pos_label=1)
    auc = roc_auc_score(y_test, y_prob)
    return acc, recall_0, recall_1, auc
    

def print_results_for_field(dataset, field):
    dataset = drop_na(dataset, field)
    
    X_full = dataset["study"].to_numpy().reshape(-1,1)
    y_full = dataset[field].to_numpy()
    print("==> simple    ", list_to_4g_str(calc_results_smart_bias(X_full, X_full, y_full, y_full)))
    
    
    
    
    kf = StratifiedKFold(n_splits=5, shuffle=True)
    
    all_rez = []
    for i, (train_index, test_index) in enumerate(kf.split(X_full, y_full)):
        X_train, X_test = X_full[train_index], X_full[test_index]
        y_train, y_test = y_full[train_index], y_full[test_index]
                
        all_rez.append(calc_results_smart_bias(X_train, X_test, y_train, y_test))
        
    print("==> avg_split5", list_to_4g_str(np.mean(all_rez, axis=0)))
    
treat_dataset = read_treat_dataset()

print("==> pCR")
print_results_for_field(treat_dataset, "pCR")
print("")
print("")

print("==> RFS")
print_results_for_field(treat_dataset, "RFS")
print("")
print("")

print("==> DFS")
print_results_for_field(treat_dataset, "DFS")
print("")
print("")

print("==> posOutcome")
print_results_for_field(treat_dataset, "posOutcome")
print("")
print("")


