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
from funs_common_balanced import get_balanced_split
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
        
def print_results_for_field(dataset, field):
    
    
    dataset = drop_na(dataset, field)
    
    
    print_order = ["full_xgboost", "full_logi"]
    max_len_order = max(map(len,print_order))
    
    rez = defaultdict(list)
    for i in range(10):
        print("run ", i)
        ds = get_balanced_split(dataset)
        rez["full_xgboost"].append(calc_results_simple(*ds, XGBClassifier()))
        rez["full_logi"].append(   calc_results_simple(*ds, LogisticRegression(max_iter=1000)))
                
        for order in print_order:
            print(order, " "*(max_len_order - len(order)), ": ", list_to_4g_str(rez[order][-1]))
        print ("")
        sys.stdout.flush()
        
    for order in print_order:
        print("==> ", order, " "*(max_len_order - len(order)), ": ", list2d_to_4g_str_pm(rez[order]))
                
    

trea_dataset = read_treat_dataset()


print("==> pCR (full) ")
print_results_for_field(trea_dataset, "pCR")
print("")
print("")


print("==> RFS (full)")
print_results_for_field(trea_dataset, "RFS")
print("")
print("")


print("==> DFS (full)")
print_results_for_field(trea_dataset, "DFS")
print("")
print("")

print("==> posOutcome (full)")
print_results_for_field(trea_dataset, "posOutcome")
print("")
print("")


