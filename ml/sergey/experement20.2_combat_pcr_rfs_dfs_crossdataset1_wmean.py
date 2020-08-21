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

def calc_results_for_test_study(full_dataset, test_study, clf):    
    all_train_studies = list(set(full_dataset['study']))
    all_train_studies.remove(test_study)
    
    (X_test, y_test)  = prepare_dataset(full_dataset, test_study)
    (X_train,y_train) = prepare_datasets(full_dataset, all_train_studies)
    return calc_results_simple(X_train, X_test, y_train, y_test, clf)
    

def print_results_for_field(dataset, field):
    
    dataset = drop_na(dataset, field)
    notrea_dataset =drop_trea(dataset)
    
    all_studies = list(set(dataset['study']))
    
    print_order = ["full_xgboost", "full_logi", "notrea_xgboost", "notrea_logi"]
    max_len_order = max(map(len,print_order))

    all_rez = defaultdict(list)
        
    for test_study in sorted(all_studies):
        print("==> %s study to test:"%field, test_study)
        
        rez = defaultdict(list)
        
        rez["full_xgboost"] = calc_results_for_test_study(dataset, test_study, XGBClassifier())
        rez["full_logi"]    = calc_results_for_test_study(dataset, test_study, LogisticRegression(max_iter=10000))
        
        rez["notrea_xgboost"] = calc_results_for_test_study(notrea_dataset, test_study, XGBClassifier())
        rez["notrea_logi"]    = calc_results_for_test_study(notrea_dataset, test_study, LogisticRegression(max_iter=10000))
        
        for order in print_order:
            all_rez[order].append(rez[order][-1])
                                
        for order in print_order:
            print(field, order, " "*(max_len_order - len(order)), ": ", list_to_4g_str(rez[order]))
    
    for order in print_order:
        print(f"==> mean_auc {order}", "%.4g"%np.mean( all_rez[order]))
                    

mike_dataset = read_combat_dataset()


print("==> pCR")
print_results_for_field(mike_dataset, "pCR")
print("")
print("")

print("==> RFS")
print_results_for_field(mike_dataset, "RFS")
print("")
print("")

print("==> DFS")
print_results_for_field(mike_dataset, "DFS")
print("")
print("")

print("==> posOutcome")
print_results_for_field(mike_dataset, "posOutcome")
print("")
print("")


