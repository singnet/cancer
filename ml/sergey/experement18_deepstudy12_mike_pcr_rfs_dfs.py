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
from funs_common_balanced import get_balanced_studies_except_test_study
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

def calc_results_onlystudy(X, y, train_index, test_index, clf):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    X_train, y_train = random_upsample_balance(X_train, y_train)
    X_test, y_test   = random_upsample_balance(X_test, y_test)
    
    return calc_results_simple(X_train, X_test, y_train, y_test, clf)

def calc_results_withfull_balanced2(X, y, train_index, test_index, full_dataset, test_study, clf):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    X_train, y_train = random_upsample_balance(X_train, y_train)
    X_test, y_test   = random_upsample_balance(X_test, y_test)
    
                    
    X_train_other, y_train_other = get_balanced_studies_except_test_study(full_dataset, test_study)
    N_rep =  int(len(y_train_other) / len(y_train))
    
    X_train_rep = np.repeat(X_train, N_rep, axis=0)
    y_train_rep = np.repeat(y_train, N_rep, axis=0)
    
    X_train = np.concatenate([X_train_rep, X_train_other])
    y_train = np.concatenate([y_train_rep, y_train_other])
    
    return calc_results_simple(X_train, X_test, y_train, y_test, clf)

def calc_results_withfull_simple(X, y, train_index, test_index, full_dataset, test_study, clf):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    X_train, y_train = random_upsample_balance(X_train, y_train)
    X_test, y_test   = random_upsample_balance(X_test, y_test)
    
                    
    X_train_other, y_train_other = get_balanced_studies_except_test_study(full_dataset, test_study)

    X_train = np.concatenate([X_train, X_train_other])
    y_train = np.concatenate([y_train, y_train_other])
    
    return calc_results_simple(X_train, X_test, y_train, y_test, clf)

def count_to_str(y):
    c = Counter(y)
    return "count_01=%i/%i"%(c[0], c[1])
        

def print_results_for_field(dataset, field):
    
    dataset = drop_na(dataset, field)
    dataset_notrea = drop_trea(dataset)
    
    print_order = ["withtrea_xgboost_fb", "withtrea_xgboost_fs", "withtrea_xgboost_s", "withtrea_logi_fb", "withtrea_logi_fs", "withtrea_logi_s"]
    max_len_order = max(map(len,print_order))
    all_studies = list(set(dataset['study']))
    
    for study in sorted(all_studies):            
        X_study_full, y_study_full     = prepare_dataset(dataset,        study)
        X_study_notrea, y_study_notrea = prepare_dataset(dataset_notrea, study)
        print("==> (%s)"%field, study, count_to_str(y_study_full))
        
        rez = defaultdict(list)
        kf = StratifiedKFold(n_splits=5, shuffle = True)
        for i, (train_index, test_index) in enumerate(kf.split(X_study_full, y_study_full)):
            rez["withtrea_xgboost_fb"].append(calc_results_withfull_balanced2(X_study_full, y_study_full, train_index, test_index, dataset, study,  XGBClassifier()))
            rez["withtrea_xgboost_fs"].append(calc_results_withfull_simple(   X_study_full, y_study_full, train_index, test_index, dataset, study,  XGBClassifier()))
            rez["withtrea_xgboost_s"].append( calc_results_onlystudy(         X_study_full, y_study_full, train_index, test_index, XGBClassifier()))
            rez["withtrea_logi_fb"].append(   calc_results_withfull_balanced2(X_study_full, y_study_full, train_index, test_index, dataset, study,  LogisticRegression(max_iter=1000)))
            rez["withtrea_logi_fs"].append(   calc_results_withfull_simple(   X_study_full, y_study_full, train_index, test_index, dataset, study,  LogisticRegression(max_iter=1000)))
            rez["withtrea_logi_s"].append(    calc_results_onlystudy(         X_study_full, y_study_full, train_index, test_index, LogisticRegression(max_iter=1000)))
            
        
            for order in print_order:
                print(order, " "*(max_len_order - len(order)), ": ", list_to_4g_str(rez[order][-1]))
            print ("")
            sys.stdout.flush()
            
        for order in print_order:
            print("==> (%s)"%field, order, " "*(max_len_order - len(order)), ": ", list2d_to_4g_str_pm(rez[order]))
                
mike_dataset = read_mike_dataset()


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


