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
from sklearn.decomposition import LatentDirichletAllocation


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

def remove_nany_Xy(X,y):
    keep = np.isfinite(y)
    return X[keep], y[keep]

def print_results_for_field(Xt_full, dataset_for_y, field, prefix):
    
    y_full = dataset_for_y[field].to_numpy()    
    
    Xt_full, y_full = remove_nany_Xy(Xt_full, y_full)
    
    kf = RepeatedStratifiedKFold(n_splits=5, n_repeats = 10)
    print_order = ["xgboost", "logi"]
    max_len_order = max(map(len,print_order))
    
    rez = defaultdict(list)
    for i, (train_index, test_index) in enumerate(kf.split(Xt_full, y_full)):
        rez["xgboost"].append(calc_results_simple_fold(Xt_full, y_full, train_index, test_index,  XGBClassifier()))
        rez["logi"].append(   calc_results_simple_fold(Xt_full, y_full, train_index, test_index,  LogisticRegression(max_iter=10000)))
        
#        for order in print_order:
#            print(order, " "*(max_len_order - len(order)), ": ", list_to_4g_str(rez[order][-1]))
#        print ("")
        sys.stdout.flush()
        
    for order in print_order:
        print("==> ", field, prefix, order, " "*(max_len_order - len(order)), ": ", list2d_to_4g_str_pm(rez[order]))
                

def print_results(X_full, n_components, dataset_sel, prefix):
    lda = LatentDirichletAllocation(n_components=n_components)
    Xt_full = lda.fit_transform(X_full - np.min(X_full))
    
    print("==> pCR", prefix)
    print_results_for_field(Xt_full, dataset_sel, "pCR", prefix)
    print("")
    print("")


    print("==> RFS", prefix)
    print_results_for_field(Xt_full, dataset_sel, "RFS", prefix)
    print("")
    print("")


    print("==> DFS", prefix)
    print_results_for_field(Xt_full, dataset_sel, "DFS", prefix)
    print("")
    print("")

    print("==> posOutcome", prefix)
    print_results_for_field(Xt_full, dataset_sel, "posOutcome", prefix)
    print("")
    print("")
    

combat_dataset = read_combat_dataset()

#dataset_sel   = select_pam50(combat_dataset)

dataset_sel   = drop_trea(combat_dataset)

X_full, _ = prepare_full_dataset(dataset_sel)



for n_components in [5, 10, 20, 100, 200]:
    print_results(X_full, n_components, dataset_sel, "nc" + str(n_components))

