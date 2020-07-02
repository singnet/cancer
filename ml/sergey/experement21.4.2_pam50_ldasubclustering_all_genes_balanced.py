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
from funs_common_balanced_with_addX import get_balanced_split_with_addX
from funs_common import read_treat_dataset, list2d_to_4g_str_pm, get_pam50_list,drop_trea, read_combat_dataset, prepare_full_dataset
import sys
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



def print_results_for_field(dataset, Xt_full_dict, field, prefix):

    dataset = drop_na(dataset, field)
    dataset_notrea_dataset =drop_trea(dataset)

    print_order = ["full_xgboost", "full_logi", "notrea_xgboost", "notrea_logi"]
    max_len_order = max(map(len,print_order))
    rez = defaultdict(list)
    for i in range(100):
        ds = get_balanced_split_with_addX(dataset, Xt_full_dict,  field)
        rez["full_xgboost"].append(calc_results_simple(*ds, XGBClassifier()))
        rez["full_logi"].append(calc_results_simple(*ds, LogisticRegression(max_iter=1000)))

        ds = get_balanced_split_with_addX(dataset_notrea_dataset, Xt_full_dict, field)
        rez["notrea_xgboost"].append(calc_results_simple(*ds, XGBClassifier()))
        rez["notrea_logi"].append(   calc_results_simple(*ds, LogisticRegression(max_iter=1000)))

#        for order in print_order:
#            print(order, " "*(max_len_order - len(order)), ": ", list_to_4g_str(rez[order][-1]))
#        print ("")

    for order in print_order:
        print("==> ", field, prefix, order, " "*(max_len_order - len(order)), ": ", list2d_to_4g_str_pm(rez[order]))



def print_results(dataset, Xt_full_dict, prefix):
    print("==> pCR", prefix)
    print_results_for_field(dataset, Xt_full_dict, "pCR", prefix)
    print("")
    print("")
    
    print("==> RFS", prefix)
    print_results_for_field(dataset, Xt_full_dict, "RFS", prefix)
    print("")
    print("")
    
    print("==> DFS", prefix)
    print_results_for_field(dataset, Xt_full_dict, "DFS", prefix)
    print("")
    print("")
    
    print("==> posOutcome", prefix)
    print_results_for_field(dataset, Xt_full_dict, "posOutcome", prefix)
    print("")
    print("")


    

treat_dataset          = read_treat_dataset()
combat_dataset = read_combat_dataset()
pam_types_cat_dataset = read_pam_types_cat_dataset()

X_full, _ = prepare_full_dataset(drop_trea(combat_dataset))

    
assert all(pam_types_cat_dataset['patient_ID'] == combat_dataset['patient_ID'])

#print(list(pam_types_cat_dataset))
#print_results(pam_types_cat_dataset, "old")



for n_cluster in [1, 5,10,20, 100, 200]:
    lda = LatentDirichletAllocation(n_components=n_cluster)
    Xt_full = lda.fit_transform(X_full - np.min(X_full))
    Xt_full_dict = {i: x for i,x in zip(combat_dataset['patient_ID'], Xt_full)}
    
    print_results(pam_types_cat_dataset, Xt_full_dict, "nc" + str(n_cluster))
      
