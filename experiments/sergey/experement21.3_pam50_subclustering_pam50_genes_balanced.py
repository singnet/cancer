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
from funs_common_balanced import get_balanced_split
from funs_common import read_treat_dataset, list2d_to_4g_str_pm, get_pam50_list,drop_trea
import sys
from sklearn.cluster import KMeans


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


def print_results_for_field(dataset, field, prefix):

    dataset = drop_na(dataset, field)
    dataset_notrea_dataset =drop_trea(dataset)

    print_order = ["full_xgboost", "full_logi", "notrea_xgboost", "notrea_logi"]
    max_len_order = max(map(len,print_order))
    rez = defaultdict(list)
    for i in range(100):
        ds = get_balanced_split(dataset, field)
        rez["full_xgboost"].append(calc_results_simple(*ds, XGBClassifier()))
        rez["full_logi"].append(calc_results_simple(*ds, LogisticRegression(max_iter=1000)))

        ds = get_balanced_split(dataset_notrea_dataset, field)
        rez["notrea_xgboost"].append(calc_results_simple(*ds, XGBClassifier()))
        rez["notrea_logi"].append(   calc_results_simple(*ds, LogisticRegression(max_iter=1000)))

#        for order in print_order:
#            print(order, " "*(max_len_order - len(order)), ": ", list_to_4g_str(rez[order][-1]))
#        print ("")

    for order in print_order:
        print("==> ", field, prefix, order, " "*(max_len_order - len(order)), ": ", list2d_to_4g_str_pm(rez[order]))


def make_subclustering(pd_genes, pd_types, n_clusters):
    pam_types = list(sorted(set(pd_types["pam_name"])))
    rez = {}
    for t in pam_types:
        t_ids = pd_types.loc[pd_types["pam_name"] == t]["patient_ID"]
    
        selected = pd_genes.loc[pd_genes["patient_ID"].isin(t_ids)]
        selected_ids = selected["patient_ID"].to_numpy()
        selected = selected.drop(columns = ["patient_ID"])
        
        kmeans = KMeans(n_clusters=n_clusters).fit(selected.to_numpy())
        for p,c in zip(selected_ids, kmeans.labels_):
            rez[p] = t + "_" + str(c)
    return rez

    

def add_subclusters_as_dummies(dataset, subclusters):
    sub = [subclusters[i] for i in dataset['patient_ID']]
    return pd.concat([dataset, pd.get_dummies(sub)], axis=1)

def print_results(dataset, prefix):
    print("==> pCR", prefix)
    print_results_for_field(dataset, "pCR", prefix)
    print("")
    print("")
    
    print("==> RFS", prefix)
    print_results_for_field(dataset, "RFS", prefix)
    print("")
    print("")
    
    print("==> DFS", prefix)
    print_results_for_field(dataset, "DFS", prefix)
    print("")
    print("")
    
    print("==> posOutcome", prefix)
    print_results_for_field(dataset, "posOutcome", prefix)
    print("")
    print("")


    

treat_dataset          = read_treat_dataset()
pd_types  = pd.read_csv('coincideTypes.csv')
pd_combat = pd.read_csv("merged-combat15.csv")

selected_pam50  = list(set(get_pam50_list()).intersection(set(pd_combat)))    
pd_combat_pam50 = pd_combat[["patient_ID"] + selected_pam50]

pam_types_cat_dataset = read_pam_types_cat_dataset()

#print(list(pam_types_cat_dataset))
print_results(pam_types_cat_dataset, "old")



for n_cluster in range(1,8):
    subclusters = make_subclustering(pd_combat_pam50, pd_types, n_cluster)
    dataset = add_subclusters_as_dummies(treat_dataset, subclusters)
    
#    for t in all_types:
#        print(t, sum(dataset[t]))
    
    print_results(dataset, "nc" + str(n_cluster))
