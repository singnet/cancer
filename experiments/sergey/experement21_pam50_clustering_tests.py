import pandas as pd
import os
import numpy as np
import random
from funs_common import *
import sys
from sklearn.cluster import KMeans


    

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
    notrea_dataset =drop_trea(dataset)


    X_full, y_full = prepare_full_dataset(dataset, y_field = field)
    X_notrea,_     = prepare_full_dataset(notrea_dataset, y_field = field)

    print(X_full.shape, X_notrea.shape)

    kf = StratifiedKFold(n_splits=5, shuffle=True)
    print_order = ["full_xgboost", "full_logi", "notrea_xgboost", "notrea_logi"]
    max_len_order = max(map(len,print_order))
    
    rez = defaultdict(list)
    for i, (train_index, test_index) in enumerate(kf.split(X_full, y_full)):
        print("split ", i)
        rez["full_xgboost"].append(calc_results_simple_fold(X_full, y_full, train_index, test_index, XGBClassifier()))
        rez["full_logi"].append(   calc_results_simple_fold(X_full, y_full, train_index, test_index, LogisticRegression(max_iter=1000)))
        rez["notrea_xgboost"].append(calc_results_simple_fold(X_notrea, y_full, train_index, test_index, XGBClassifier()))
        rez["notrea_logi"].append(   calc_results_simple_fold(X_notrea, y_full, train_index, test_index, LogisticRegression(max_iter=1000)))
        
        for order in print_order:
            print(order, " "*(max_len_order - len(order)), ": ", list_to_4g_str(rez[order][-1]))
        print ("")
    
    for order in print_order:
        print("==> ", order, " "*(max_len_order - len(order)), ": ", list_to_4g_str(np.mean(rez[order], axis=0)))
                

def make_clustering(pd_genes, pd_types, n_clusters):
    pam_types = list(sorted(set(pd_types["pam_name"])))
    rez = {}
    for t in pam_types:
        t_ids = pd_types.loc[pd_types["pam_name"] == t]["patient_ID"]
    
        selected = pd_genes.loc[pd_genes["patient_ID"].isin(t_ids)]
        selected_ids = selected["patient_ID"].to_numpy()
        selected = selected.drop(columns = ["patient_ID"])
        
        kmeans = KMeans(n_clusters=n_clusters).fit(selected.to_numpy())
        rez[t] = [ set(p for p,c in zip(selected_ids, kmeans.labels_) if (c == i)) for i in range(n_clusters)]
    return rez
                                                                                    

treat_dataset          = read_treat_dataset()
pd_types  = pd.read_csv('coincideTypes.csv')
pd_combat = pd.read_csv("merged-combat15.csv")

selected_pam50 = list(set(get_pam50_list()).intersection(set(pd_combat)))    
pd_combat_pam50 = pd_combat[["patient_ID"] + selected_pam50]

n_clusters = 4

#rez_cluster = make_clustering(pd_combat, pd_types, n_clusters)
rez_cluster = make_clustering(pd_combat_pam50, pd_types, 4)


studies = list(set(treat_dataset["study"]))
pam_types = list(sorted(set(pd_types["pam_name"])))




for t in pam_types:
    for i in range(n_clusters):
        print(t, i, len(rez_cluster[t][i]))
    all_class_sizes = []
    for study in studies:
        study_patients = set(treat_dataset.loc[treat_dataset["study"] == study]["patient_ID"])
        pd_types_study = pd_types.loc[pd_types["patient_ID"].isin(study_patients)]
        total_n = len(pd_types_study.loc[pd_types_study["pam_name"] == t])
        class_sizes = [len(rez_cluster[t][i].intersection(study_patients))  for i in range(n_clusters)]        
        total_n_2 =  sum(class_sizes)
        assert total_n == total_n_2
        all_class_sizes.append(class_sizes)
    print(t)
    print(all_class_sizes)



