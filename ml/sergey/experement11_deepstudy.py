import pandas as pd
import os
import numpy as np
import random
import math 
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score,f1_score,recall_score
from sklearn.model_selection import KFold, RepeatedKFold, RepeatedStratifiedKFold
from collections import Counter,defaultdict
from funs_balance import random_upsample_balance
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from funs_common import select_pam50, drop_trea, read_full_dataset, read_mike_dataset, read_treat_dataset, prepare_dataset, list_to_4g_str





def calc_results_for_fold(X, y, train_index, test_index, clf):        
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    X_train, y_train = random_upsample_balance(X_train, y_train)
    X_test, y_test   = random_upsample_balance(X_test, y_test)
    
    clf.fit(X_train,y_train)    
    y_pred  = clf.predict(X_test)
    
    acc = np.mean(y_test == y_pred)
    recall_0 =  recall_score(y_test, y_pred, pos_label=0)
    recall_1 =  recall_score(y_test, y_pred, pos_label=1)
    
    return acc, recall_0, recall_1

def calc_results_for_ensamble(X, y, train_index, test_index, nrun, clf):        
    
    all_rez = []
    for i in range(nrun):
        
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
    
        X_train, y_train = random_upsample_balance(X_train, y_train)
        X_test, y_test   = random_upsample_balance(X_test, y_test)
    
        clf.fit(X_train,y_train)    
        y_pred  = clf.predict(X_test)
        all_rez.append(y_pred)
    
    y_pred = np.array([ Counter(ys).most_common()[0][0] for ys in np.array(all_rez).transpose()])
        
    acc = np.mean(y_test == y_pred)
    recall_0 =  recall_score(y_test, y_pred, pos_label=0)
    recall_1 =  recall_score(y_test, y_pred, pos_label=1)
    
    return acc, recall_0, recall_1


def count_to_str(y):
    c = Counter(y)
    return "count_01=%i/%i"%(c[0], c[1])

full_dataset = read_full_dataset()
mike_dataset = read_mike_dataset()
treat_dataset = read_treat_dataset()
full_notrea_dataset = drop_trea(full_dataset)
mike_notrea_dataset = drop_trea(mike_dataset)


all_studies = list(set(full_dataset['study']))


print_order = ["full", "full_notrea", "full_pam50", "mike", "mike_svm", "mike_logi", "mike_notrea", "mike_notrea_svm", "mike_notrea_logi", "mike_pam50", "mike_pam50_svm", "mike_pam50_logi", "trea", "trea_svm", "trea_logi"]
max_len_order = max(map(len,print_order))

for study in ['study_20194_GPL96_all-bmc15']:
#for study in ['study_17705_GPL96_MDACC_Tissue_BC_Tamoxifen-bmc15']:
    
    
    X_full, y_full   = prepare_dataset(full_dataset, study)
    X_full_notrea, _ = prepare_dataset(full_notrea_dataset, study)
    X_mike, _        = prepare_dataset(mike_dataset, study)
    X_mike_notrea, _ = prepare_dataset(mike_notrea_dataset, study)
    X_trea, _        = prepare_dataset(treat_dataset, study)
    
    
    print("==>", study, count_to_str(y_full))
    kf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10)
    
    rez = defaultdict(list)
    for i, (train_index, test_index) in enumerate(kf.split(X_full, y_full)):
        y_train, y_test = y_full[train_index], y_full[test_index]
        nrun = 11
        print("split: ", i, "train: ", count_to_str(y_train), "  test:", count_to_str(y_test))
        
#        rez["full"].append(calc_results_for_ensamble(       X_full,        y_full, train_index, test_index, nrun,XGBClassifier()))
#        rez["full_notrea"].append(calc_results_for_ensamble(X_full_notrea, y_full, train_index, test_index, nrun, XGBClassifier()))
        rez["mike"].append(calc_results_for_ensamble(       X_mike,        y_full, train_index, test_index, nrun,XGBClassifier()))
        rez["mike_notrea"].append(calc_results_for_ensamble(X_mike_notrea, y_full, train_index, test_index, nrun, XGBClassifier()))
        rez["trea"].append(calc_results_for_ensamble(       X_trea,        y_full, train_index, test_index, nrun, XGBClassifier()))

        rez["mike_svm"].append(calc_results_for_ensamble(       X_mike,        y_full, train_index, test_index, nrun, svm.SVC()))
        rez["mike_notrea_svm"].append(calc_results_for_ensamble(X_mike_notrea, y_full, train_index, test_index, nrun, svm.SVC()))
        rez["trea_svm"].append(calc_results_for_ensamble(       X_trea,        y_full, train_index, test_index, nrun, svm.SVC()))
        
        rez["mike_logi"].append(calc_results_for_ensamble(       X_mike,        y_full, train_index, test_index, nrun, LogisticRegression(max_iter=1000)))
        rez["mike_notrea_logi"].append(calc_results_for_ensamble(X_mike_notrea, y_full, train_index, test_index, nrun, LogisticRegression(max_iter=1000)))
        rez["trea_logi"].append(calc_results_for_ensamble(       X_trea,        y_full, train_index, test_index, nrun, LogisticRegression(max_iter=1000)))
        
        for order in print_order:
            if (order in rez):
                print(order, " "*(max_len_order - len(order)), ": ", list_to_4g_str(rez[order][-1]))
        print("")
    for order in print_order:
        if (order in rez):
            print("==> ", order, " "*(max_len_order - len(order)), ": ", list_to_4g_str(np.mean(rez[order], axis=0)))
    print("")
    print("")
