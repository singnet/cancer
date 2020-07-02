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
from funs_common import select_pam50, drop_trea, read_full_dataset, read_mike_dataset, read_treat_dataset, prepare_dataset, list_to_4g_str
import sys


def get_balanced_study(full_dataset, study):
    X,y = prepare_dataset(full_dataset, study)
    return random_upsample_balance(X, y)
        
def get_balanced_studies_except_test_study(full_dataset, test_study):
    all_studies = list(set(full_dataset['study']))
    all_Xy = [get_balanced_study(full_dataset, study) for study in all_studies if (study != test_study)]
    all_X, all_y = zip(*all_Xy)
    return np.concatenate(all_X),  np.concatenate(all_y)
                        
    

def calc_results_onlystudy(X, y, train_index, test_index, clf):
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
                                                                        

def calc_results_withfull(X, y, train_index, test_index, full_dataset, test_study, clf):        
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
#    X_train, y_train = stack_datasets_upsample( [(X_train, y_train), (X_train_other, y_train_other)])
    
    
    
    clf.fit(X_train,y_train)    
    y_pred  = clf.predict(X_test)
    
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
mike_pam50          = select_pam50(mike_dataset)
full_pam50          = select_pam50(full_dataset)

all_studies = list(set(full_dataset['study']))


print_order = [ "mike_notrea_f", "mike_notrea_s", "mike_notrea_svm_f", "mike_notrea_svm_s", "mike_notrea_logi_f", "mike_notrea_logi_s", "mike_pam50_svm_f", "mike_pam50_svm_s","mike_pam50_logi_f", "mike_pam50_logi_s"]
max_len_order = max(map(len,print_order))

for study in sorted(all_studies):
#for study in ['study_20194_GPL96_all-bmc15']:
#for study in ['study_17705_GPL96_MDACC_Tissue_BC_Tamoxifen-bmc15']:
    
    
    X_full, y_full   = prepare_dataset(full_dataset, study)
    X_full_notrea, _ = prepare_dataset(full_notrea_dataset, study)
    X_mike, _        = prepare_dataset(mike_dataset, study)
    X_mike_notrea, _ = prepare_dataset(mike_notrea_dataset, study)
    X_trea, _        = prepare_dataset(treat_dataset, study)
    X_full_pam50,_   = prepare_dataset(full_pam50, study)
    X_mike_pam50,_   = prepare_dataset(mike_pam50, study)
    
    print("==>", study, count_to_str(y_full))
    kf = StratifiedKFold(n_splits=5)
    
    rez = defaultdict(list)
    for i, (train_index, test_index) in enumerate(kf.split(X_full, y_full)):
        y_train, y_test = y_full[train_index], y_full[test_index]
        print("split: ", i, "train: ", count_to_str(y_train), "  test:", count_to_str(y_test))
        
#        rez["full"].append(calc_results(       X_full,        y_full, train_index, test_index, full_dataset, study, XGBClassifier()))
#        rez["full_notrea"].append(calc_results(X_full_notrea, y_full, train_index, test_index, full_notrea_dataset, study,  XGBClassifier()))
#        rez["mike"].append(calc_results_withfull(       X_mike,        y_full, train_index, test_index, mike_dataset, study, XGBClassifier()))
#        rez["mike_notrea_f"].append(calc_results_withfull( X_mike_notrea, y_full, train_index, test_index, mike_notrea_dataset, study, XGBClassifier()))
#        rez["mike_notrea_s"].append(calc_results_onlystudy(X_mike_notrea, y_full, train_index, test_index, XGBClassifier()))
#        rez["trea"].append(calc_results_withfull(       X_trea,        y_full, train_index, test_index, treat_dataset, study, XGBClassifier()))

#        rez["mike_svm"].append(calc_results_withfull(       X_mike,        y_full, train_index, test_index, mike_dataset, study,  svm.SVC()))
#        rez["mike_notrea_svm_f"].append(calc_results_withfull( X_mike_notrea, y_full, train_index, test_index, mike_notrea_dataset, study,  svm.SVC()))
#        rez["mike_notrea_svm_s"].append(calc_results_onlystudy(X_mike_notrea, y_full, train_index, test_index, svm.SVC()))
#        rez["trea_svm"].append(calc_results_withfull(       X_trea,        y_full, train_index, test_index, treat_dataset, study, svm.SVC()))
        
#        rez["mike_logi"].append(calc_results_withfull(       X_mike,        y_full, train_index, test_index, mike_dataset, study,  LogisticRegression(max_iter=1000)))
        rez["mike_notrea_f"].append(calc_results_withfull( X_mike_notrea, y_full, train_index, test_index, mike_notrea_dataset, study,  XGBClassifier()))
        rez["mike_notrea_s"].append(calc_results_onlystudy(X_mike_notrea, y_full, train_index, test_index, XGBClassifier()))


        rez["mike_notrea_logi_f"].append(calc_results_withfull( X_mike_notrea, y_full, train_index, test_index, mike_notrea_dataset, study,  LogisticRegression(max_iter=1000)))
        rez["mike_notrea_logi_s"].append(calc_results_onlystudy(X_mike_notrea, y_full, train_index, test_index, LogisticRegression(max_iter=1000)))
        rez["mike_notrea_svm_f"].append(calc_results_withfull( X_mike_notrea, y_full, train_index, test_index, mike_notrea_dataset, study,  svm.SVC()))
        rez["mike_notrea_svm_s"].append(calc_results_onlystudy(X_mike_notrea, y_full, train_index, test_index, svm.SVC()))
        #        rez["trea_logi"].append(calc_results_withfull(       X_trea,        y_full, train_index, test_index, treat_dataset, study, LogisticRegression(max_iter=1000)))
        
        rez["mike_pam50_logi_f"].append(calc_results_withfull( X_mike_pam50, y_full, train_index, test_index, mike_pam50, study,  LogisticRegression(max_iter=1000)))
        rez["mike_pam50_logi_s"].append(calc_results_onlystudy(X_mike_pam50, y_full, train_index, test_index, LogisticRegression(max_iter=1000)))
        rez["mike_pam50_svm_f"].append(calc_results_withfull( X_mike_pam50, y_full, train_index, test_index, mike_pam50, study,  svm.SVC()))
        rez["mike_pam50_svm_s"].append(calc_results_onlystudy(X_mike_pam50, y_full, train_index, test_index, svm.SVC()))

        for order in print_order:
            if (order in rez):
                print(order, " "*(max_len_order - len(order)), ": ", list_to_4g_str(rez[order][-1]))
        print("")
        sys.stdout.flush()
    for order in print_order:
        if (order in rez):
            print("==> ", order, " "*(max_len_order - len(order)), ": ", list_to_4g_str(np.mean(rez[order], axis=0)))
    print("")
    print("")
