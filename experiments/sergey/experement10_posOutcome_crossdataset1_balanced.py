import pandas as pd
import os
import numpy as np
import random
import math 
from xgboost import XGBClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score
from sklearn.model_selection import RepeatedStratifiedKFold
from funs_balance import random_upsample_balance
from collections import Counter,defaultdict
import sys
from funs_common import select_pam50, drop_trea, read_full_dataset, read_mike_dataset, read_treat_dataset, prepare_dataset, list_to_4g_str

            
def get_balanced_study(full_dataset, study):
    X,y = prepare_dataset(full_dataset, study)
    return random_upsample_balance(X, y)
        
def get_balanced_studies_except_test_study(full_dataset, test_study):
    all_studies = list(set(full_dataset['study']))
    all_Xy = [get_balanced_study(full_dataset, study) for study in all_studies if (study != test_study)]
    all_X, all_y = zip(*all_Xy)
    return np.concatenate(all_X),  np.concatenate(all_y)

def calc_results_for_test_study(full_dataset, test_study, clf):
    X_train, y_train = get_balanced_studies_except_test_study(full_dataset, test_study)
    X_test, y_test = get_balanced_study(full_dataset, test_study)    
    
    
    clf.fit(X_train,y_train)
    
    y_pred      = clf.predict(X_test)
    
    acc = np.mean(y_pred == y_test)
    recall_0 =  recall_score(y_test, y_pred, pos_label=0)
    recall_1 =  recall_score(y_test, y_pred, pos_label=1)
    
    return acc, recall_0, recall_1
                    
    


full_dataset = read_full_dataset()
mike_dataset = read_mike_dataset()
treat_dataset = read_treat_dataset()

full_notrea_dataset = drop_trea(full_dataset)
mike_notrea_dataset = drop_trea(mike_dataset)
mike_pam50          = select_pam50(mike_dataset)
full_pam50          = select_pam50(full_dataset)



all_studies = list(set(full_dataset['study']))

print_order = ["full", "full_notrea", "full_pam50", "mike", "mike_svm", "mike_logi", "mike_notrea", "mike_notrea_svm", "mike_notrea_logi", "mike_pam50", "mike_pam50_svm", "mike_pam50_logi", "trea", "trea_svm", "trea_logi"]
max_len_order = max(map(len,print_order))
for test_study in sorted(all_studies):    
    print("study to test:", test_study)    

    # we make random upsampling, so results might be different
    rez = defaultdict(list)
    
    for i in range(5):
        rez["full"].append(            calc_results_for_test_study(full_dataset,        test_study, XGBClassifier()))
        rez["full_notrea"].append(     calc_results_for_test_study(full_notrea_dataset, test_study, XGBClassifier()))
        rez["full_pam50"].append(      calc_results_for_test_study(full_pam50,          test_study, XGBClassifier()))
        
        rez["mike"].append(            calc_results_for_test_study(mike_dataset, test_study, XGBClassifier()))
        rez["mike_svm"].append(        calc_results_for_test_study(mike_dataset, test_study, svm.SVC()))
        rez["mike_logi"].append(       calc_results_for_test_study(mike_dataset, test_study, LogisticRegression(max_iter=1000)))
        
        rez["mike_notrea"].append(     calc_results_for_test_study(mike_notrea_dataset, test_study, XGBClassifier()))
        rez["mike_notrea_svm"].append( calc_results_for_test_study(mike_notrea_dataset, test_study, svm.SVC()))
        rez["mike_notrea_logi"].append(calc_results_for_test_study(mike_notrea_dataset, test_study, LogisticRegression(max_iter=1000)))

        rez["mike_pam50"].append(      calc_results_for_test_study(mike_pam50, test_study, XGBClassifier()))
        rez["mike_pam50_svm"].append(  calc_results_for_test_study(mike_pam50, test_study, svm.SVC()))
        rez["mike_pam50_logi"].append( calc_results_for_test_study(mike_pam50, test_study, LogisticRegression(max_iter=1000)))

        
        rez["trea"].append(            calc_results_for_test_study(treat_dataset, test_study, XGBClassifier()))
        rez["trea_svm"].append(        calc_results_for_test_study(treat_dataset, test_study, svm.SVC()))
        rez["trea_logi"].append(       calc_results_for_test_study(treat_dataset, test_study, LogisticRegression(max_iter=1000)))
        for order in print_order:
            print(order, " "*(max_len_order - len(order)), ": ", list_to_4g_str(rez[order][-1]))
        print ("")
        sys.stdout.flush()
        
    for order in print_order:
        print("==> ", order, " "*(max_len_order - len(order)), ": ", list_to_4g_str(np.mean(rez[order], axis=0)))
    print("")
    print("")
