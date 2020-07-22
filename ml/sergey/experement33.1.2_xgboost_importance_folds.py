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
from funs_common import read_alltreat_dataset, read_combat_dataset, prepare_full_dataset, list2d_to_4g_str_pm, drop_trea, list_to_4g_str, read_coincide_types_dataset
import sys
import itertools    
from class_experement32_BiasedXgboost import BiasedXgboost
from class_experement32_DoubleXgboost import DoubleXgboost

def read_pam_types_num_dataset():
    coincide_types_dataset = read_coincide_types_dataset()
    print(list(coincide_types_dataset))
    return coincide_types_dataset[["patient_ID", "study", 'pCR', 'RFS', 'DFS','radio','surgery','chemo','hormone', 'posOutcome', 'pam_cat']]

def print_counter(pam_type, y):
    c = Counter(zip(pam_type,y))
    for t in range(1,6):
        n0 = c[(t,0)]
        n1 = c[(t,1)]
        n01 = n0 + n1
        surv_frac = 0
        if n01 > 0:
            surv_frac = n1 / n01
        
        n_frac = n01 / len(y)
        
        print(f"pam_type, n01, n_frac, surv_frac: {t}  {n01:3}  {n_frac*100:5.0f}%  {surv_frac*100:5.0f}%")

        
def add_one_features_tail(X, f):
    return np.concatenate([X, np.ones((X.shape[0], 1)) * f ], axis=1)

def print_mean_fold_importance(X,y,genes_list):
    kf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10) 
    all_list20 = []
    for i, (train_index, test_index) in enumerate(kf.split(X, y)):
        #        print("fold", i)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
                
        clf = XGBClassifier()
        clf.fit(X_train,y_train)
        #        print(sorted(np.argsort(-1 * clf.feature_importances_)[:20]))
        list20_current = list(np.argsort(-1 * clf.feature_importances_)[:20])
        all_list20 += list20_current
#        if (i == 0):
#            set20_0 = set(list20_current)
#        else:
            #            print(set20_0.intersection(list20_current))
        #print(np.sort(-1 * clf.feature_importances_)[:20])
    print([(genes_list[gene_id], n) for gene_id, n in Counter(all_list20).most_common(10)])

def print_results(dataset, set1, set2):
    genes_list = list(dataset.drop(columns=['study', 'patient_ID','pCR', 'RFS', 'DFS', 'posOutcome']))
    X_set1, y_set1 = prepare_full_dataset(dataset.loc[dataset['patient_ID'].isin(set1)])
    X_set2, y_set2 = prepare_full_dataset(dataset.loc[dataset['patient_ID'].isin(set2)])

    print("set1")
    print_mean_fold_importance(X_set1,y_set1,genes_list)

#    print("set2")
#    print_mean_fold_importance(X_set2,y_set2,genes_list)

    print("stack")
    print_mean_fold_importance(np.concatenate([X_set1, X_set2]) , np.concatenate([y_set1, y_set2]) , genes_list)
     

    

    X_set1_wf = add_one_features_tail(X_set1, 0)
    X_set2_wf = add_one_features_tail(X_set2, 1)

    print("stack_with_set")
    print_mean_fold_importance(np.concatenate([X_set1_wf, X_set2_wf]),np.concatenate([y_set1, y_set2]), genes_list)
         
    

    
    
                
atreat_dataset = read_alltreat_dataset()
combat_dataset = read_combat_dataset()
notrea_dataset = drop_trea(combat_dataset)
#dataset = read_pam_types_num_dataset()
#notrea_dataset = drop_trea(dataset)


#Variant 1  study_20194_GPL96_all-bmc15 protocol 1 vs protocol 5

set1 = atreat_dataset.loc[(atreat_dataset['study'] == "study_20194_GPL96_all-bmc15") & (atreat_dataset['treatment_protocol_number'] == '1')]['patient_ID']
set2 = atreat_dataset.loc[(atreat_dataset['study'] == "study_20194_GPL96_all-bmc15") & (atreat_dataset['treatment_protocol_number'] == '5')]['patient_ID']

print("==> study_20194_GPL96_all-bmc15 protocol 1  vs protocol 5")
print_results(notrea_dataset, set1, set2)
print("==>")


#Variant 2  study_9893_GPL5049_all-bmc15 protocol 1 vs protocol 2

set1 = atreat_dataset.loc[(atreat_dataset['study'] == "study_9893_GPL5049_all-bmc15") & (atreat_dataset['treatment_protocol_number'] == '1')]['patient_ID']
set2 = atreat_dataset.loc[(atreat_dataset['study'] == "study_9893_GPL5049_all-bmc15") & (atreat_dataset['treatment_protocol_number'] == '2')]['patient_ID']

print("==> study_9893_GPL5049_all-bmc15 protocol 1 vs protocol 2")
print_results(notrea_dataset, set1, set2)
print("==> ")


