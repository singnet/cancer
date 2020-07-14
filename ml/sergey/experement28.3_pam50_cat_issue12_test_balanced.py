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
import sys
from sklearn.preprocessing import OneHotEncoder
import itertools
from funs_common import read_combat_dataset, read_treat_dataset


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


def calc_results_with_ys(y_test, y_pred, y_pred_prob):
    acc = np.mean(y_test == y_pred)    
    recall_0 =  recall_score(y_test, y_pred, pos_label=0)
    recall_1 =  recall_score(y_test, y_pred, pos_label=1)    
    auc = roc_auc_score(y_test, y_pred_prob)
    return acc, recall_0, recall_1, auc


def calc_results_simple_fold(X, y, train_index, test_index, clf):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    return calc_results_simple(X_train, X_test, y_train, y_test, clf)

def prepare_dataset_septrea(full_dataset, study, y_field = 'posOutcome'):
    study_dataset = full_dataset.loc[full_dataset['study'] == study]
    list_trea = ['radio','surgery','chemo','hormone']
    X_notrea = study_dataset.drop(columns=list_trea + ['study', 'patient_ID','pCR', 'RFS', 'DFS', 'posOutcome']).to_numpy(dtype = np.float)
    X_trea   = study_dataset[list_trea].to_numpy()
    y_posOutcome = study_dataset[y_field].to_numpy(dtype = np.float)
    return X_trea, X_notrea, y_posOutcome
            

def prepare_full_dataset_septrea(full_dataset, y_field = 'posOutcome'):
    list_trea = ['radio','surgery','chemo','hormone']
    X_notrea = full_dataset.drop(columns=list_trea + ['study', 'patient_ID','pCR', 'RFS', 'DFS', 'posOutcome']).to_numpy()
    X_trea   = full_dataset[list_trea].to_numpy()
    y_posOutcome = full_dataset[y_field].to_numpy()
    return X_trea, X_notrea, y_posOutcome


def get_balanced_split_for_study_septrea(full_dataset, study, y_field = "posOutcome"):
    X_trea, X_notrea, y = prepare_dataset_septrea(full_dataset, study, y_field = y_field)
    X =np.concatenate([X_trea, X_notrea], axis=1)
        
    
    kf = RepeatedStratifiedKFold(n_splits=5)
    (train_index, test_index) = next(kf.split(X,y))
    
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
                        
    X_train, y_train = random_upsample_balance(X_train, y_train)
    X_test, y_test   = random_upsample_balance(X_test, y_test)
    
    return X_train, X_test, y_train, y_test
     
def get_balanced_split_septrea(full_dataset, y_field = 'posOutcome'):
    
    all_X_train, all_X_test, all_y_train, all_y_test = [],[],[],[]
    
    for study in list(set(full_dataset['study'])):
        X_train, X_test, y_train, y_test = get_balanced_split_for_study_septrea(full_dataset, study, y_field)
        all_X_train.append(X_train)
        all_X_test.append(X_test)
        all_y_train.append(y_train)
        all_y_test.append(y_test)
    return np.concatenate(all_X_train),  np.concatenate(all_X_test), np.concatenate(all_y_train), np.concatenate(all_y_test)

def get_possible_treatments_combinations(X_trea):
    possible_treats = [list(set(Xt)) for Xt in np.transpose(X_trea)]
    return np.array(list(itertools.product(*possible_treats)))

def calc_results_max_treatments(clf, X_test, y_test, possible_treat):
    Ndiff = 0
    y_pred_prob = []
    for Xi,yi in zip(X_test, y_test):
        Xir = np.tile(Xi, (possible_treat.shape[0], 1))
        Xir[:, :possible_treat.shape[1]] = possible_treat
        yir_pred_prob = clf.predict_proba(Xir)[:,1]
        if (np.min(yir_pred_prob) != np.max(yir_pred_prob)):
            Ndiff +=1 
        
        if (yi == 0):
            y_pred_prob.append(np.min(yir_pred_prob))
        else:
            y_pred_prob.append(np.max(yir_pred_prob))
    return Ndiff , roc_auc_score(y_test, y_pred_prob)
        
def print_results_for_field(dataset, field):
    
    dataset = drop_na(dataset, field)
    
    X_trea, _, _ = prepare_full_dataset_septrea(dataset)
    
    possible_treat = get_possible_treatments_combinations(X_trea)
    
    for i in range(10):            
        X_train, X_test, y_train, y_test = get_balanced_split_septrea(dataset)
                
        clf =  XGBClassifier()
        clf.fit(X_train,y_train)
        assert clf.classes_[1] == 1
        
        orig_auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:,1])
        N_diff, auc = calc_results_max_treatments(clf, X_test, y_test, possible_treat)
        print("fold ", i, orig_auc, auc, N_diff)
        
                
    

#full_dataset = read_full_dataset()
#mike_dataset = read_combat_dataset()
dataset = read_pam_types_cat_dataset()


print("==> pCR")
print_results_for_field(dataset, "pCR")
print("")
print("")


print("==> RFS")
print_results_for_field(dataset, "RFS")
print("")
print("")


print("==> DFS")
print_results_for_field(dataset, "DFS")
print("")
print("")


print("==> posOutcome")
print_results_for_field(dataset, "posOutcome")
print("")
print("")


