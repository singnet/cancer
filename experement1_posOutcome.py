import pandas as pd
import os
import numpy as np
import random
import math 
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score


nval = 3


def convert_surgery(x): 
    if (x == "mastectomy"): 
        return 1 
    if (x == "breast preserving"): 
        return 2 
    if (x == 'NA'):  
        return 0 
    raise Exception("bad surgery") 

def convert_posOutcome(x):
    if (x == '2'):
        return 1
    return int(x)

def read_treat_dataset(path="./"):
    return pd.read_csv(os.path.join(path, 'bmc15mldata1.csv'), converters=dict(surgery = convert_surgery, posOutcome = convert_posOutcome))

def read_full_dataset(path = "example15bmc"):
    datasets_fn = ["study_12093_GPL96_all-bmc15.csv.xz",
    "study_1379_GPL1223_all-bmc15.csv.xz",
    "study_16391_GPL570_all-bmc15.csv.xz",
    "study_16446_GPL570_all-bmc15.csv.xz",
    "study_17705_GPL96_JBI_Tissue_BC_Tamoxifen-bmc15.csv.xz",
    "study_17705_GPL96_MDACC_Tissue_BC_Tamoxifen-bmc15.csv.xz",
    "study_19615_GPL570_all-bmc15.csv.xz",
    "study_20181_GPL96_all-bmc15.csv.xz",
    "study_20194_GPL96_all-bmc15.csv.xz",
    "study_2034_GPL96_all-bmc15.csv.xz",
    "study_22226_GPL1708_all-bmc15.csv.xz",
    "study_22358_GPL5325_all-bmc15.csv.xz",
    "study_25055_GPL96_MDACC_M-bmc15.csv.xz",
    "study_25065_GPL96_MDACC-bmc15.csv.xz",
    "study_25065_GPL96_USO-bmc15.csv.xz",
    "study_32646_GPL570_all-bmc15.csv.xz",
    "study_9893_GPL5049_all-bmc15.csv.xz"]

    datasets = [pd.read_csv(os.path.join(path,d)) for d in datasets_fn]
    join_dataset = pd.concat(datasets)
    
    treat_dataset = read_treat_dataset()
    return pd.merge(treat_dataset, join_dataset)

def read_oleg_dataset():
    join_dataset = pd.read_csv("ex15bmcMerged.csv")
    treat_dataset = read_treat_dataset()
    return pd.merge(treat_dataset, join_dataset)

def prepare_dataset(full_dataset, study):
    study_dataset = full_dataset.loc[full_dataset['study'] == study]
    X = study_dataset.drop(columns=['study', 'patient_ID','pCR', 'RFS', 'DFS', 'posOutcome']).to_numpy()
    y_posOutcome = study_dataset['posOutcome'].to_numpy()
    return X, y_posOutcome

def prepare_datasets(full_dataset, studies):
    Xys = [prepare_dataset(full_dataset, study ) for study in studies] 
    Xs, ys = zip(*Xys)
    return np.concatenate(Xs), np.concatenate(ys)

def _get_shuffled_study_list(full_dataset):
    all_studies = list(set(full_dataset['study']))
    random.shuffle(all_studies)
    return all_studies

def study_fold(full_dataset, shuffled_study_list, nval):
    return prepare_datasets(full_dataset, shuffled_study_list[:nval]), prepare_datasets(full_dataset, shuffled_study_list[nval:])


def calc_results_for_fold(full_dataset, shuffled_study_list, nval):
    (X_train, y_train), (X_test, y_test) = study_fold(full_dataset, shuffled_study_list, nval)
    
    clf = XGBClassifier()
    clf.fit(X_train,y_train)
    
    y_true_prob = clf.predict_proba(X_test)[:,1]
    y_true      = clf.predict(X_test)
    
    acc = np.mean(y_true == y_test)
    auc_1 = roc_auc_score(y_test, y_true_prob)
    auc_2 = roc_auc_score(y_test, y_true)
    
    return acc, auc_1, auc_2

def calc_results_for_fold_random(full_dataset, shuffled_study_list, nval):
    (X_train, y_train), (X_test, y_test) = study_fold(full_dataset, shuffled_study_list, nval)
    
    X_train = np.random.rand(*X_train.shape)
    X_test  = np.random.rand(*X_test.shape)
    
    clf = XGBClassifier()
    clf.fit(X_train,y_train)
    
    y_true_prob = clf.predict_proba(X_test)[:,1]
    y_true      = clf.predict(X_test)
    
    acc = np.mean(y_true == y_test)
    auc_1 = roc_auc_score(y_test, y_true_prob)
    auc_2 = roc_auc_score(y_test, y_true)
    
    return acc, auc_1, auc_2


full_dataset = read_full_dataset()
oleg_dataset = read_oleg_dataset()
treat_dataset = read_treat_dataset()

for i in range(20):
    print("run", i)
    shuffled_study_list = _get_shuffled_study_list(full_dataset)
    print("val:", shuffled_study_list[:nval])
    
    acc_f, auc_1_f, auc_2_f = calc_results_for_fold(full_dataset, shuffled_study_list, nval)
    acc_o, auc_1_o, auc_2_o = calc_results_for_fold(oleg_dataset, shuffled_study_list, nval)
    acc_t, auc_1_t, auc_2_t = calc_results_for_fold(treat_dataset, shuffled_study_list, nval)
    acc_r, auc_1_r, auc_2_r = calc_results_for_fold_random(treat_dataset, shuffled_study_list, nval)

    
    print("full: ", acc_f, auc_1_f, auc_2_f)
    print("oleg: ", acc_o, auc_1_o, auc_2_o)
    print("trea: ", acc_t, auc_1_t, auc_2_t)
    print("rand: ", acc_r, auc_1_r, auc_2_r)
    
