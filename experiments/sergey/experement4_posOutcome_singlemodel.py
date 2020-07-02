import pandas as pd
import os
import numpy as np
import random
import math 
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score,f1_score,recall_score
from sklearn import svm
from sklearn.linear_model import LogisticRegression


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
    join_dataset = pd.concat(datasets, sort=False, ignore_index=True)
    
    treat_dataset = read_treat_dataset()
    return pd.merge(treat_dataset, join_dataset)


def drop_trea(full_dataset):
    return full_dataset.drop(columns=['radio','surgery','chemo','hormone'])
    

def read_mike_dataset():
    join_dataset = pd.read_csv("ex15bmcMerged.csv")
    treat_dataset = read_treat_dataset()
    return pd.merge(treat_dataset, join_dataset)

def prepare_dataset(full_dataset, study):
    study_dataset = full_dataset.loc[full_dataset['study'] == study]
    X = study_dataset.drop(columns=['study', 'patient_ID','pCR', 'RFS', 'DFS', 'posOutcome']).to_numpy(dtype = np.float)
    y_posOutcome = study_dataset['posOutcome'].to_numpy(dtype = np.float)
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
    return prepare_datasets(full_dataset, shuffled_study_list[nval:]), prepare_datasets(full_dataset, shuffled_study_list[:nval])


def calc_results(full_dataset, test_study, clf):
    all_train_studies = list(set(full_dataset['study']) - set([test_study]))
    (X_test, y_test)  = prepare_dataset(full_dataset, test_study)
    (X_train,y_train) = prepare_datasets(full_dataset, all_train_studies) 
        
    clf.fit(X_train,y_train)
    
    y_pred      = clf.predict(X_test)
#    print(y_test)
#    print(y_pred)
    acc = np.mean(y_pred == y_test)
#    auc_1 = roc_auc_score(y_test, y_pred_prob)
#    auc_2 = roc_auc_score(y_test, y_pred)
    recall_0 =  recall_score(y_test, y_pred, pos_label=0)
    recall_1 =  recall_score(y_test, y_pred, pos_label=1)
    
    return acc, recall_0, recall_1


full_dataset = read_full_dataset()
mike_dataset = read_mike_dataset()
treat_dataset = read_treat_dataset()

full_notrea_dataset = drop_trea(full_dataset)
mike_notrea_dataset = drop_trea(mike_dataset)

all_studies = list(set(full_dataset['study']))

for test_study in sorted(all_studies):
    print("study to test:", test_study)
    print("full:            ", *calc_results(full_dataset,        test_study, XGBClassifier()))
    print("full_notrea:     ", *calc_results(full_notrea_dataset, test_study, XGBClassifier()))
    print("")
    print("mike:            ", *calc_results(mike_dataset, test_study, XGBClassifier()))
    print("mike_svm:        ", *calc_results(mike_dataset, test_study, svm.SVC()))
    print("mike_logi:       ", *calc_results(mike_dataset, test_study, LogisticRegression(max_iter=1000)))
    print("mike_notrea:     ", *calc_results(mike_notrea_dataset, test_study, XGBClassifier()))
    print("mike_notrea_svm: ", *calc_results(mike_notrea_dataset, test_study, svm.SVC()))
    print("mike_notrea_logi:", *calc_results(mike_notrea_dataset, test_study, LogisticRegression(max_iter=1000)))
    print("")
    print("trea:            ", *calc_results(treat_dataset, test_study, XGBClassifier()))
    print("trea_svm:        ", *calc_results(treat_dataset, test_study, svm.SVC()))
    print("trea_logi:       ", *calc_results(treat_dataset, test_study, LogisticRegression(max_iter=1000)))
    print("")
    print("")
    print("")
        
