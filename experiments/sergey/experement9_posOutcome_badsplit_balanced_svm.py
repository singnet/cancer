import pandas as pd
import os
import numpy as np
import random
import math 
from sklearn.metrics import roc_auc_score,f1_score,recall_score
from sklearn.model_selection import RepeatedStratifiedKFold
from funs_balance import random_upsample_balance
from sklearn import svm


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

def read_mike_dataset():
    join_dataset = pd.read_csv("ex15bmcMerged.csv")
    treat_dataset = read_treat_dataset()
    return pd.merge(treat_dataset, join_dataset)


def drop_trea(full_dataset):
    return full_dataset.drop(columns=['radio','surgery','chemo','hormone'])

def prepare_full_dataset(full_dataset):
    X = full_dataset.drop(columns=['study', 'patient_ID','pCR', 'RFS', 'DFS', 'posOutcome']).to_numpy()
    y_posOutcome = full_dataset['posOutcome'].to_numpy()
    return X, y_posOutcome


def prepare_dataset(full_dataset, study):
    study_dataset = full_dataset.loc[full_dataset['study'] == study]
    X = study_dataset.drop(columns=['study', 'patient_ID','pCR', 'RFS', 'DFS', 'posOutcome']).to_numpy(dtype = np.float)
    y_posOutcome = study_dataset['posOutcome'].to_numpy(dtype = np.float)
    return X, y_posOutcome
            
def get_balanced_split_for_study(full_dataset, study):
    X,y = prepare_dataset(full_dataset, study)
    

    kf = RepeatedStratifiedKFold(n_splits=5)
    (train_index, test_index) = next(kf.split(X,y))
    
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    X_train, y_train = random_upsample_balance(X_train, y_train)
    X_test, y_test   = random_upsample_balance(X_test, y_test)
    
    
    return X_train, X_test, y_train, y_test
        

def get_balanced_split(full_dataset):
    
    all_X_train, all_X_test, all_y_train, all_y_test = [],[],[],[]

    for study in list(set(full_dataset['study'])):
        X_train, X_test, y_train, y_test = get_balanced_split_for_study(full_dataset, study)
        all_X_train.append(X_train)
        all_X_test.append(X_test)
        all_y_train.append(y_train)
        all_y_test.append(y_test)
    return np.concatenate(all_X_train),  np.concatenate(all_X_test), np.concatenate(all_y_train), np.concatenate(all_y_test)


def calc_results_for_fold(full_dataset):
    
    X_train, X_test, y_train, y_test = get_balanced_split(full_dataset)
    
    
    clf = svm.SVC()
    clf.fit(X_train,y_train)
    
    y_pred      = clf.predict(X_test)
    
    acc = np.mean(y_pred == y_test)
    recall_0 =  recall_score(y_test, y_pred, pos_label=0)
    recall_1 =  recall_score(y_test, y_pred, pos_label=1)
    
    return acc, recall_0, recall_1



#full_dataset = read_full_dataset()
mike_dataset = read_mike_dataset()
treat_dataset = read_treat_dataset()

#full_notrea_dataset = drop_trea(full_dataset)
mike_notrea_dataset = drop_trea(mike_dataset)

#rez_full = []
rez_mike = []
rez_trea = []
#rez_notrea_full = []
rez_notrea_mike = []

for i in range(10):
    print("split ", i)
 #   rez_full.append(       calc_results_for_fold(full_dataset))    
 #   rez_notrea_full.append(calc_results_for_fold(full_notrea_dataset))    
    rez_mike.append(       calc_results_for_fold(mike_dataset))
    rez_notrea_mike.append(calc_results_for_fold(mike_notrea_dataset))
    rez_trea.append(       calc_results_for_fold(treat_dataset))

    
  #  print("full:        ", *rez_full[-1])
  #  print("full_notrea: ", *rez_notrea_full[-1])
    print("mike:        ", *rez_mike[-1])
    print("mike_notrea: ", *rez_notrea_mike[-1])
    print("trea:        ", *rez_trea[-1])

print("==> mean mike:       ", *np.mean(rez_mike, axis=0))
print("==> mean notrea_mike:", *np.mean(rez_notrea_mike, axis=0))
print("==> mean trea:       ", *np.mean(rez_trea, axis=0))
            
