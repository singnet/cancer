import pandas as pd
import os
import numpy as np
import random
import math 
from xgboost import XGBClassifier
from sklearn.metrics import recall_score
from sklearn.model_selection import RepeatedStratifiedKFold
from funs_balance import random_upsample_balance
from collections import defaultdict
from sklearn import svm
from sklearn.linear_model import LogisticRegression


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

def select_pam50(full_dataset):
    pam50 = ['ACTR3B', 'ANLN', 'BAG1', 'BCL2', 'BIRC5', 'BLVRA', 'CCNB1', 'CCNE1', 'CDC20', 'CDC6', 'NUF2', 'CDH3', 'CENPF', 'CEP55', 'CXXC5', 'EGFR', 'ERBB2', 'ESR1', 'EXO1', 'FGFR4', 'FOXA1', 'FOXC1', 'GPR160', 'GRB7', 'KIF2C', 'NDC80', 'KRT14', 'KRT17', 'KRT5', 'MAPT', 'MDM2', 'MELK', 'MIA', 'MKI67', 'MLPH', 'MMP11', 'MYBL2', 'MYC', 'NAT1', 'ORC6', 'PGR', 'PHGDH', 'PTTG1', 'RRM2', 'SFRP1', 'SLC39A6', 'TMEM45B', 'TYMS', 'UBE2C', 'UBE2T']
    selected_pam50 = list(set(pam50).intersection(set(full_dataset)))
    return full_dataset[selected_pam50 + ['study', 'posOutcome', 'patient_ID', 'pCR', 'RFS', 'DFS']]
    
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


def calc_results_for_fold(full_dataset, clf):
    
    X_train, X_test, y_train, y_test = get_balanced_split(full_dataset)
    
    clf.fit(X_train,y_train)
    
    y_pred      = clf.predict(X_test)
    
    acc = np.mean(y_pred == y_test)
    recall_0 =  recall_score(y_test, y_pred, pos_label=0)
    recall_1 =  recall_score(y_test, y_pred, pos_label=1)
    
    return acc, recall_0, recall_1
                    
def list_to_4g_str(l):
    return "  ".join("%.4g"%d for d in l)
        


full_dataset = read_full_dataset()
mike_dataset = read_mike_dataset()
treat_dataset = read_treat_dataset()

full_notrea_dataset = drop_trea(full_dataset)
mike_notrea_dataset = drop_trea(mike_dataset)
mike_pam50          = select_pam50(mike_dataset)
full_pam50          = select_pam50(full_dataset)


rez = defaultdict(list)
    
print_order = ["full", "full_notrea", "full_pam50", "mike", "mike_svm","mike_logi", 
               "mike_notrea", "mike_notrea_svm", "mike_notrea_logi",
               "mike_pam50", "mike_pam50_svm", "mike_pam50_logi",
               "trea", "trea_svm", "trea_logi"]
max_len_order = max(map(len,print_order))

for i in range(10):
    print("split ", i)
    rez["full"].append(       calc_results_for_fold(full_dataset, XGBClassifier()))    
    rez["full_notrea"].append(calc_results_for_fold(full_notrea_dataset, XGBClassifier()))
    rez["full_pam50"].append( calc_results_for_fold(full_pam50, XGBClassifier()))
    
    rez["mike"].append(       calc_results_for_fold(mike_dataset, XGBClassifier()))    
    rez["mike_svm"].append(   calc_results_for_fold(mike_dataset, svm.SVC()))
    rez["mike_logi"].append(  calc_results_for_fold(mike_dataset, LogisticRegression(max_iter=1000)))
    
    rez["mike_notrea"].append(     calc_results_for_fold(mike_notrea_dataset, XGBClassifier()))
    rez["mike_notrea_svm"].append( calc_results_for_fold(mike_notrea_dataset, svm.SVC()))
    rez["mike_notrea_logi"].append(calc_results_for_fold(mike_notrea_dataset, LogisticRegression(max_iter=1000)))
    
    rez["mike_pam50"].append(     calc_results_for_fold(mike_pam50, XGBClassifier()))
    rez["mike_pam50_svm"].append( calc_results_for_fold(mike_pam50, svm.SVC()))
    rez["mike_pam50_logi"].append(calc_results_for_fold(mike_pam50, LogisticRegression(max_iter=1000)))
    
    rez["trea"].append(     calc_results_for_fold(treat_dataset, XGBClassifier()))
    rez["trea_svm"].append( calc_results_for_fold(treat_dataset, svm.SVC()))
    rez["trea_logi"].append(calc_results_for_fold(treat_dataset, LogisticRegression(max_iter=1000)))

    
    for order in print_order:
        print(order, " "*(max_len_order - len(order)), ": ", list_to_4g_str(rez[order][-1]))

for order in print_order:
    print("==> ", order, " "*(max_len_order - len(order)), ": ", list_to_4g_str(np.mean(rez[order], axis=0)))
            
