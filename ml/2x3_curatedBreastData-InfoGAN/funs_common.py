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

def read_treat_dataset(path="./data"):
    return pd.read_csv(os.path.join(path, 'bmc15mldata1.csv'), converters=dict(surgery = convert_surgery, posOutcome = convert_posOutcome))


def read_coincide_types_dataset(path="./"):
    dataset = pd.read_csv(os.path.join(path, 'coincideTypes.csv'))
    treat_dataset = read_treat_dataset()
    return pd.merge(treat_dataset, dataset)

def read_alltreat_dataset(path="./"):
    atreat_dataset = pd.read_csv(os.path.join(path, 'bcClinicalTable.csv'))
    treat_dataset  = read_treat_dataset()
    return pd.merge(treat_dataset, atreat_dataset)
    

def read_pam_types_cat_dataset():
    coincide_types_dataset = read_coincide_types_dataset()
    keep = coincide_types_dataset[["patient_ID", "study", 'pCR', 'RFS', 'DFS','radio','surgery','chemo','hormone', 'posOutcome']]
    return pd.concat([keep, pd.get_dummies(coincide_types_dataset["pam_name"])], axis=1)
        
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

def read_combat_dataset():
    #join_dataset = pd.read_csv("merged-combat15.csv")
    join_dataset = pd.read_csv("data/ex15bmcMerged.csv.xz")
    treat_dataset = read_treat_dataset()
    return pd.merge(treat_dataset, join_dataset)
    

def drop_trea(full_dataset):
    return full_dataset.drop(columns=['radio','surgery','chemo','hormone'])

def get_pam50_list():
    return ['ACTR3B', 'ANLN', 'BAG1', 'BCL2', 'BIRC5', 'BLVRA', 'CCNB1', 'CCNE1', 'CDC20', 'CDC6', 'NUF2', 'CDH3', 'CENPF', 'CEP55', 'CXXC5', 'EGFR', 'ERBB2', 'ESR1', 'EXO1', 'FGFR4', 'FOXA1', 'FOXC1', 'GPR160', 'GRB7', 'KIF2C', 'NDC80', 'KRT14', 'KRT17', 'KRT5', 'MAPT', 'MDM2', 'MELK', 'MIA', 'MKI67', 'MLPH', 'MMP11', 'MYBL2', 'MYC', 'NAT1', 'ORC6', 'PGR', 'PHGDH', 'PTTG1', 'RRM2', 'SFRP1', 'SLC39A6', 'TMEM45B', 'TYMS', 'UBE2C', 'UBE2T']
    
def select_pam50(full_dataset):
    pam50 = get_pam50_list()
    selected_pam50 = list(set(pam50).intersection(set(full_dataset)))
    return full_dataset[selected_pam50 + ['study', 'posOutcome', 'patient_ID', 'pCR', 'RFS', 'DFS']]
    
def prepare_full_dataset(full_dataset, y_field = 'posOutcome'):
    X = full_dataset.drop(columns=['study', 'patient_ID','pCR', 'RFS', 'DFS', 'posOutcome']).to_numpy()
    y_posOutcome = full_dataset[y_field].to_numpy()
    return X, y_posOutcome


def prepare_dataset(full_dataset, study, y_field = 'posOutcome'):
    study_dataset = full_dataset.loc[full_dataset['study'] == study]
    X = study_dataset.drop(columns=['study', 'patient_ID','pCR', 'RFS', 'DFS', 'posOutcome']).to_numpy(dtype = np.float)
    y_posOutcome = study_dataset[y_field].to_numpy(dtype = np.float)
    return X, y_posOutcome

def prepare_datasets(full_dataset, studies):
    Xys = [prepare_dataset(full_dataset, study ) for study in studies]
    Xs, ys = zip(*Xys)
    return np.concatenate(Xs), np.concatenate(ys)
            
            
def list_to_4g_str(l):
    return "  ".join("%.4g"%d for d in l)
    
def list2d_to_4g_str_pm(l):
    return " ".join("%.4g (±%.4g)  "%(mean,sigma) for mean,sigma in zip(np.mean(l, axis=0), np.std(l, axis=0)))
    
