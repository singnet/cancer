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

def read_metagx_ml_data():
    return pd.read_csv('/home/daddywesker/bc_data_processor/MetaGX/mldata.csv')

def read_metagx_covar_data(path="./"):
    return pd.read_csv(os.path.join(path, '/mnt/fileserver/shared/datasets/biodata/MetaGX/metaGXcovarTable.csv'))

# def read_metagx_additional_data(columns_to_left, path="./", ):
#     covar = pd.read_csv(os.path.join(path, '/home/daddywesker/bc_data_processor/MetaGX/metaGXcovarTable.csv'))
#     return covar.drop(covar.columns.difference(columns_to_left), 1)
    
def read_metagx_dataset(path):
    join_dataset = pd.read_csv(path)
    treat_dataset = read_metagx_ml_data()
    return pd.merge(treat_dataset, join_dataset)

def read_metagx_dataset_with_stuff(path, columns_to_left):
    join_dataset = pd.read_csv(path)
    treat_dataset = read_metagx_ml_data()
    additional_dataset = read_metagx_additional_data(columns_to_left)
    temp_dataset = pd.merge(treat_dataset, join_dataset, left_on="sample_name", right_on="sample_name")
    return pd.merge(temp_dataset, additional_dataset, left_on="sample_name", right_on="sample_name")

def drop_trea(full_dataset):
    return full_dataset.drop(columns=['chemo','hormone'])

def get_pam50_list():
    return ['ACTR3B', 'ANLN', 'BAG1', 'BCL2', 'BIRC5', 'BLVRA', 'CCNB1', 'CCNE1', 'CDC20', 'CDC6', 'NUF2', 'CDH3', 'CENPF', 'CEP55', 'CXXC5', 'EGFR', 'ERBB2', 'ESR1', 'EXO1', 'FGFR4', 'FOXA1', 'FOXC1', 'GPR160', 'GRB7', 'KIF2C', 'NDC80', 'KRT14', 'KRT17', 'KRT5', 'MAPT', 'MDM2', 'MELK', 'MIA', 'MKI67', 'MLPH', 'MMP11', 'MYBL2', 'MYC', 'NAT1', 'ORC6', 'PGR', 'PHGDH', 'PTTG1', 'RRM2', 'SFRP1', 'SLC39A6', 'TMEM45B', 'TYMS', 'UBE2C', 'UBE2T']
    
def select_pam50(full_dataset):
    pam50 = get_pam50_list()
    selected_pam50 = list(set(pam50).intersection(set(full_dataset)))
    return full_dataset[selected_pam50 + ['study', 'posOutcome', 'patient_ID', 'pCR', 'RFS', 'DFS']]
    
def prepare_full_dataset(full_dataset, y_field = 'posOutcome'):
    X = full_dataset.drop(columns=['sample_name', 'alt_sample_name', 'recurrence', 'posOutcome', 'study'], errors='ignore').to_numpy()
    y_posOutcome = full_dataset[y_field].to_numpy()
    return X, y_posOutcome


def prepare_dataset(full_dataset, study, y_field = 'posOutcome'):
    study_dataset = full_dataset.loc[full_dataset['study'] == study]
    X = study_dataset.drop(columns=['sample_name', 'alt_sample_name', 'recurrence', 'posOutcome', 'study']).to_numpy(dtype = np.float)
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
    
def drop_na(d, field):
    return d.loc[d[field].notna()]
