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
from funs_common import read_alltreat_dataset, read_combat_dataset, prepare_full_dataset, list2d_to_4g_str_pm, drop_trea, list_to_4g_str, read_coincide_types_dataset, mutual_information
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
    # x = pam50
    # y = outcome
    p_x_y = dict()
    p_x = dict()
    p_y = dict()
    print('bias {}'.format(y.mean()))
    p_y[1] = y.mean()
    p_y[0] = 1 - p_y[1]
    for t in range(1,6):
        n0 = c[(t,0)]
        n1 = c[(t,1)]
        n01 = n0 + n1
        surv_frac = 0
        if n01 > 0:
            surv_frac = n1 / n01

        n_frac = n01 / len(y)
        p_x_y[(t, 0)] = n0 / len(y)
        p_x_y[(t, 1)] = n1 / len(y)
        p_x[t] = n_frac

        print(f"pam_type, n01, n_frac, surv_frac: {t}  {n01:3}  {n_frac*100:5.0f}%  {surv_frac*100:5.0f}%")
    mutual_info = mutual_information(p_x_y, p_x, p_y)
    print("mutual information {0}".format(mutual_info))


def print_set(dataset, set1, y_field):
    X_set1, y_set1 = prepare_full_dataset(dataset.loc[dataset['patient_ID'].isin(set1)],
            y_field)
    print_counter(X_set1[:,0], y_set1)

def print_results(dataset, set1, set2, y_field='posOutcome'):
    print("set1")
    print_set(dataset, set1, y_field)
    print("set2")
    print_set(dataset, set2, y_field)


atreat_dataset = read_alltreat_dataset()
dataset = read_pam_types_num_dataset()
notrea_dataset = drop_trea(dataset)


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
print()
print('posOutcome')
print(notrea_dataset.shape)
posOut = atreat_dataset.patient_ID
print_set(notrea_dataset, posOut, 'posOutcome')
print()
print('pCR')
pcr = atreat_dataset[~atreat_dataset.pCR.isnull()].patient_ID
print_set(notrea_dataset, pcr, 'pCR')
print()
print('DFS')
dfs = atreat_dataset[~atreat_dataset.DFS.isnull()].patient_ID
print_set(notrea_dataset, dfs, 'DFS')
print()
print('RFS')
rfs = atreat_dataset[~atreat_dataset.RFS.isnull()].patient_ID
print_set(notrea_dataset, rfs, 'RFS')
