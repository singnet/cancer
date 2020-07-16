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
from funs_common import *
import sys
from sklearn.preprocessing import OneHotEncoder
import itertools



def read_pam_types_num_dataset():    
    coincide_types_dataset = read_coincide_types_dataset()
    print(list(coincide_types_dataset))
    return coincide_types_dataset[["patient_ID", "study", 'pCR', 'RFS', 'DFS','radio','surgery','chemo','hormone', 'posOutcome', 'pam_cat']]

def drop_na(d, field):
    return d.loc[d[field].notna()]
    
        
def print_results_for_field(dataset, field):
    
    dataset = drop_na(dataset, field)
    
    X_full, y_full = prepare_full_dataset(dataset)
    
    rez = {i : defaultdict(int) for i in range(1,6)}
    for xi,yi in zip(X_full, y_full):
        rez[xi[4]][(tuple(xi[:4]), yi)] += 1
        
    total_sum_surv = 0
    for pam in range(1,6):
        used = set()
        max_prob = 0
        sum_N_major = 0
        
        all_N_out1_prob = []
        # maximal survival prob
        for treat,yi in sorted(rez[pam].keys()):
            if (treat not in used):
                used.add(treat)
                out0 = rez[pam][(treat, 0)]
                out1 = rez[pam][(treat, 1)]
                N = out0 + out1
                prob = out1 / (out0 + out1)
                all_N_out1_prob.append((N,out1,prob))
                
                if (N >= 10):                    
                    max_prob = max(prob, max_prob)
        for N,out1,prob in all_N_out1_prob:
            if (prob <= max_prob):
                total_sum_surv += max_prob * N
            else: # could be for small groub with N < 10
                total_sum_surv += out1
    print(sum(y_full) / len(y_full), total_sum_surv / len(y_full))
            
    
    

#full_dataset = read_full_dataset()
pam_types_cat_dataset = read_pam_types_num_dataset()



print("==> pCR")
print_results_for_field(pam_types_cat_dataset, "pCR")
print("")
print("")


print("==> RFS")
print_results_for_field(pam_types_cat_dataset, "RFS")
print("")
print("")


print("==> DFS")
print_results_for_field(pam_types_cat_dataset, "DFS")
print("")
print("")


print("==> posOutcome")
print_results_for_field(pam_types_cat_dataset, "posOutcome")
print("")
print("")

