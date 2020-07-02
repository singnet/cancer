import pandas as pd
import os
import numpy as np
import random
import math 
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from collections import Counter



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

def count_to_str(y):
    c = Counter(y)
    return "count_01=%i/%i"%(c[0], c[1])
        
        
def print_stats_with_posoutcome(dataset, treat):
    dataset_t = dataset[treat].to_numpy()
    dataset_o = dataset['posOutcome'].to_numpy()
    
    
    c = Counter(zip(dataset_t, dataset_o))
    
    set_t = set(dataset_t)
    print(treat, set_t)
    if (len(set_t) == 1):
        return
    for t in set_t:
        N0 = c[(t,0)]
        N1 = c[(t,1)]
        print(t, N0, N1, N0 / (N1 + N0), N1 / (N1 + N0))
    print("")



treat_dataset = read_treat_dataset()
all_treats = ['radio', 'surgery', 'chemo', 'hormone']



all_studies = list(set(treat_dataset['study']))

for study in sorted(all_studies):
    study_dataset = treat_dataset.loc[treat_dataset['study'] == study]
    
    c = Counter(study_dataset['posOutcome'].to_numpy())
    print("==>", study, "count_01=%i/%i"%(c[0], c[1]))
    N0,N1 = c[0],c[1]
    print(N0 / (N1 + N0), N1 / (N1 + N0))

    for treat in all_treats:
        print_stats_with_posoutcome(study_dataset, treat)
    print("")
    print("")
