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



def read_coincide_types_dataset(path="./"):
    dataset = pd.read_csv(os.path.join(path, 'coincideTypes.csv'))
    treat_dataset = read_treat_dataset()
    return pd.merge(treat_dataset, dataset)

def format(cs):
    lens = [len(cs.loc[cs["pam_name"] == t]) for t in pam_types]
    proc = np.array(lens) / sum(lens)
    rez = ",".join(f"{p*100:.3g}%" for l,p in zip(lens, proc))
    return rez

coincide_types_dataset = read_coincide_types_dataset()
treat_dataset          = read_treat_dataset()

pam_types = list(sorted(set(coincide_types_dataset["pam_name"])))
studies = list(set(treat_dataset["study"]))

print("name,N," + ",".join(pam_types))
print(f"total,{len(coincide_types_dataset)},{format(coincide_types_dataset)}")
for study in studies:
    study_patients = treat_dataset.loc[treat_dataset["study"] == study]["patient_ID"]
    cs = coincide_types_dataset.loc[coincide_types_dataset["patient_ID"].isin(study_patients)]    
    print(f"{study},{len(study_patients)},{format(cs)}")
        
