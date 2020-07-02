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
from funs_common import prepare_dataset


def get_balanced_split_for_study(full_dataset, study, y_field = "posOutcome"):
    X,y = prepare_dataset(full_dataset, study, y_field = y_field)
    
    
    kf = RepeatedStratifiedKFold(n_splits=5)
    (train_index, test_index) = next(kf.split(X,y))
    
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    X_train, y_train = random_upsample_balance(X_train, y_train)
    X_test, y_test   = random_upsample_balance(X_test, y_test)
    
    return X_train, X_test, y_train, y_test
                                

def get_balanced_split(full_dataset, y_field = 'posOutcome'):
    
    all_X_train, all_X_test, all_y_train, all_y_test = [],[],[],[]
    
    for study in list(set(full_dataset['study'])):
        X_train, X_test, y_train, y_test = get_balanced_split_for_study(full_dataset, study, y_field)
        all_X_train.append(X_train)
        all_X_test.append(X_test)
        all_y_train.append(y_train)
        all_y_test.append(y_test)
    return np.concatenate(all_X_train),  np.concatenate(all_X_test), np.concatenate(all_y_train), np.concatenate(all_y_test)


def get_balanced_study(full_dataset, study):
    X,y = prepare_dataset(full_dataset, study)
    return random_upsample_balance(X, y)
        
def get_balanced_studies_except_test_study(full_dataset, test_study):
    all_studies = list(set(full_dataset['study']))
    all_Xy = [get_balanced_study(full_dataset, study) for study in all_studies if (study != test_study)]
    all_X, all_y = zip(*all_Xy)
    return np.concatenate(all_X),  np.concatenate(all_y)
