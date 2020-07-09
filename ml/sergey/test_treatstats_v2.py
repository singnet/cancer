import pandas as pd
import os
import numpy as np
import random
import math 
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from collections import Counter
from funs_common import read_treat_dataset



def read_alltreat_dataset(path="./"):
    return pd.read_csv(os.path.join(path, 'bcClinicalTable.csv'))


atreat_dataset = read_alltreat_dataset()
treat_dataset  = read_treat_dataset()


dataset = pd.merge(treat_dataset, atreat_dataset)

misc_fields = ['study', 'patient_ID', 'pCR', 'RFS', 'DFS', 'posOutcome', 'dbUniquePatientID', 'clinical_AJCC_stage', 'HER2 IHC', 'ESR1_preTrt', 'HER2_preTrt', 'ERbb2_preTrt',
               'DFS_months_or_MIN_months_of_DFS', 'relapseOneYearVsThreePlus', 'original_study_patient_ID', 'near_pCR',
               'relapseOneYearVsFivePlus', 'GEO_GSMID', 'tumor_stage_preTrt', 'age', 'preTrt_totalLymphNodes', 'tumor_size_cm_preTrt_preSurgery', 'preTrt_numPosLymphNodes',
               'ER_preTrt', 'site_ID_preprocessed', 'hist_grade', 'path_diagnosis', 'preTrt_lymph_node_status', 'PR_preTrt', 'preTrt_posDichLymphNodes', 'dead', 'died_from_cancer_if_dead', 
               'path', 'metastasis', 'pam50', 'race', 'ER_percentage_preTrt', 'HER2_IHC_score_preTrt', 'HER2_fish_cont_score_preTrt', 'OS', 'OS_months_or_MIN_months_of_OS',
               'chemosensitivity_prediction', 'OS_up_until_death', 'RCB', 'RFS_months_or_MIN_months_of_RFS', 'age_bin', 'intarvenous', 'intramuscular', 'menopausal_status', 
               'metastasis_months', 'nuclear_grade_preTrt', 'oral', 'other', 'p53', 'p53_mutation', 'pCR_spectrum', 'postmenopausal_only', 'top2atri_preTrt', 'topoihc_preTrt']
t_fields    = list(set(dataset) - set(misc_fields))


all_studies = list(set(dataset['study']))


good = set()

for study in sorted(all_studies):
    study_dataset = dataset.loc[treat_dataset['study'] == study]
    
    print("==>", study, len(study_dataset))
    for f in t_fields:
        if (study_dataset[f].nunique(dropna = False) > 1):
            good.add(f)
            print(f, "\n", study_dataset[f].value_counts(dropna=False), "\n\n")
    print("\n\n")
    
print(sorted(good))
