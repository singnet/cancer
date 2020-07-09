import pandas as pd
import os
import numpy as np
from collections import defaultdict, Counter
from funs_common import read_treat_dataset



def read_alltreat_dataset(path="./"):
    return pd.read_csv(os.path.join(path, 'bcClinicalTable.csv'))

def check_protocol_number(dataset):
    d1 = dataset['treatment_protocol_number'].to_numpy(dtype = np.int)
    d2 = dataset['study_specific_protocol_number'].to_numpy(dtype = np.int)
    assert all(d1 == d2)
    
    
def convert_to_str(f, d):
    if (f == "treatment_protocol_number"):
        return ' / '.join(map(str,reversed(sorted(d.values()))))
    assert len(d.items()) == 2
    if (f == "neoadjuvant_or_adjuvant"):
        return " / ".join(map(lambda p: f"{(p[0])}:{p[1]}", sorted(d.items())))
    else:
        return " / ".join(map(lambda p: f"{int(p[0])}:{p[1]}", sorted(d.items())))


atreat_dataset = read_alltreat_dataset()
treat_dataset  = read_treat_dataset()


dataset = pd.merge(treat_dataset, atreat_dataset)

check_protocol_number(dataset)

misc_fields = ['study', 'patient_ID', 'pCR', 'RFS', 'DFS', 'posOutcome', 'dbUniquePatientID', 'clinical_AJCC_stage', 'HER2_IHC', 'ESR1_preTrt', 'HER2_preTrt', 'ERbb2_preTrt',
               'DFS_months_or_MIN_months_of_DFS', 'relapseOneYearVsThreePlus', 'original_study_patient_ID', 'near_pCR',
               'relapseOneYearVsFivePlus', 'GEO_GSMID', 'tumor_stage_preTrt', 'age', 'preTrt_totalLymphNodes', 'tumor_size_cm_preTrt_preSurgery', 'preTrt_numPosLymphNodes',
               'ER_preTrt', 'site_ID_preprocessed', 'hist_grade', 'path_diagnosis', 'preTrt_lymph_node_status', 'PR_preTrt', 'preTrt_posDichLymphNodes', 'dead', 'died_from_cancer_if_dead', 
               'path', 'metastasis', 'pam50', 'race', 'ER_percentage_preTrt', 'HER2_IHC_score_preTrt', 'HER2_fish_cont_score_preTrt', 'OS', 'OS_months_or_MIN_months_of_OS',
               'chemosensitivity_prediction', 'OS_up_until_death', 'RCB', 'RFS_months_or_MIN_months_of_RFS', 'age_bin', 'intarvenous', 'intramuscular', 'menopausal_status', 
               'metastasis_months', 'nuclear_grade_preTrt', 'oral', 'other', 'p53', 'p53_mutation', 'pCR_spectrum', 'postmenopausal_only', 'top2atri_preTrt', 
               'topoihc_preTrt'] + ["study_specific_protocol_number", "surgery_type", "radiotherapyClass", "hormone_therapyClass", "chemotherapy", "chemotherapyClass"]
t_fields    = list(set(dataset) - set(misc_fields))


all_studies = list(set(dataset['study']))


goodf = set()
goods = set()

rez = defaultdict(dict)

for study in sorted(all_studies):
    study_dataset = dataset.loc[treat_dataset['study'] == study]
    
    for f in t_fields:
        if (study_dataset[f].nunique(dropna = False) > 1):
            goodf.add(f)
            goods.add(study)
            rez[study][f] = study_dataset[f].value_counts(dropna=False)

rezd = { f:[] for f in goodf}
rezd["study"] = []


for study in sorted(goods):
    rezd['study'].append(study)
    for f in goodf:
        if (f in rez[study]):
            rezd[f].append(convert_to_str(f, dict(rez[study][f])))
        else:
            rezd[f].append("")

pd.DataFrame(rezd, columns = ["study"] + sorted(goodf)).to_csv("test_treatstats_v2_tab.csv")


#check that in each protocol patients indeed have the same treatments

#print(list(goodf))                                                        
