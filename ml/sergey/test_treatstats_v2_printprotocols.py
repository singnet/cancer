import pandas as pd
import os
import numpy as np
from collections import defaultdict, Counter
from funs_common import read_treat_dataset, read_pam_types_cat_dataset



def read_alltreat_dataset(path="./"):
    return pd.read_csv(os.path.join(path, 'bcClinicalTable.csv'))

def count_results_stats(dataset, v_list):
    rez = []
    for v in v_list:
        if (np.isnan(dataset[v].to_numpy()[0])):
            rez.append("%s=-"%v)
        else:
            c = Counter(dataset[v])
            rez.append(f"{v}={c[1]}={100*c[1]/(c[0]+c[1]):.1f}%")
    return "  ".join(rez)
        

atreat_dataset = read_alltreat_dataset()
treat_dataset  = read_treat_dataset()
pam_types_cat_dataset = read_pam_types_cat_dataset()

print(list(pam_types_cat_dataset))

dataset = pd.merge(treat_dataset, atreat_dataset)
dataset = pd.merge(dataset, pam_types_cat_dataset)


goodf = ['methotrexate', 'docetaxel', 'surgery', 'radio', 'estrogen_receptor_blocker', 'tamoxifen', 'hormone', 'anti_estrogen', 'doxorubicin', 'letrozole', 'taxaneGeneral', 'aromatase_inhibitor', 'anti_HER2', 'paclitaxel', 'neoadjuvant_or_adjuvant', 'anthracycline', 'estrogen_receptor_blocker_and_stops_production', 'fluorouracil', 'anastrozole', 'taxane', 'cyclophosphamide', 'capecitabine', 'trastuzumab', 'treatment_protocol_number', 'no_treatment', 'epirubicin', 'chemo']

all_studies = list(set(dataset['study']))


for study in sorted(all_studies):
    study_dataset = dataset.loc[treat_dataset['study'] == study]
    print(study, len(study_dataset))

    study_dataset = study_dataset[list(goodf) + ['pCR', "RFS", "DFS", "posOutcome"] + ['Basal', 'Her2', 'LumA', 'LumB', 'Normal']]
    for protocol_name, protocol_dataset in study_dataset.groupby("treatment_protocol_number"):
        
        print("protocol", protocol_name, len(protocol_dataset))
        print(count_results_stats(protocol_dataset, ['pCR', "RFS", "DFS", "posOutcome"]))
        print(count_results_stats(protocol_dataset, ['Basal', 'Her2', 'LumA', 'LumB', 'Normal']))
        for f in goodf:
            v = protocol_dataset[f].to_numpy()[0]
            if (v != 0):
                print(f, v)
        print("")
    print("")
