import pandas as pd
import os
import numpy as np
from collections import defaultdict, Counter
from funs_common import read_treat_dataset



def read_alltreat_dataset(path="./"):
    return pd.read_csv(os.path.join(path, 'bcClinicalTable.csv'))

   
   
atreat_dataset = read_alltreat_dataset()
treat_dataset  = read_treat_dataset()


dataset = pd.merge(treat_dataset, atreat_dataset)

goodf = ['methotrexate', 'docetaxel', 'surgery', 'radio', 'estrogen_receptor_blocker', 'tamoxifen', 'hormone', 'anti_estrogen', 'doxorubicin', 'letrozole', 'taxaneGeneral', 'aromatase_inhibitor', 'anti_HER2', 'paclitaxel', 'neoadjuvant_or_adjuvant', 'anthracycline', 'estrogen_receptor_blocker_and_stops_production', 'fluorouracil', 'anastrozole', 'taxane', 'cyclophosphamide', 'capecitabine', 'trastuzumab', 'treatment_protocol_number', 'no_treatment', 'epirubicin', 'chemo']

all_studies = list(set(dataset['study']))


for study in sorted(all_studies):
    print(study)
    study_dataset = dataset.loc[treat_dataset['study'] == study]
    study_dataset = study_dataset[list(goodf)]
    for protocol_name, protocol_dataset in study_dataset.groupby("treatment_protocol_number"):
        a = protocol_dataset.to_numpy()
        a = np.nan_to_num(a, nan=-100)
        n_diff = np.sum(a[0] != a )
        if (n_diff != 0):
            print(protocol_name, n_diff)
#        assert (a[0] == a).all()

