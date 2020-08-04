import sys
sys.path.append('/usr/local/lib/python3/dist-packages/')

import sys
import os
import lzma
import random
from collections import defaultdict
import math

import numpy
import pandas

from util import *

cancer_data_dir = '/home/noskill/projects/cancer.old/data'
dataset_dict = load_merged_dataset(cancer_data_dir)

bmc = dataset_dict['bmc']
bmc = bmc.sort_values(by='patient_ID')
treatment = dataset_dict['treatment'].sort_values(by='patient_ID')
genes_features = dataset_dict['genes_features']
genes_features = genes_features.sort_values(by='patient_ID')


aggregated_treatment_columns = ['radio', 'surgery', 'chemo', 'hormone']
label_columns = ['pCR', 'RFS', 'DFS', 'posOutcome']
label_column = 'posOutcome'
label_column = 'RFS'
label_columns = [label_column]
genes_columns = genes_features.columns.to_list()[1:]
feature_columns = [x for x in xgboost_top_100 if x not in treatment_columns] + treatment_columns #xgboost_top_100 #genes_columns + treatment_columns # label_columns +  # pam50col #  +   + aggregated_treatment_columns
merged = dataset_dict['merged']
# filter rows with nans in label_column
merged = merged[~merged[label_column].isnull()]

bin_data = binarize_dataset(merged, genes_columns, feature_columns, to_letters=False)

subset_moses_features = label_columns + feature_columns
subset_moses_features = [x for x in subset_moses_features if x in bin_data.columns]

from util import study_mapping,split_by_study
res = defaultdict(list)
for study_name, study_id in study_mapping.items():
    train_data, train_labels, val_data, val_labels = next(split_by_study(bin_data,
                                                              subset_moses_features,
                                                              label_columns,
                                                              study=study_id,
                                                              to_numpy=False))
    print(study_name)
    print(val_data.shape)
    if not val_data.shape[0]:
        print('skipping {0}'.format(study_name))
        continue
    train_data.to_csv('/tmp/bin/cancer_bin_100_train_leave_{0}.csv'.format(study_name),
                      header=True, index=False)
    val_data.to_csv('/tmp/bin/cancer_bin_100_val_leave_{0}.csv'.format(study_name),
                    header=True, index=False)
