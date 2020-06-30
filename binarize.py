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

cancer_data_dir = '/home/noskill/projects/cancer/data'
dump_dir = os.path.join(cancer_data_dir, 'bcDump/example15bmc')
clinical_table_path = os.path.join(cancer_data_dir, 'bcClinicalTable.csv')
merged_path = os.path.join(dump_dir, 'ex15bmcMerged.csv.xz')
bmc_all_path = os.path.join(dump_dir, 'bmc15mldata1.csv')

dtype = {'DFS': pandas.Int64Dtype(),
         'pCR': pandas.Int64Dtype(),
         'RFS': pandas.Int64Dtype(),
         'DFS': pandas.Int64Dtype(),
         'posOutcome': pandas.Int64Dtype()}


# load averaged treatment table
bmc = pandas.read_csv(bmc_all_path, dtype=dtype, converters=converters)
bmc = bmc.sort_values(by='patient_ID')


# load detailed treatment
treatment = pandas.read_csv(clinical_table_path, converters=converters).sort_values(by='patient_ID')
treatment = treatment[treatment.patient_ID.isin(bmc.patient_ID)]


# load genes expression data
gene_expression = pandas.read_csv(lzma.open(merged_path))

genes_features = gene_expression[gene_expression.patient_ID.isin(bmc.patient_ID)]
genes_features = genes_features.sort_values(by='patient_ID')

aggregated_treatment_columns = ['radio', 'surgery', 'chemo', 'hormone']
label_columns = ['pCR', 'RFS', 'DFS', 'posOutcome']
label_columns = ['posOutcome']
genes_columns = genes_features.columns.to_list()[1:]
feature_columns = xgboost_top_100 #genes_columns + treatment_columns # label_columns +  # pam50col #  +   + aggregated_treatment_columns


## merge genes expression + averaged treatment + detailed treatment

merged = pandas.merge(genes_features, bmc, left_on='patient_ID', right_on='patient_ID')
merged = pandas.merge(merged, treatment, left_on='patient_ID', right_on='patient_ID')
merged.insert(0, 'row_num', range(0,len(merged)))

bin_data = binarize_dataset(merged, genes_columns, feature_columns, to_letters=False)

bin_data.drop('patient_ID', axis=1, inplace=True)

print(bin_data.taxaneGeneral.max())
subset_moses_features = ['posOutcome'] + feature_columns
bin_data[subset_moses_features].to_csv('/tmp/cancer_bin_100.csv', header=True, index=False)

