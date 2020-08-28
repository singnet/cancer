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

import metagx_util
from util import *
from util import split_by_study


def get_curated():
    cancer_data_dir = '/home/noskill/projects/cancer/data/curatedBreastData'
    dataset_dict = load_curated(cancer_data_dir)
    return dataset_dict


def get_metagx():
    cancer_data_dir = '/home/noskill/projects/cancer/data/metaGxBreast'
    data = metagx_util.load_metagx_dataset(cancer_data_dir, 5000)
    return data


def main():
    dataset_type = 'metagx'
    if dataset_type == 'metagx':
        dataset_dict = get_metagx()
    elif dataset_type == 'curated':
        dataset_dict = get_curated()

    genes_columns = dataset_dict['genes_columns']
    if dataset_type == 'metagx':
        feature_columns = genes_columns + list(metagx_util.treatment_columns_metagx)
        from metagx_util import metagx_study_mapping
        study_mapping = metagx_study_mapping
    elif dataset_type == 'curated':
        from util import study_mapping
        aggregated_treatment_columns = ['radio', 'chemo', 'hormone']
        feature_columns = genes_columns + aggregated_treatment_columns# [x for x in xgboost_top_100 if x not in treatment_columns] + treatment_columns #xgboost_top_100 # + treatment_columns # label_columns +  # pam50col #  +   + aggregated_treatment_columns
    label_columns = ['pCR', 'RFS', 'DFS', 'posOutcome']
    label_column = 'pCR'
    label_column = 'posOutcome'
    label_columns = [label_column]
    merged = dataset_dict['merged']
    # filter rows with nans in label_column
    merged = merged[~merged[label_column].isnull()]
    if dataset_type == 'metagx':
       merged = merged[merged.posOutcome.isin([-1, 1])]

    bin_data = binarize_dataset(merged, genes_columns, feature_columns, to_letters=False)

    subset_moses_features = label_columns + feature_columns
    subset_moses_features = [x for x in subset_moses_features if x in bin_data.columns]
    print('target labels: {0}'.format(label_columns))

    res = defaultdict(list)
    studies = bin_data.study.unique()
    for study_name, study_id in study_mapping.items():
        if study_id not in studies:
            continue
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
        os.makedirs('/tmp/bin', exist_ok=True)
        l = train_data.shape[1]
        train_data.to_csv('/tmp/bin/cancer_bin_{0}_train_leave_study_{1}.csv'.format(l, study_name),
                          header=True, index=False)
        val_data.to_csv('/tmp/bin/cancer_bin_{0}_val_leave_study_{1}.csv'.format(l, study_name),
                        header=True, index=False)

if __name__ == '__main__':
   main()

