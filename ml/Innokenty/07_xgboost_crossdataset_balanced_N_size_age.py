import pandas as pd
import os
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, recall_score
from funs_balance import random_upsample_balance
from funs_common_balanced import get_balanced_split
from funs_common import *
from collections import defaultdict
import sys

output = open("07_results.txt", "w")


def get_balanced_study(full_dataset, study):
    X,y = prepare_dataset(full_dataset, study)
    return random_upsample_balance(X, y)

def get_balanced_studies_except_test_study(full_dataset, test_study):
    all_studies = list(set(full_dataset['study']))
    all_Xy = [get_balanced_study(full_dataset, study) for study in all_studies if (study != test_study)]
    all_X, all_y = zip(*all_Xy)
    return np.concatenate(all_X),  np.concatenate(all_y)

def calc_results_for_test_study(full_dataset, test_study, clf):
    X_train, y_train = get_balanced_studies_except_test_study(full_dataset, test_study)
    X_test, y_test = get_balanced_study(full_dataset, test_study)

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    acc = np.mean(y_pred == y_test)
    recall_0 = recall_score(y_test, y_pred, pos_label=0)
    recall_1 = recall_score(y_test, y_pred, pos_label=1)

    return acc, recall_0, recall_1

# path = "/mnt/fileserver/shared/datasets/biodata/MetaGX/merged/"
path = "/home/daddywesker/bc_data_processor/MetaGX/merged/"
datasets = [ "noNormBatchedMergedAnd5k.csv", "noNormBatchedMergedAnd10k.csv", "noNormMergedAnd5k.csv", "noNormMergedAnd10k.csv",
            "rlNormBatchedMergedAnd5k.csv", "rlNormBatchedMergedAnd10k.csv", "rlNormMergedAnd5k.csv", "rlNormMergedAnd10k.csv",
            "noNormBatchedMergedOr5k.csv", "noNormBatchedMergedOr10k.csv", "noNormMergedOr5k.csv", "noNormMergedOr10k.csv",
            "rlNormBatchedMergedOr5k.csv", "rlNormBatchedMergedOr10k.csv", "rlNormMergedOr5k.csv", "rlNormMergedOr10k.csv"]




print_order = ["full", "full_notrea"]

for dataset in datasets:
    metagx_dataset = read_metagx_dataset_with_stuff(path+dataset, ['study', 'sample_name', 'tumor_size', 'N', 'age_at_initial_pathologic_diagnosis', 'grade' ])
    metagx_dataset = drop_na(metagx_dataset, "posOutcome")
    all_studies = list(set(metagx_dataset['study']))
    print(metagx_dataset.columns)
    output.write(str(metagx_dataset.columns) + "\n")
    metagx_notrea_dataset = drop_trea(metagx_dataset)

    print("==> posOutcome "+str(dataset))
    output.write("==> posOutcome "+str(dataset) + "\n")
    max_len_order = max(map(len, print_order))
    for test_study in sorted(all_studies):
        print("study to test:", test_study)

        # we make random upsampling, so results might be different
        rez = defaultdict(list)

        for i in range(5):
            rez["full"].append(calc_results_for_test_study(metagx_dataset, test_study, XGBClassifier()))
            rez["full_notrea"].append(calc_results_for_test_study(metagx_notrea_dataset, test_study, XGBClassifier()))
            for order in print_order:
                print(order, " " * (max_len_order - len(order)), ": ", list_to_4g_str(rez[order][-1]))
            print("")
            sys.stdout.flush()

        for order in print_order:
            output.write("==> "+str(order)+ " " * (max_len_order - len(order))+ ": "+str(list_to_4g_str(np.mean(rez[order], axis=0))))
            output.write("\n")
            print("==> ", order, " " * (max_len_order - len(order)), ": ", list_to_4g_str(np.mean(rez[order], axis=0)))
        print("")
        print("")
        output.write("\n")
        output.write("\n")

output.close()
