import os
import numpy as np
import random
import math
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import pandas as pd
from gene_combination_generator import CombinationsGenerator
from utils import *


labels = read_treat_dataset("/home/oleg/Work/bc_data_procesor/git/bc_data_procesor/")

X_lab, y_pos_out = prepare_full_dataset(labels)

num_of_combo_items = 3

comb_gen = CombinationsGenerator("ex15bmcMerged70genes.csv")
datasets, gene_names_comb = comb_gen.generate(num_of_combo_items)
print(datasets[0].shape)

clf1 = XGBClassifier()
clf2 = LogisticRegression()
clf_map = {"xgboost": clf1}
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True)
results_folder = "without_treatment/"+str(num_of_combo_items)+"_genes/"
for k, v in clf_map.items():
    print("::::::::::::::::::::"+k+"::::::::::::::::::::")
    best_gene_combo = []
    best_auc_2_f_avg = 0.
    with open(results_folder+k+"_results.txt", 'a') as out:
        for X_full, gene_pair in zip(datasets, gene_names_comb):
            print('========================', gene_pair, '==============================')
            out.write(str(gene_pair))
            out.write('\n')
            auc_2_f_avg = 0.
            for i, (train_index, test_index) in enumerate(kf.split(X_full)):
                acc_f, auc_1_f, auc_2_f = calc_results_for_fold(X_full, y_pos_out, train_index, test_index, v)
                print("full: ", acc_f, auc_1_f, auc_2_f)
                out.write(str([acc_f, auc_1_f, auc_2_f]))
                out.write('\n')
                auc_2_f_avg += auc_2_f
            auc_2_f_avg /= n_splits
            best_gene_combo = gene_pair if auc_2_f_avg > best_auc_2_f_avg else best_gene_combo
            best_auc_2_f_avg = auc_2_f_avg if auc_2_f_avg > best_auc_2_f_avg else best_auc_2_f_avg
        print(best_gene_combo, best_auc_2_f_avg)
        out.write(str(best_gene_combo))
        out.write('\n')
        out.write(str(best_auc_2_f_avg))
        out.close()

print(X_lab.shape)
print(y_pos_out.shape)