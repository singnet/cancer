import sys
sys.path.append('/usr/local/lib/python3/dist-packages/')
import pickle
import os
import lzma
import random
from collections import defaultdict
import math
import numpy
import pandas
import xgboost as xgb
import sklearn
from catboost import Pool, CatBoostClassifier
from sklearn.decomposition import PCA, FastICA, SparsePCA
import catboost
from data_util.util import *


def main():
    cancer_data_dir = '/home/noskill/projects/cancer/data/curatedBreastData'
    embed_dir = '/home/noskill/projects/cancer/data/embed'
    dataset_dict = load_curated(cancer_data_dir, version='empirical_bayes')
    bmc = dataset_dict['bmc']
    bmc = bmc.sort_values(by='patient_ID')
    treatment = dataset_dict['treatment'].sort_values(by='patient_ID')
    genes_features = dataset_dict['genes_features']
    genes_features = genes_features.sort_values(by='patient_ID')
    with open(os.path.join(embed_dir, 'test12_getemb_for_anatoly_PTE.p'), 'rb') as f:
        embedings = pickle.load(f)
    drug_to_id = dict(cetuximab = ",PubChemSID:46507042",
        methotrexate="ChEBI:44185",
        letrozole="ChEBI:6413",
        gefitinib="ChEBI:49668",
        anastrozole="ChEBI:2704",
        cyclophosphamide="ChEBI:4027",
        paclitaxel="ChEBI:45863",
        fluorouracil="ChEBI:46345",
        capecitabine="ChEBI:31348",
        docetaxel="ChEBI:4672",
        doxorubicin="ChEBI:28748",
        tamoxifen="ChEBI:41774",
        carboplatin="ChEBI:31355",
        fulvestrant="ChEBI:31638",
        trastuzumab="PubChemSID:46507516",
        epirubicin="ChEBI:47898",
        anthracycline="ChEBI:135779")
    mean_treatments = []
    for i in range(len(treatment)):
        row = treatment.iloc[i]
        treatments = []
        for drug_name, drug_id in drug_to_id.items():
            if row[drug_name]:
                treatments.append(embedings[drug_id])
        if treatments:
            mean_treatments.append(numpy.mean(treatments, axis=0))
        else:
            mean_treatments.append(numpy.zeros(50))
    mean_treatments = numpy.array(mean_treatments)
    merged = dataset_dict['merged']
    for i in range(50):
        merged.insert(5, 'treatment{0}'.format(i), mean_treatments[:, i])
    pam50col = genes_features.columns[genes_features.columns.isin(pam50).nonzero()[0]].to_list()
    aggregated_treatment_columns = ['radio', 'chemo', 'hormone', 'surgery']
    label_columns = ['pCR', 'RFS', 'DFS', 'RFSorDFS' 'posOutcome']
    genes_columns = genes_features.columns.to_list()[1:]
    treatment_embed = ['treatment{0}'.format(i) for i in range(50)]
    feature_columns = genes_columns  + aggregated_treatment_columns# +treatment_embed#+ + aggregated_treatment_columns # genes_columns aggregated_treatment_columns #  +  #  # label_columns +  # pam50col #  +   + aggregated_treatment_columns
    merged.insert(0, 'RFSorDFS', [numpy.nan,] * len(merged) )
    merged.loc[~merged['RFS'].isnull(), 'RFSorDFS'] = merged.loc[~merged['RFS'].isnull(), 'RFS']
    merged.loc[~merged['DFS'].isnull(), 'RFSorDFS'] = merged.loc[~merged['DFS'].isnull(), 'DFS']
    rfs = merged.loc[~merged['RFSorDFS'].isnull(), 'RFS']
    dfs = merged.loc[~merged['RFSorDFS'].isnull(), 'DFS']

    label_columns = ['RFSorDFS']
    print("running baseline\n")
    run_experiment(merged, feature_columns, label_columns)
    print("running PCA\n")
    run_pca(merged, label_columns, genes_columns, aggregated_treatment_columns)


def run_pca(merged, label_columns, genes_columns, aggregated_treatment_columns):
    for pca_dim in range(25, 300, 25):
        print('\n')
        print('PCA components {0}\n'.format(pca_dim))
        X = merged[genes_columns].to_numpy()
        pca = PCA(n_components=pca_dim)
        pca.fit(X)
        X1 = pca.transform(X)
        merged_pca = merged.copy()
        PCA_features = ['PCA{0}'.format(i) for i in range(X1.shape[1])]
        for i in range(X1.shape[1]):
            merged_pca.insert(0, PCA_features[i], X1[:, i])
        feature_columns = genes_columns + PCA_features + aggregated_treatment_columns
        run_experiment(merged_pca, feature_columns, label_columns)


def run_experiment(data_merged, feature_columns, label_columns):
    res = defaultdict(list)
    merged1 = data_merged[~data_merged[label_columns[0]].isnull()]
    print('testing on {}'.format(label_columns))
    for i in range(5):
        better_than_bias_studies = set()
        print(i)
        for study_name, study_id in study_mapping.items():
            if study_id not in merged1.study.unique():
                continue

            train_data, train_labels, val_data, val_labels = next(split_by_study(merged1,
                                                                      feature_columns,
                                                                      label_columns,
                                                                      study=study_id,
                                                                      to_numpy=True))
            model = xgb.XGBClassifier()
    #         model = xgb.XGBClassifier(booster='dart',
    #                                   n_estimators=300,
    #                                   max_depth=4,
    #                                   subsample=1,
    #                                   colsample_bytree=0.5,
    #                                   learning_rate=0.3)
            # train the model
            clf = model.fit(train_data, train_labels,
    #                         eval_set=[(train_data, train_labels)],
                            verbose=False)
            y_pred = clf.predict_proba(val_data)[:, 1]
            x_pred = clf.predict_proba(train_data)[:, 1]
            compute_metrics(res, val_labels.flatten(), y_pred, train_labels, x_pred)
            if res['accuracy'][-1] > res['bias'][-1]:
                print(study_name)
                better_than_bias_studies.add(study_name)
            for key in ['accuracy', 'auc', 'bias']:
                if res['accuracy'][-1] > res['bias'][-1]:
                    ave = res[key][-1]
                    print('{0}: {1}'.format(key, ave))

        print('studies better than bias {0}'.format(len(better_than_bias_studies)))
    for key in res:
        ave = numpy.asarray(res[key]).mean(axis=0)
        print('{0}: {1}'.format(key, ave))
    print('studies better than bias {0}'.format(better_than_bias_studies))


main()
