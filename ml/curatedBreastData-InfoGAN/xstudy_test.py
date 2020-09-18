import argparse
import numpy as np
import pandas as pd
from funs_common import read_treat_dataset
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score
from collections import defaultdict

import ipdb


def apply_test(dataset, idx2name, clf_callback, treatments=[]):
    outcomes = {outcome: None for outcome in ['pCR', 'RFS', 'DFS', 'posOutcome']}
    for label_name in outcomes:
        # test data
        study_name = idx2name[study_idx]
        test_study_df = dataset.loc[dataset.study == study_name]
        # filter out NaNs
        test_study_df = test_study_df[test_study_df[label_name].notnull()]  # + treatments
        test_study_f = test_study_df.iloc[:, 10:].values
        test_study_y = test_study_df[label_name].values
        if (np.unique(test_study_y).size == 1) or (test_study_y.size == 0):
            continue
        if treatments:
            test_study_t = test_study_df[treatments].values
            test_study_f = np.concatenate([test_study_f, test_study_t], 1)
            #import ipdb; ipdb.set_trace()
        # train data
        train_study_df = dataset.loc[dataset.study != study_name]
        # filter out NaNs
        train_study_df = train_study_df[train_study_df[label_name].notnull()]  # + treatments
        train_study_f = train_study_df.iloc[:, 10:].values
        train_study_y = train_study_df[label_name].values
        #import ipdb; ipdb.set_trace()
        if (np.unique(train_study_y).size == 1) or (train_study_y.size == 0):
            continue
        if treatments:
            train_study_t = train_study_df[treatments].values
            train_study_f = np.concatenate([train_study_f, train_study_t], 1)
            #import ipdb; ipdb.set_trace()
        # if treatments:  # if treatments != []
        #     train_trt_df = train_study_df[treatments]
        #     train_trt_notn = train_trt_df.notnull().values
        #     if not train_trt_notn.all():
        #         import ipdb; ipdb.set_trace()
        train_acc, test_acc, train_auc, recall_0, recall_1, test_auc = clf_callback( # , test_auc
            train_study_f, train_study_y, test_study_f, test_study_y
        )
        outcomes[label_name] = {'train_acc': train_acc, 'test_acc': test_acc,
                                     'train_auc': train_auc, 'recall_0': recall_0, 'recall_1': recall_1}
        if test_auc:
            outcomes[label_name]['test_auc'] = test_auc
    return outcomes


def apply_logistic_regression(train_x, train_y, test_x, test_y):
    logreg = LogisticRegression(solver='liblinear').fit(train_x, train_y)
    train_predictions = logreg.predict(train_x)
    test_predictions = logreg.predict(test_x)
    all_train_logits = logreg.predict_proba(train_x)
    all_test_logits = logreg.predict_proba(test_x)
    #train_logits = all_train_logits[range(train_y.size), train_y.astype(np.int)]
    #test_logits = all_test_logits[range(test_y.size), test_y.astype(np.int)]
    train_logits = all_train_logits[:, 1]
    test_logits = all_test_logits[:, 1]
    # compute accuracies
    train_accuracy = accuracy_score(train_y, train_predictions)
    test_accuracy = accuracy_score(test_y, test_predictions)
    # compute AUC
    train_auc = roc_auc_score(train_y, train_logits)
    # compute recall(s)
    y0_args = np.argwhere(test_y == 0).reshape(-1)
    y1_args = np.argwhere(test_y == 1).reshape(-1)
    test_predictions_0 = test_predictions[y0_args]
    test_predictions_1 = test_predictions[y1_args]
    #recall_score(np.zeros_like(test_predictions_0), test_predictions_0, zero_division=0)
    recall_0 = test_predictions_0.size / (test_predictions_0.size + test_predictions_0.sum())
    #recall_score(np.ones_like(test_predictions_1), test_predictions_1, zero_division=0)
    recall_1 = test_predictions_1.size / (test_predictions_1.size + (test_predictions_1.size - test_predictions_1.sum()))
    # if recall_0 == recall_1 == 0.0:
    #     ipdb.set_trace()
    #test_auc = roc_auc_score(test_predictions, test_logits)
    test_auc = None
    if np.unique(test_predictions).size == 2:
        test_auc = roc_auc_score(test_predictions, test_logits)
    return train_accuracy, test_accuracy, train_auc, recall_0, recall_1, test_auc


def apply_xgb(train_x, train_y, test_x, test_y):
    xgb = XGBClassifier()
    xgb.fit(train_x, train_y)
    train_predictions = xgb.predict(train_x)
    test_predictions = xgb.predict(test_x)
    all_train_logits = xgb.predict_proba(train_x)
    all_test_logits = xgb.predict_proba(test_x)
    train_logits = all_train_logits[range(train_y.size), train_y.astype(np.int)]
    test_logits = all_test_logits[range(test_y.size), test_y.astype(np.int)]
    train_accuracy = accuracy_score(train_y, train_predictions)
    test_accuracy = accuracy_score(test_y, test_predictions)
    train_auc = roc_auc_score(train_y, train_logits)
    # compute recall(s)
    y0_args = np.argwhere(test_y == 0)
    y1_args = np.argwhere(test_y == 1)
    test_predictions_0 = test_predictions[y0_args]
    test_predictions_1 = test_predictions[y1_args]
    # recall_score(np.zeros_like(test_predictions_0), test_predictions_0, zero_division=0)
    recall_0 = test_predictions_0.size / (test_predictions_0.size + test_predictions_0.sum())
    # recall_score(np.ones_like(test_predictions_1), test_predictions_1, zero_division=0)
    recall_1 = test_predictions_1.size / (
                test_predictions_1.size + (test_predictions_1.size - test_predictions_1.sum()))
    #test_auc = roc_auc_score(test_predictions, test_logits)
    test_auc = None
    if np.unique(test_predictions).size == 2:
        test_auc = roc_auc_score(test_predictions, test_logits)
    return train_accuracy, test_accuracy, train_auc, recall_0, recall_1, test_auc


def handle_outs(outs):
    missing_pcr, missing_rfs, missing_dfs = 0, 0, 0
    pcrs, rfss, dfss, posoutcomes = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    for study_idx in outs:
        out = outs[study_idx]
        if out['pCR'] is None:
            missing_pcr += 1
        else:
            for m in out['pCR']:
                pcrs[m].append(out['pCR'][m])
        if out['RFS'] is None:
            missing_rfs += 1
        else:
            for m in out['RFS']:
                rfss[m].append(out['RFS'][m])
        if out['DFS'] is None:
            missing_dfs += 1
        else:
            for m in out['DFS']:
                dfss[m].append(out['DFS'][m])
        for m in out['posOutcome']:
            posoutcomes[m].append(out['posOutcome'][m])
    # train_acc {0:3.3f} train_auc {1:3.3f}
    #ipdb.set_trace()
    # print('\txstudy pCR accuracy: {0:3.3f} min: {1:3.3f} max: {2:3.3f} std: {3:3.3f} recall0: {4:3.3f} recall1: {5:3.3f}'.format(
    #     np.mean(pcrs['test_acc']), np.min(pcrs['test_acc']), np.max(pcrs['test_acc']), np.std(pcrs['test_acc']),
    #     np.mean(pcrs['recall_0']), np.mean(pcrs['recall_1'])
    # ))
    # print('\txstudy RFS accuracy: {0:3.3f} min: {1:3.3f} max: {2:3.3f} std: {3:3.3f} recall0: {4:3.3f} recall1: {5:3.3f}'.format(
    #     np.mean(rfss['test_acc']), np.min(rfss['test_acc']), np.max(rfss['test_acc']), np.std(rfss['test_acc']),
    #     np.mean(rfss['recall_0']), np.mean(rfss['recall_1'])
    # ))
    # print('\txstudy DFS accuracy: {0:3.3f} min: {1:3.3f} max: {2:3.3f} std: {3:3.3f} recall0: {4:3.3f} recall1: {5:3.3f}'.format(
    #     np.mean(dfss['test_acc']), np.min(dfss['test_acc']), np.max(dfss['test_acc']), np.std(dfss['test_acc']),
    #     np.mean(dfss['recall_0']), np.mean(dfss['recall_1'])
    # ))
    # #import ipdb; ipdb.set_trace()
    # print('\txstudy posOutcome accuracy: {0:3.3f} min: {1:3.3f} max: {2:3.3f} std: {3:3.3f} recall0: {4:3.3f} recall1: {5:3.3f}'.format(
    #     np.mean(posoutcomes['test_acc']), np.min(posoutcomes['test_acc']), np.max(posoutcomes['test_acc']), np.std(posoutcomes['test_acc']),
    #     np.mean(posoutcomes['recall_0']), np.mean(posoutcomes['recall_1'])
    # ))
    if pcrs['test_auc']:
        print(
            '\txstudy pCR accuracy: {0:3.3f} AUC: {1:3.3f}'.format(
                np.mean(pcrs['test_acc']), np.mean(pcrs['test_auc'])
            ))
    else:
        print(
            '\txstudy pCR accuracy: {0:3.3f} AUC: -'.format(
                np.mean(pcrs['test_acc'])
            ))
    if rfss['test_auc']:
        print(
            '\txstudy RFS accuracy: {0:3.3f} AUC: {1:3.3f}'.format(
                np.mean(rfss['test_acc']), np.mean(rfss['test_auc'])
            ))
    else:
        print(
            '\txstudy RFS accuracy: {0:3.3f} AUC: -'.format(
                np.mean(rfss['test_acc'])
            ))
    if dfss['test_auc']:
        print(
            '\txstudy DFS accuracy: {0:3.3f} AUC: {1:3.3f}'.format(
                np.mean(dfss['test_acc']), np.mean(dfss['test_auc'])
            ))
    else:
        print(
            '\txstudy DFS accuracy: {0:3.3f} AUC: -'.format(
                np.mean(dfss['test_acc'])
            ))
    if posoutcomes['test_auc']:
        print(
            '\txstudy posOutcome accuracy: {0:3.3f} AUC: {1:3.3f}'.format(
                np.mean(posoutcomes['test_acc']), np.mean(posoutcomes['test_auc'])
            ))
    else:
        print(
            '\txstudy posOutcome accuracy: {0:3.3f} AUC: -'.format(
                np.mean(posoutcomes['test_acc'])
            ))


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('embds_file', help='file with embeddings')
parser.add_argument('bmc_root', help='root bmc15mldata1.csv')
args = parser.parse_args()

genes_dataset = pd.read_csv('data/ex15bmcMerged.csv.xz')
codes_dataset = pd.read_csv(args.embds_file)
treat_dataset = read_treat_dataset(args.bmc_root)
dataset = pd.merge(treat_dataset, codes_dataset)
dataset = pd.merge(dataset, genes_dataset)  #

studies = dataset.study.unique()
studies_idx2name = {i: n for i, n in enumerate(studies)}
#train_accs, test_accs = [], []
logreg_outs = {}
xgb_outs = {}
for study_idx in studies_idx2name:
    print('\rSTUDY {0:02d} in processing'.format(study_idx), end='', flush=True)
    logreg_out = apply_test(dataset, studies_idx2name, apply_logistic_regression)
    logreg_outs[study_idx] = logreg_out
    xgb_out = apply_test(dataset, studies_idx2name, apply_xgb)
    xgb_outs[study_idx] = xgb_out
print()

print('Logistic Regression')
handle_outs(logreg_outs)
print('XGBoost')
handle_outs(xgb_outs)

# run test with chemo
logreg_outs = {}
xgb_outs = {}
TREATMENTS = ['radio', 'chemo', 'hormone'] # , 'surgery'
current_treatments = []
for treatment in TREATMENTS:
    current_treatments.append(treatment)
    for study_idx in studies_idx2name:
        print('\rSTUDY {0:02d} in processing'.format(study_idx), end='', flush=True)
        logreg_out = apply_test(dataset, studies_idx2name, apply_logistic_regression, current_treatments)
        logreg_outs[study_idx] = logreg_out
        xgb_out = apply_test(dataset, studies_idx2name, apply_xgb, current_treatments)
        xgb_outs[study_idx] = xgb_out
    print()
    print(', '.join(current_treatments))
    print('Logistic Regression')
    handle_outs(logreg_outs)
    print('XGBoost')
    handle_outs(xgb_outs)
#for study_idx in logreg_outs:
#    pass
#import ipdb; ipdb.set_trace()



