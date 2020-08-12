import argparse
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, recall_score
from metagxdataset import MetaGxDataset, metaGxConfigLoader
from tqdm import tqdm


def test_classifier(x_train, y_train, x_test, y_test, classifier):
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    acc = np.mean(y_test == y_pred)
    y_pred_prob = classifier.predict_proba(x_test)[:, 1]
    assert classifier.classes_[1] == 1
    recall_0 = recall_score(y_test, y_pred, pos_label=0)
    recall_1 = recall_score(y_test, y_pred, pos_label=1)
    auc = roc_auc_score(y_test, y_pred_prob)
    return acc, recall_0, recall_1, auc


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('features_file', help='file with features')
parser.add_argument('test_config', help='test configuration file')
#parser.add_argument('--n_splits', default=10, type=int, help='number of splits')
args = parser.parse_args()

# load configuration and data
test_config = metaGxConfigLoader(args.test_config)
test_dataset = MetaGxDataset(
    test_config.gexs_csv, test_config.treatments_csv,
    test_config.studies_csv, test_config.batch_size, test_config.normalization
)

# prepare data
codes_dataset = pd.read_csv(args.features_file)  # load features
test_dataset.add_features(codes_dataset)  # merge features with MetaGx

results_logit = []
results_xgb = []
for study_name in tqdm(test_dataset.study_names):
    # extract features and posOutcome as labels
    study_x, study_y, nstudy_x, nstudy_y = test_dataset.extract_study_data(
        test_dataset.features.columns[1:],
        'posOutcome',
        study=study_name
    )
    logit_acc, logit_recall_0, logit_recall_1, logit_auc = test_classifier(
        nstudy_x, nstudy_y, study_x, study_y, LogisticRegression()
    )
    xgb_acc, xgb_recall_0, xgb_recall_1, xgb_auc = test_classifier(
        nstudy_x, nstudy_y, study_x, study_y, XGBClassifier()
    )
    results_logit.append([logit_acc, logit_recall_0, logit_recall_1, logit_auc])
    results_xgb.append([xgb_acc, xgb_recall_0, xgb_recall_1, xgb_auc])

mean_ests = np.concatenate([np.mean(results_logit, 0).reshape(1, -1), np.mean(results_xgb, 0).reshape(1, -1)], 0)

summary_df = pd.DataFrame(data=mean_ests, index=['logit', 'xgb'], columns=['accuracy', 'recall_0', 'recall_1', 'AUC'])
print(summary_df)



# kf = StratifiedKFold(n_splits=args.n_splits, shuffle=True)
# results = defaultdict(list)
# # experiments with features and labels
# print('Features testing', flush=True)
# for i, (train_index, test_index) in tqdm(enumerate(kf.split(x_features, y)), total=args.n_splits):
#     k_train_x_features, k_train_y = x_features[train_index], y[train_index]
#     k_test_x_features, k_test_y = x_features[test_index], y[test_index]
#
#     k_logit_f_acc, k_logit_f_recall_0, k_logit_f_recall_1, k_logit_f_auc = test_classifier(
#         k_train_x_features, k_train_y,
#         k_test_x_features, k_test_y,
#         LogisticRegression()
#     )
#     k_xgb_f_acc, k_xgb_f_recall_0, k_xgb_f_recall_1, k_xgb_f_auc = test_classifier(
#         k_train_x_features, k_train_y,
#         k_test_x_features, k_test_y,
#         XGBClassifier()
#     )
#     results['logit_f'].append((k_logit_f_acc, k_logit_f_recall_0, k_logit_f_recall_1, k_logit_f_auc))
#     results['xgb_f'].append((k_xgb_f_acc, k_xgb_f_recall_0, k_xgb_f_recall_1, k_xgb_f_auc))
#
# # experiments with features and labels
# print('Features and treatments testing', flush=True)
# for i, (train_index, test_index) in tqdm(enumerate(kf.split(x_treatments_features, y_ft)), total=args.n_splits):
#     k_train_x_treatments_features, y_ft_train = x_treatments_features[train_index], y_ft[train_index]
#     k_test_x_treatments_features, y_ft_test = x_treatments_features[test_index], y_ft[test_index]
#
#     k_logit_ft_acc, k_logit_ft_recall_0, k_logit_ft_recall_1, k_logit_ft_auc = test_classifier(
#         k_train_x_treatments_features, y_ft_train,
#         k_test_x_treatments_features, y_ft_test,
#         LogisticRegression()
#     )
#     k_xgb_ft_acc, k_xgb_ft_recall_0, k_xgb_ft_recall_1, k_xgb_ft_auc = test_classifier(
#         k_train_x_treatments_features, y_ft_train,
#         k_test_x_treatments_features, y_ft_test,
#         XGBClassifier()
#     )
#     results['logit_ft'].append((k_logit_ft_acc, k_logit_ft_recall_0, k_logit_ft_recall_1, k_logit_ft_auc))
#     results['xgb_ft'].append((k_xgb_ft_acc, k_xgb_ft_recall_0, k_xgb_ft_recall_1, k_xgb_ft_auc))
#
# summary = []
# summary_index = ['logit_f', 'xgb_f', 'logit_ft', 'xgb_ft']
# for ke in summary_index:
#     summary.append(np.mean(results[ke], 0))
#
# summary_df = pd.DataFrame(data=summary, index=summary_index, columns=['accuracy', 'recall_0', 'recall_1', 'AUC'])
# print(summary_df)
#summary_df.to_excel()


# import argparse
# import numpy as np
# import pandas as pd
# from funs_common import read_treat_dataset
# from sklearn.linear_model import LogisticRegression
# from xgboost import XGBClassifier
# from sklearn.metrics import accuracy_score, roc_auc_score, recall_score
# from collections import defaultdict
#
# import ipdb
#
#
# def apply_test(dataset, idx2name, clf_callback):
#     outcomes = {outcome: None for outcome in ['pCR', 'RFS', 'DFS', 'posOutcome']}
#     for label_name in outcomes:
#         # test data
#         study_name = idx2name[study_idx]
#         test_study_df = dataset.loc[dataset.study == study_name]
#         # filter out NaNs
#         test_study_df = test_study_df[test_study_df[label_name].notnull()]
#         test_study_f = test_study_df.iloc[:, 10:].values
#         test_study_y = test_study_df[label_name].values
#
#         if (np.unique(test_study_y).size == 1) or (test_study_y.size == 0):
#             continue
#         # train data
#         train_study_df = dataset.loc[dataset.study != study_name]
#         # filter out NaNs
#         train_study_df = train_study_df[train_study_df[label_name].notnull()]
#         train_study_f = train_study_df.iloc[:, 10:].values
#         train_study_y = train_study_df[label_name].values
#         if (np.unique(train_study_y).size == 1) or (train_study_y.size == 0):
#             continue
#         train_acc, test_acc, train_auc, recall_0, recall_1 = clf_callback( # , test_auc
#             train_study_f, train_study_y, test_study_f, test_study_y
#         )
#         outcomes[label_name] = {'train_acc': train_acc, 'test_acc': test_acc,
#                                      'train_auc': train_auc, 'recall_0': recall_0, 'recall_1': recall_1}
#     return outcomes
#
#
# def apply_logistic_regression(train_x, train_y, test_x, test_y):
#     logreg = LogisticRegression(solver='liblinear').fit(train_x, train_y)
#     train_predictions = logreg.predict(train_x)
#     test_predictions = logreg.predict(test_x)
#     all_train_logits = logreg.predict_proba(train_x)
#     train_logits = all_train_logits[range(train_y.size), train_y.astype(np.int)]
#     # compute accuracies
#     train_accuracy = accuracy_score(train_y, train_predictions)
#     test_accuracy = accuracy_score(test_y, test_predictions)
#     # compute AUC
#     train_auc = roc_auc_score(train_y, train_logits)
#     # compute recall(s)
#     y0_args = np.argwhere(test_y == 0).reshape(-1)
#     y1_args = np.argwhere(test_y == 1).reshape(-1)
#     test_predictions_0 = test_predictions[y0_args]
#     test_predictions_1 = test_predictions[y1_args]
#     #recall_score(np.zeros_like(test_predictions_0), test_predictions_0, zero_division=0)
#     recall_0 = test_predictions_0.size / (test_predictions_0.size + test_predictions_0.sum())
#     #recall_score(np.ones_like(test_predictions_1), test_predictions_1, zero_division=0)
#     recall_1 = test_predictions_1.size / (test_predictions_1.size + (test_predictions_1.size - test_predictions_1.sum()))
#     # if recall_0 == recall_1 == 0.0:
#     #     ipdb.set_trace()
#     #test_auc = roc_auc_score(test_predictions, test_logits)
#     return train_accuracy, test_accuracy, train_auc, recall_0, recall_1
#
#
# def apply_xgb(train_x, train_y, test_x, test_y):
#     xgb = XGBClassifier()
#     xgb.fit(train_x, train_y)
#     train_predictions = xgb.predict(train_x)
#     test_predictions = xgb.predict(test_x)
#     all_train_logits = xgb.predict_proba(train_x)
#     train_logits = all_train_logits[range(train_y.size), train_y.astype(np.int)]
#     train_accuracy = accuracy_score(train_y, train_predictions)
#     test_accuracy = accuracy_score(test_y, test_predictions)
#     train_auc = roc_auc_score(train_y, train_logits)
#     # compute recall(s)
#     y0_args = np.argwhere(test_y == 0)
#     y1_args = np.argwhere(test_y == 1)
#     test_predictions_0 = test_predictions[y0_args]
#     test_predictions_1 = test_predictions[y1_args]
#     # recall_score(np.zeros_like(test_predictions_0), test_predictions_0, zero_division=0)
#     recall_0 = test_predictions_0.size / (test_predictions_0.size + test_predictions_0.sum())
#     # recall_score(np.ones_like(test_predictions_1), test_predictions_1, zero_division=0)
#     recall_1 = test_predictions_1.size / (
#                 test_predictions_1.size + (test_predictions_1.size - test_predictions_1.sum()))
#     #test_auc = roc_auc_score(test_predictions, test_logits)
#     return train_accuracy, test_accuracy, train_auc, recall_0, recall_1
#
#
# def handle_outs(outs):
#     missing_pcr, missing_rfs, missing_dfs = 0, 0, 0
#     pcrs, rfss, dfss, posoutcomes = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
#     for study_idx in outs:
#         out = outs[study_idx]
#         if out['pCR'] is None:
#             missing_pcr += 1
#         else:
#             for m in out['pCR']:
#                 pcrs[m].append(out['pCR'][m])
#         if out['RFS'] is None:
#             missing_rfs += 1
#         else:
#             for m in out['RFS']:
#                 rfss[m].append(out['RFS'][m])
#         if out['DFS'] is None:
#             missing_dfs += 1
#         else:
#             for m in out['DFS']:
#                 dfss[m].append(out['DFS'][m])
#         for m in out['posOutcome']:
#             posoutcomes[m].append(out['posOutcome'][m])
#     # train_acc {0:3.3f} train_auc {1:3.3f}
#     #ipdb.set_trace()
#     print('\txstudy pCR accuracy: {0:3.3f} min: {1:3.3f} max: {2:3.3f} std: {3:3.3f} recall0: {4:3.3f} recall1: {5:3.3f}'.format(
#         np.mean(pcrs['test_acc']), np.min(pcrs['test_acc']), np.max(pcrs['test_acc']), np.std(pcrs['test_acc']),
#         np.mean(pcrs['recall_0']), np.mean(pcrs['recall_1'])
#     ))
#     print('\txstudy RFS accuracy: {0:3.3f} min: {1:3.3f} max: {2:3.3f} std: {3:3.3f} recall0: {4:3.3f} recall1: {5:3.3f}'.format(
#         np.mean(rfss['test_acc']), np.min(rfss['test_acc']), np.max(rfss['test_acc']), np.std(rfss['test_acc']),
#         np.mean(rfss['recall_0']), np.mean(rfss['recall_1'])
#     ))
#     print('\txstudy DFS accuracy: {0:3.3f} min: {1:3.3f} max: {2:3.3f} std: {3:3.3f} recall0: {4:3.3f} recall1: {5:3.3f}'.format(
#         np.mean(dfss['test_acc']), np.min(dfss['test_acc']), np.max(dfss['test_acc']), np.std(dfss['test_acc']),
#         np.mean(dfss['recall_0']), np.mean(dfss['recall_1'])
#     ))
#     #import ipdb; ipdb.set_trace()
#     print('\txstudy posOutcome accuracy: {0:3.3f} min: {1:3.3f} max: {2:3.3f} std: {3:3.3f} recall0: {4:3.3f} recall1: {5:3.3f}'.format(
#         np.mean(posoutcomes['test_acc']), np.min(posoutcomes['test_acc']), np.max(posoutcomes['test_acc']), np.std(posoutcomes['test_acc']),
#         np.mean(posoutcomes['recall_0']), np.mean(posoutcomes['recall_1'])
#     ))
#
#
# parser = argparse.ArgumentParser(description='Process some integers.')
# parser.add_argument('embds_file', help='file with embeddings')
# parser.add_argument('bmc_root', help='root bmc15mldata1.csv')
# args = parser.parse_args()
#
# codes_dataset = pd.read_csv(args.embds_file)
# treat_dataset = read_treat_dataset(args.bmc_root)
# dataset = pd.merge(treat_dataset, codes_dataset)
#
# studies = dataset.study.unique()
# studies_idx2name = {i: n for i, n in enumerate(studies)}
# #train_accs, test_accs = [], []
# logreg_outs = {}
# xgb_outs = {}
# for study_idx in studies_idx2name:
#     print('\rSTUDY {0:02d} in processing'.format(study_idx), end='', flush=True)
#     logreg_out = apply_test(dataset, studies_idx2name, apply_logistic_regression)
#     logreg_outs[study_idx] = logreg_out
#     xgb_out = apply_test(dataset, studies_idx2name, apply_xgb)
#     xgb_outs[study_idx] = xgb_out
# print()
#
# print('Logistic Regression')
# handle_outs(logreg_outs)
# print('XGBoost')
# handle_outs(xgb_outs)
#for study_idx in logreg_outs:
#    pass
#import ipdb; ipdb.set_trace()
