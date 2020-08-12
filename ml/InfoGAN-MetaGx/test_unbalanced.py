import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
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
parser.add_argument('--n_splits', default=10, type=int, help='number of splits')
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
# extract features and posOutcome as labels
x_features, y = test_dataset.extract_data(test_dataset.features.columns[1:], 'posOutcome')
x_nan_features, y_nan = test_dataset.extract_data(test_dataset.features.columns[1:], 'posOutcome', do_dropna=False)
# extract features, treatments and posOutcome as labels
x_treatments_features, y_ft = test_dataset.extract_data(
    test_dataset.features.columns[1:].tolist() + ['hormone', 'chemo_x'],  # may be add 'recurrence'
    'posOutcome'
)
x_nan_treatments_features, y_nan_ft = test_dataset.extract_data(
    test_dataset.features.columns[1:].tolist() + ['hormone', 'chemo_x'],  # may be add 'recurrence'
    'posOutcome', do_dropna=False
)

kf = StratifiedKFold(n_splits=args.n_splits, shuffle=True)
results = defaultdict(list)
# experiments with features and labels
print('Features testing', flush=True)
for i, (train_index, test_index) in tqdm(enumerate(kf.split(x_features, y)), total=args.n_splits):
    k_train_x_features, k_train_y = x_features[train_index], y[train_index]
    k_test_x_features, k_test_y = x_features[test_index], y[test_index]
    k_logit_f_acc, k_logit_f_recall_0, k_logit_f_recall_1, k_logit_f_auc = test_classifier(
        k_train_x_features, k_train_y,
        k_test_x_features, k_test_y,
        LogisticRegression()
    )
    results['logit_f'].append((k_logit_f_acc, k_logit_f_recall_0, k_logit_f_recall_1, k_logit_f_auc))
    k_xgb_f_acc, k_xgb_f_recall_0, k_xgb_f_recall_1, k_xgb_f_auc = test_classifier(
        k_train_x_features, k_train_y,
        k_test_x_features, k_test_y,
        XGBClassifier()
    )
    results['xgb_f'].append((k_xgb_f_acc, k_xgb_f_recall_0, k_xgb_f_recall_1, k_xgb_f_auc))

# experiments with features and labels
print('Features and treatments testing', flush=True)
for i, (train_index, test_index) in tqdm(enumerate(kf.split(x_treatments_features, y_ft)), total=args.n_splits):
    k_train_x_treatments_features, y_ft_train = x_treatments_features[train_index], y_ft[train_index]
    k_test_x_treatments_features, y_ft_test = x_treatments_features[test_index], y_ft[test_index]

    k_logit_ft_acc, k_logit_ft_recall_0, k_logit_ft_recall_1, k_logit_ft_auc = test_classifier(
        k_train_x_treatments_features, y_ft_train,
        k_test_x_treatments_features, y_ft_test,
        LogisticRegression()
    )
    results['logit_ft'].append((k_logit_ft_acc, k_logit_ft_recall_0, k_logit_ft_recall_1, k_logit_ft_auc))
    k_xgb_ft_acc, k_xgb_ft_recall_0, k_xgb_ft_recall_1, k_xgb_ft_auc = test_classifier(
        k_train_x_treatments_features, y_ft_train,
        k_test_x_treatments_features, y_ft_test,
        XGBClassifier()
    )
    results['xgb_ft'].append((k_xgb_ft_acc, k_xgb_ft_recall_0, k_xgb_ft_recall_1, k_xgb_ft_auc))
    #import ipdb; ipdb.set_trace()

summary = []
summary_index = ['logit_f', 'xgb_f', 'logit_ft', 'xgb_ft']
for ke in summary_index:
    summary.append(np.mean(results[ke], 0))

summary_df = pd.DataFrame(data=summary, index=summary_index, columns=['accuracy', 'recall_0', 'recall_1', 'AUC'])
print(summary_df)
#import ipdb; ipdb.set_trace()
#summary_df.to_excel()

