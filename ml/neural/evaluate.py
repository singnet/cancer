"""
Evaluate discriminator features in cross-study validation
"""
from collections import defaultdict

import numpy
import torch

from dataset import get_merged_common_dataset
from infogan import Generator, Discriminator
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from metrics import compute_metrics, compute_auc



def get_data(t_set):
    data = t_set.features
    labels = t_set.labels.iloc[:, 1]
    data = data.to_numpy()
    labels = labels.to_numpy()
    nonzero = labels.nonzero()[0].flatten()
    labels = labels[nonzero]
    labels += labels < 0
    return data[nonzero], labels

xgb = False
def xgboost_test(extractor, opt):
    import xgboost as xgb
    res = defaultdict(list)
    res_train = defaultdict(list)
    for study_num in range(7):
        #print(study_name)
        train_set, test_set = get_merged_common_dataset(opt, skip_study=study_num)
        train_data, train_labels = get_data(train_set)
        val_data, val_labels = get_data(test_set)
        if True:
            train_features = extractor(train_data).detach().numpy()
            val_features = extractor(val_data).detach().numpy()
        else:
            train_features = train_data
            val_features = val_data
        # train the model
        if xgb:
            model = xgb.XGBClassifier()
            clf = model.fit(train_features, train_labels.astype(int),
                            eval_set=[(val_features, val_labels)],
                            early_stopping_rounds=50, verbose=False,
                            eval_metric='auc')
        else:
            #model = LogisticRegression(max_iter=1000)
            model = SVC(probability=True, class_weight='balanced')
            clf = model.fit(train_features, train_labels.astype(int))

        assert train_labels.astype(int).min() >= 0
        print(val_data.shape)
        res['bias'].append(val_labels.sum() / len(val_labels))
        print(res['bias'][-1])
        y_pred = clf.predict_proba(val_features)[:, 1]
        x_pred = clf.predict_proba(train_features)[:, 1]
        compute_metrics(res, val_labels.flatten() > 0.5, y_pred > 0.5)
        compute_auc(res, val_labels.flatten() > 0.5 , y_pred)
        compute_metrics(res_train, train_labels.flatten() > 0.5, x_pred > 0.5)
        compute_auc(res_train, train_labels.flatten() > 0.5 , x_pred)
    for key in res_train:
        ave = numpy.asarray(res_train[key]).mean(axis=0)
        print('Train {0}: {1}'.format(key, ave))
    for key in res:
        ave = numpy.asarray(res[key]).mean(axis=0)
        print('Test {0}: {1}'.format(key, ave))


def main():
    from train_infogan import parse_args
    opt = parse_args()
    import pdb;pdb.set_trace()
    train_set, test_set = get_merged_common_dataset(opt)
    size = train_set.features.shape[1]
    discriminator = Discriminator(opt, size, train_set.binary)
    discriminator.load_state_dict(torch.load(opt.disc_path))
    discriminator.eval()
    xgboost_test(discriminator.extract_features, opt)


if __name__ == '__main__':
    main()
