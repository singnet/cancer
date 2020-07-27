import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from data_loader import CuratedBreastCancerData

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('embds_file', help='file with embeddings')
args = parser.parse_args()

data = CuratedBreastCancerData(100, test_split=0.0)
codes_dataset = pd.read_csv(args.embds_file).values  # load embeddings
study_labels = data.data['study'].values  # study labels

accs = []
for _ in range(10):  # 10 folds for linear classifier
    pidxs = np.random.permutation(codes_dataset.shape[0])
    # shuffle to simplify train/test separation
    codes_dataset = codes_dataset[pidxs]
    study_labels = study_labels[pidxs]
    # train/test 80/20 (~2000/400) separation
    train_codes = codes_dataset[:-400]
    train_labels = study_labels[:-400]
    test_codes = codes_dataset[-400:]
    test_labels = study_labels[-400:]

    clf = LogisticRegression().fit(train_codes, train_labels)
    p_labels = clf.predict(test_codes)
    acc = accuracy_score(test_labels, p_labels)
    accs.append(acc)
    print('Study recognition acc: {0:3.3f}'.format(acc))

print('Average acc: {0:3.3f}'.format(np.mean(accs)))

