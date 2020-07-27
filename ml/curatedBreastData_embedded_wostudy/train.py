import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import yaml
from data_loader import CuratedBreastCancerData
from models import cuda, FloatTensor
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.e = nn.Sequential(
            nn.Linear(8832, 24),
            nn.SELU(),
            nn.BatchNorm1d(24)
        )
        self.c = nn.Sequential(
            nn.Linear(24, 1),
            nn.Sigmoid()
        )

    def forward(self, gex):
        z = self.e(gex)
        p = self.c(z)
        return z, p


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.d = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(24*2, 128)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Linear(128, 128)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, z1, z2):
        return self.d(torch.cat([z1, z2], 1))


parser = argparse.ArgumentParser()
parser.add_argument("config", help="config file name")
args = parser.parse_args()

# load config
with open(args.config, 'r') as f:
    y = yaml.load(f, Loader=yaml.SafeLoader)
opt = argparse.Namespace(**y)

# Loss functions
classification_loss = nn.BCELoss()
discriminator_loss = nn.BCELoss()

# Initialize generator and discriminator
classifier = Classifier()
discriminator = Discriminator()

if cuda:
    classifier.cuda()
    discriminator.cuda()
    classification_loss.cuda()
    discriminator_loss.cuda()

# Configure data loader
data = CuratedBreastCancerData(
    opt.batch_size, test_split=opt.test_split,
    study17_file=os.path.join(opt.dataset_root, 'ex15bmcMerged.csv.xz'),
    treatments_file=os.path.join(opt.dataset_root, 'bmc15mldata1.csv')
)

# Optimizers
c_optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)  # 1e-2
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-4)

summary = SummaryWriter('summary/')


def test(studies, label):
    global classifier
    classifier = classifier.eval()
    accs, recall, loss = [], [], []
    z_codes = []
    study_labels = []
    with torch.no_grad():
        for study_idx in studies:
            study = studies[study_idx]
            z, p_posOutcome = classifier(FloatTensor(study.gexs))
            study_loss = classification_loss(p_posOutcome, FloatTensor(study.treatments[:, -1].reshape(-1, 1))).item()
            p_posOutcome = p_posOutcome.cpu().numpy().astype(np.int).reshape(-1)
            study_acc = accuracy_score(study.treatments[:, -1], p_posOutcome)
            # logistic regression on studies
            study_label = study.treatments[:, 0]
            z_code = z.cpu().numpy()
            #study_recall = recall_score(study.treatments[:, -1], p_posOutcome)
            accs.append(study_acc)
            #recall.append(study_recall)
            loss.append(study_loss)
            z_codes.append(z_code)
            study_labels.append(study_label)
        # log reg
        z_codes = np.vstack(z_codes)
        study_labels = np.hstack(study_labels)
        lrg = LogisticRegression(solver='liblinear').fit(z_codes, study_labels)
        lrg_p_studies = lrg.predict(z_codes)
        lrg_study_acc = accuracy_score(study_labels, lrg_p_studies)
        # ++++++++
        summary.add_scalar('{0}-overal/accuracy'.format(label), np.mean(accs), epoch)
        #summary.add_scalar('{0}-overal/recall'.format(label), np.mean(recall), epoch)
        summary.add_scalar('{0}-overal/loss'.format(label), np.mean(loss), epoch)
        summary.add_scalar('{0}-overal/study_acc'.format(label), lrg_study_acc, epoch)


batch_global = 0
for epoch in range(1, opt.n_epochs + 1):
    for batch_i in range(1, opt.batch_per_epoch + 1):
        # ------------------
        # Train classifier to predict posOutcome
        # ------------------
        classifier = classifier.train()
        c_optimizer.zero_grad()

        # train classifier
        batch_gex, batch_study, batch_posOutcome = data.get_batch_bspo(opt.batch_size // 17, data.train_studies)
        batch_gex_tensor = FloatTensor(batch_gex)
        batch_study_tensor = FloatTensor(batch_study)
        batch_posOutcome_tensor = FloatTensor(batch_posOutcome.reshape(-1, 1))

        _, p_posOutcome = classifier(batch_gex_tensor)
        c_loss = classification_loss(p_posOutcome, batch_posOutcome_tensor)

        np_p_posOutcome = p_posOutcome.detach().cpu().numpy().reshape(-1)
        batch_acc = accuracy_score(batch_posOutcome, np_p_posOutcome.round())

        c_loss.backward()
        c_optimizer.step()

        # train discriminator
        #study_batch = data.get_supervised_study_posOutcome_batch(opt.batch_size // 17)
        #batch_gex
        #z, _ = classifier(batch_gex_tensor)
        # todo: update discriminator's weights


        print('\rEpoch {0:04d} Batch {1:04d}'.format(epoch, batch_i), end='')
        #ipdb.set_trace()
        summary.add_scalar('train-batch/loss', c_loss.item(), batch_global)
        summary.add_scalar('train-batch/acc', batch_acc, batch_global)
        #summary.add_scalar('test-batch/loss', c_test_loss.item(), batch_global)
        #summary.add_scalar('test-batch/acc', batch_test_acc, batch_global)

        batch_global += 1
    #test(data.train_studies, 'train')
    test(data.test_studies, 'test')
print()
