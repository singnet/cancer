import torch
from infogan import cuda, FloatTensor, LongTensor, to_categorical, Generator, Discriminator
from data_loader import CuratedBreastCancerData
import yaml
import argparse
from tqdm import tqdm
from ipywidgets import interact
import ipywidgets as widgets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from scipy import stats

# load config
with open('config1.yml', 'r') as f:
    y = yaml.load(f, Loader=yaml.SafeLoader)
opt = argparse.Namespace(**y)

data = CuratedBreastCancerData(opt.batch_size, test_split=0.0)

generator = Generator(opt)
generator.load_state_dict(torch.load('generator.pth'))
discriminator = Discriminator(opt)
discriminator.load_state_dict(torch.load('discriminator.pth'))

if cuda:
    generator = generator.cuda().eval()
    discriminator = discriminator.cuda().eval()
    print('CUDA is here')
else:
    print('CPU using')

codes, hidden, clusters, tags, valids = [], [], [], [], []
for gex in data.traverse_gex_study():
    with torch.no_grad():
        batch_gex = FloatTensor(gex)
        valid, cluster, pred_code = discriminator(batch_gex)
        hi_z = discriminator.hidden(batch_gex)
    valids.append(valid.detach().cpu().numpy())
    codes.append(pred_code.detach().cpu().numpy())
    clusters.append(cluster.detach().cpu().numpy().argmax(1))
    hidden.append(hi_z.detach().cpu().numpy())
codes = np.vstack(codes)
valids = np.vstack(valids)
hidden = np.vstack(hidden)
clusters = np.hstack(clusters)

patient_idxs = data.data.loc[np.invert(data.data['patient_ID_x'].isnull().values), ['patient_ID_x']].values.reshape(-1)
patient_idxs.shape

codes_df = pd.DataFrame(codes, index=patient_idxs)
codes_df.to_csv('data/codes_{0}.csv'.format(opt.code_dim), index_label='patient_ID')

h_df = pd.DataFrame(hidden, index=patient_idxs)
h_df.to_csv('data/hidden.csv', index_label='patient_ID')





