import os
import torch
from infogan import cuda, FloatTensor, Discriminator
from metagxdataset import MetaGxDataset, metaGxConfigLoader
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument("train_config", help="train config file name")
parser.add_argument("test_config", help="test config file name")
#parser.add_argument("code_type", help="type of discriminators outcome")
args = parser.parse_args()

# load config
train_config = metaGxConfigLoader(args.train_config)
test_config = metaGxConfigLoader(args.test_config)

#data = CuratedBreastCancerData(opt.batch_size, test_split=0.0)
train_dataset = MetaGxDataset(
    train_config.gexs_csv, train_config.treatments_csv,
    train_config.studies_csv, train_config.batch_size, train_config.normalization
)

if args.train_config == args.test_config:
    test_dataset = train_dataset
else:
    test_dataset = MetaGxDataset(
        test_config.gexs_csv, test_config.treatments_csv,
        test_config.studies_csv, test_config.batch_size,
        gexs_normalization=False # use normalization parameters from train_dataset
    )
    # todo: normalize test_dataset with train_dataset.gexs_norm_params
    for gname in train_dataset.gexs_norm_params:
        gmean, gstd = train_dataset.gexs_norm_params[gname]['mean'], train_dataset.gexs_norm_params[gname]['std']
        test_dataset.gexs[gname] = (test_dataset.gexs[gname] - gmean) / gstd


# load everything
categorical_size = train_config.n_classes if train_config.n_classes else train_dataset.n_studies
load_checkpoints = 's{0:02d}_c{1:02d}_{2}'.format(
    train_config.code_dim,
    categorical_size,
    os.path.split(train_config.gexs_csv)[-1].split('.')[0]
)
load_path = os.path.join('checkpoints', load_checkpoints)
# No need to load generator
# generator = Generator(
#     train_config.latent_dim,
#     categorical_size,
#     train_config.code_dim,
#     train_dataset.n_genes
# )
# generator_checkpoint = os.path.join(load_path, 'generator.pth')
# generator.load_state_dict(torch.load(generator_checkpoint))

discriminator = Discriminator(
    categorical_size,
    train_config.code_dim,
    train_dataset.n_genes
)
discriminator_checkpoint = os.path.join(load_path, 'discriminator.pth')
discriminator.load_state_dict(torch.load(discriminator_checkpoint))

if cuda:
    #generator = generator.cuda().eval()
    discriminator = discriminator.cuda().eval()
    print('CUDA is here')
else:
    print('CPU using')

codes, hidden, clusters, tags, valids = [], [], [], [], []
for gex in tqdm(test_dataset.traverse_gexs()):
    with torch.no_grad():
        batch_gex = FloatTensor(gex)
        valid, cluster, pred_code = discriminator(batch_gex)
        #hi_z = discriminator.hidden(batch_gex)
    #valids.append(valid.detach().cpu().numpy())
    codes.append(pred_code.detach().cpu().numpy())
    #clusters.append(cluster.detach().cpu().numpy().argmax(1))
    #hidden.append(hi_z.detach().cpu().numpy())
codes = np.vstack(codes)
#valids = np.vstack(valids)
#hidden = np.vstack(hidden)
#clusters = np.hstack(clusters)

patient_idxs = test_dataset.gexs.loc[np.invert(test_dataset.gexs['sample_name'].isnull().values), ['sample_name']].values.reshape(-1)
#patient_idxs.shape

save_path = os.path.join('features', load_checkpoints)
os.makedirs(save_path, exist_ok=True)

codes_df = pd.DataFrame(codes, index=patient_idxs)
codes_df.to_csv(os.path.join(save_path, 'codes.csv'), index_label='sample_name')

#h_df = pd.DataFrame(hidden, index=patient_idxs)
#h_df.to_csv('data/hidden.csv', index_label='patient_ID')
