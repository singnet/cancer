import yaml
import argparse
import numpy as np
import pandas as pd


class MetaGxDataset:
    def __init__(self, gexs_csv, treatments_csv, studies_csv, batch_size, gexs_normalization=True):
        self.gexs = pd.read_csv(gexs_csv)
        if gexs_normalization:
            self.gexs_norm_params = {}
            for column_name in self.gexs.columns:
                if self.gexs[column_name].dtype is np.dtype(np.float):
                    column_mean = self.gexs[column_name].mean()
                    column_std = self.gexs[column_name].std()
                    self.gexs[column_name] = (self.gexs[column_name] - column_mean) / column_std
                    self.gexs_norm_params[column_name] = {'mean': column_mean, 'std': column_std}
        self.treatments = pd.read_csv(treatments_csv)
        self.studies = pd.read_csv(studies_csv)
        self.metagx = pd.merge(self.gexs, self.treatments)
        self.metagx = pd.merge(self.metagx, self.studies, left_on="sample_name", right_on="sample_name")
        self.n_genes = len(self.gexs.columns) - 1
        self.n_studies = len(self.metagx.study.unique())
        self.batch_size = batch_size
        self.batches_per_epoch = self.gexs.shape[0] // self.batch_size
        # todo: specify getters for studies and posOutcomes
        self.study_names = self.metagx[self.metagx.posOutcome.notnull()]['study'].unique()

    def add_features(self, features):
        self.features = features
        self.metagx = pd.merge(self.metagx, self.features)

    def extract_data(self, columns_as_x, column_as_y, do_dropna=True):
        if not isinstance(columns_as_x, list):
            columns_as_x = columns_as_x.tolist()
        x = self.metagx[columns_as_x + [column_as_y]]
        if do_dropna:
            x = x.dropna()
        return x[columns_as_x].values, x[column_as_y].values

    def extract_study_data(self, columns_as_x, column_as_y, study): # , do_dropna=True
        if not isinstance(columns_as_x, list):
            columns_as_x = columns_as_x.tolist()
        study_x = self.metagx[self.metagx.posOutcome.notnull()]
        study_x = study_x.loc[self.metagx.study == study]
        study_x = study_x[columns_as_x + [column_as_y]]
        nstudy_x = self.metagx[self.metagx.posOutcome.notnull()]
        nstudy_x = nstudy_x.loc[self.metagx.study != study]
        nstudy_x = nstudy_x[columns_as_x + [column_as_y]]
        #import ipdb; ipdb.set_trace()
        # if do_dropna:
        #     study_x = study_x.dropna()
        #     nstudy_x = nstudy_x.dropna()
        return study_x[columns_as_x].values, study_x[column_as_y].values, \
               nstudy_x[columns_as_x].values, nstudy_x[column_as_y].values

    def get_gexs_batch(self, batch_size):
        for _ in range(self.batches_per_epoch):
            yield self.gexs.sample(batch_size)[self.gexs.columns[1:]].values

    def traverse_gexs(self):
        for i in range(0, self.gexs.shape[0], self.batch_size):
            batch_gex = self.gexs.iloc[i:i+self.batch_size]
            yield batch_gex.iloc[:, 1:].values


def metaGxConfigLoader(config_path):
    with open(config_path, 'r') as f:
        y = yaml.load(f, Loader=yaml.SafeLoader)
    opt = argparse.Namespace(**y)
    return opt


