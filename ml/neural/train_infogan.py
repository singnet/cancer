import argparse
import time
import os
import itertools
import numpy as np
from torch.autograd import Variable
import torch
import yaml
from data_loader import CuratedBreastCancerData
from infogan import Generator, Discriminator
from train_genes import get_datasets
from dataset import GeneDataset
from transform import DataLabelCompose, ToTensor, AdditiveUniform,\
        AdditiveUniformTriary, Compose, ToType
from data_util import util, metagx_util
from infogan_loss import infogan_loss, predictor_loss
from training import train, test


def main():
    opt = parse_args()
    cuda = True

    skip_study = opt.skip_study
    if skip_study == -1:
        skip_study = None
    train_set, test_set = get_merged_common_dataset(opt, skip_study=skip_study)
    size = train_set.features.shape[1]
    train_loader = torch.utils.data.DataLoader(train_set,
            batch_size=opt.batch_size, shuffle=True, num_workers=10)

    # Initialize generator and discriminator
    generator = Generator(opt, size, train_set.binary)
    if hasattr(opt, 'gen_path') and opt.gen_path:
       generator.load_state_dict(torch.load(opt.gen_path))
    # descriminator takes as input generator's output and treatment
    discriminator = Discriminator(opt, size, train_set.binary)
    if hasattr(opt, 'disc_path') and opt.disc_path:
       discriminator.load_state_dict(torch.load(opt.disc_path))
    model = InfoGAN(generator, discriminator)
    device = 'cpu'

    if cuda:
        generator.cuda()
        discriminator.cuda()
        device = 'cuda'


    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_info = torch.optim.Adam(
        itertools.chain(generator.parameters(), discriminator.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2)
    )

    # ----------
    #  Training
    # ----------
    if opt.test_ratio:
        test_loader = torch.utils.data.DataLoader(test_set,
            batch_size=opt.batch_size, shuffle=True, num_workers=7)

    loss = infogan_loss
    if hasattr(opt, 'train_descriminator') and opt.train_descriminator:
        loss = predictor_loss
        optimizer_info = torch.optim.Adam(
            discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    optimizers = dict(G=optimizer_G, D=optimizer_D,
                      info=optimizer_info)


    for epoch in range(opt.n_epochs):
    #   start = time.monotonic()
        if (epoch + 1) % opt.test_interval == 0:
            loss_test = lambda *args, **kwargs: loss(*args, **kwargs, optimizers=None, opt=opt)
            test(model, loss_test, device, test_loader)
        loss_f = lambda *args, **kwargs: loss(*args, **kwargs, opt=opt)
        train(model, loss_f, device, train_loader, optimizers, epoch + 1)
        if (epoch + 1) % opt.save_interval == 0:
            torch.save(model.discriminator.state_dict(), 'discriminator{0}.pth'.format(epoch))
            torch.save(model.generator.state_dict(), 'generator{0}.pth'.format(epoch))
    #    end = time.monotonic()
    #    print('running time {0}'.format(end - start))
    torch.save(model.discriminator.state_dict(), 'discriminator.pth')
    torch.save(model.generator.state_dict(), 'generator.pth')



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="config file name", required=True)
    args = parser.parse_args()
    print(args)
    # load config
    with open(args.config, 'r') as f:
        y = yaml.load(f, Loader=yaml.SafeLoader)
    opt = argparse.Namespace(**y)
    return opt


def get_merged_common_dataset(opt, skip_study=None):
    cancer_data_dir = opt.curated_breast_data_dir
    dataset_dict = util.load_merged_dataset(cancer_data_dir)
    mergedCurated = dataset_dict['merged']

    data = metagx_util.load_metagx_dataset(opt.metagx_data_dir, min_genes=opt.min_genes)
    merged = data['merged']
    genes_list = data['genes_features']

    metagx_pos_outcome = merged[merged.posOutcome.isin([-1, 1])]
    if skip_study is not None:
        study_to_skip = metagx_pos_outcome.study.unique()[skip_study]
    else:
        study_to_skip = None

    merged_common = util.merge_metagx_curated(merged, mergedCurated)

    merged_treatments = list(metagx_util.treatment_columns_metagx) + util.treatment_columns_bmc
    merged_treatments = [x for x in merged_treatments if x in merged_common]
    merged_treatments = list(set(merged_treatments))
    # add continious covariates to genes
    cont_columns = [x for x in merged_treatments if len(merged_common[x].unique()) > 20]
    merged_treatments = [x for x in merged_treatments if x not in cont_columns]
    common_genes_list = [x for x in genes_list if x in merged_common]
    if study_to_skip is None:
        train_data, train_labels, val_data, val_labels = util.random_split(merged_common,
                                                              common_genes_list + cont_columns + merged_treatments + ['posOutcome'],
                                                              ['study', 'posOutcome'],
                                                              balance_validation=False,
                                                              balance_by_study=False,
                                                              ratio=opt.test_ratio,
                                                              to_numpy=False)
    else:
        train_data, train_labels, val_data, val_labels = next(util.split_by_study(merged_common,
                                                              common_genes_list + cont_columns + merged_treatments + ['posOutcome'],
                                                              ['study', 'posOutcome'],
                                                              study=study_to_skip,
                                                              to_numpy=False))
        print('validation study {0}'.format(study_to_skip))
        print(val_data.shape)

    train_data.fillna(0, inplace=True)
    val_data.fillna(0, inplace=True)
    to_tensor = ToTensor()
    to_float = ToType('float')
    add_age = AdditiveUniform(-0.5, 0.5, 'age')
    add_tumor_size = AdditiveUniform(-0.5, 0.5, 'tumor_size')
    add_posOutcome = AdditiveUniformTriary(0.0, 0.05, 'posOutcome')
    add_treat = Compose([AdditiveUniformTriary(0.0, 0.05, x) for x in merged_treatments])
    compose = Compose([add_age, add_tumor_size,
        add_posOutcome, add_treat, to_tensor, to_float])
    compose_label = Compose([add_posOutcome, to_tensor, to_float])
    num_binary = len(merged_treatments + ['posOutcome'])
    transform = DataLabelCompose(compose, compose_label)

    import pdb;pdb.set_trace()
    train_set = GeneDataset(train_data, train_labels, transform, binary=num_binary)
    test_set = GeneDataset(val_data, val_labels, transform, binary=num_binary)
    return train_set, test_set


def get_metagx_dataset(ratio=0.1):
    from data_util.metagx_util import load_metagx_dataset
    from data_util import util
    data = load_metagx_dataset('/home/noskill/projects/cancer/data/metaGxBreast/', min_genes=5000)
    merged = data['merged']
    genes_list = data['genes_features']

    train_data, train_labels, val_data, val_labels = util.random_split(merged,
                                                              genes_list,
                                                              ['study'],
                                                              balance_validation=False,
                                                              balance_by_study=False,
                                                              ratio=ratio)
    to_tensor = ToTensor()
    transform = DataLabelCompose(to_tensor, to_tensor)

    # assert val_labels.mean() == 0.5
    train_set = GeneDataset(train_data, train_labels, transform)
    test_set = GeneDataset(val_data, val_labels, transform)
    return train_set, test_set


class InfoGAN(torch.nn.Module):
    def __init__(self, generator, discriminator):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator

    def forward(self, z_code, categorical, s_code):
        gen_gexs = self.generator(z_code, categorical, s_code)
        # Loss measures generator's ability to fool the discriminator
        valid, pred_categorical, pred_code = self.discriminator(gen_gexs)
        result = dict(generated=gen_gexs,
                      validity=valid,
                      categorical=pred_categorical,
                      S=pred_code)
        return result


def detach_numpy(tensor):
    return tensor.cpu().detach().numpy()


if __name__ == '__main__':
    main()
