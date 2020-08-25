import argparse
import time
import os
import itertools
import numpy as np
from torch.autograd import Variable
import torch
import yaml
from infogan import *
from train_genes import get_datasets
from dataset import get_merged_common_dataset
from infogan_loss import infogan_loss, predictor_loss
from training import train, test
from torch import nn


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
    if hasattr(opt, 'train_polynomial_model') and opt.train_polynomial_model:
        generator = StudyGen(opt, size, train_set.binary)
    else:
        generator = Generator(opt, size, train_set.binary)
    discriminator = Discriminator(opt, size, train_set.binary)
    # load weights
    if hasattr(opt, 'disc_path') and opt.disc_path:
       discriminator.load_state_dict(torch.load(opt.disc_path))
    if hasattr(opt, 'gen_path') and opt.gen_path:
       generator.load_state_dict(torch.load(opt.gen_path))

    model = InfoGAN(generator, discriminator)
    device = 'cpu'

    if cuda:
        generator = generator.cuda()
        discriminator = discriminator.cuda()
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
        if (epoch + 1) % opt.test_interval == 0 and opt.test_ratio:
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
