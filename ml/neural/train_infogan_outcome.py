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
from dataset import get_tamoxifen_dataset
from infogan_loss import infogan_loss_outcome, predictor_loss
from training import train, test
from torch import nn


def main():
    opt = parse_args()
    cuda = True

    skip_study = opt.skip_study
    if skip_study == -1:
        skip_study = None
    train_set, test_set = get_tamoxifen_dataset(opt)
    size = train_set.features.shape[1]
    train_loader = torch.utils.data.DataLoader(train_set,
            batch_size=opt.batch_size, shuffle=True, num_workers=10)

    # Initialize generator and discriminator
    discriminator = Discriminator(opt, size + 1,
            binary=train_set.binary, continious=train_set.continious)
    if opt.use_covars:
        generator = CovarGenOutcome(opt, size, train_set.binary, train_set.continious)
        discriminator = DiscriminatorOutcome(opt, size, train_set.binary, train_set.continious)
    else:
        assert False
    # load weights
    if hasattr(opt, 'disc_path') and opt.disc_path:
        print('loading weights from ' + opt.disc_path)
        discriminator.load_state_dict(torch.load(opt.disc_path), strict=False)
    if hasattr(opt, 'gen_path') and opt.gen_path:
        print('loading weights from ' + opt.gen_path)
        generator.load_state_dict(torch.load(opt.gen_path), strict=False)

    model = InfoGAN(generator, discriminator)
    device = 'cpu'
    if cuda:
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        device = 'cuda'


    # Optimizers
    optimizer_G = torch.optim.AdamW(generator.get_params(), lr=opt.lr)
    optimizer_D = torch.optim.AdamW(discriminator.get_params(), lr=opt.lr * 0.1)
    optimizer_info = torch.optim.AdamW(
        itertools.chain(
            generator.parameters(),
            discriminator.parameters()
            ),
        lr=opt.lr, betas=(opt.b1, opt.b2)
    )

    # ----------
    #  Training
    # ----------
    if opt.test_ratio:
        test_loader = torch.utils.data.DataLoader(test_set,
            batch_size=len(test_set), shuffle=True, num_workers=7)

    loss = infogan_loss_outcome
    optimizers = dict(G=optimizer_G, D=optimizer_D,
                      info=optimizer_info)


    for epoch in range(opt.start_epoch, opt.start_epoch + opt.n_epochs):
    #   start = time.monotonic()
        if epoch % opt.test_interval == 0 and opt.test_ratio:
            loss_test = lambda *args, **kwargs: loss(*args, **kwargs, optimizers=None, opt=opt)
            test(model, loss_test, device, test_loader)
        loss_f = lambda *args, **kwargs: loss(*args, **kwargs, opt=opt)
        train(model, loss_f, device, train_loader, optimizers, epoch + 1)
        if epoch and epoch % opt.save_interval == 0:
            torch.save(model.discriminator.state_dict(), 'discriminator{0}.pth'.format(epoch))
            torch.save(model.generator.state_dict(), 'generator{0}.pth'.format(epoch))
    #    end = time.monotonic()
    #    print('running time {0}'.format(end - start))
    torch.save(model.discriminator.state_dict(), 'discriminator.pth')
    torch.save(model.generator.state_dict(), 'generator.pth')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="config file name", required=True)
    parser.add_argument("--eval-on-train", help="compute mi on train", required=False,
            default=False, action='store_true', dest='eval_on_train')
    parser.add_argument("--eval-on-test", help="compute mi on test", required=False,
            default=False, action='store_false', dest='eval_on_train')
    parser.add_argument('--start-epoch', type=int, default=0, required=False,
            help="epoch to start the count of epochs")
    args = parser.parse_args()
    print(args)
    # load config
    with open(args.config, 'r') as f:
        y = yaml.load(f, Loader=yaml.SafeLoader)
        y['eval_on_train'] = args.eval_on_train
    opt = argparse.Namespace(start_epoch=args.start_epoch, **y)
    return opt


class InfoGAN(torch.nn.Module):
    def __init__(self, generator, discriminator):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        from sklearn.tree import DecisionTreeClassifier
        self.sk_model = DecisionTreeClassifier()

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
