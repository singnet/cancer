import argparse
from collections import defaultdict
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from network import MnistNet as Net, compute_loss_adversarial_enc, compute_loss_autoenc
from training import train, test


def main():
    train_adversarial = 1
    use_cuda = True
    epochs = 200
    lr = 0.005

    batch_size = 100
    test_batch_size = 100
    torch.manual_seed(7347)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('using device {0}'.format(device))

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                       ])),
        batch_size=batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, transform=transforms.Compose([
                           transforms.ToTensor(),
                       ])),
        batch_size=test_batch_size, shuffle=True, **kwargs)

    model = Net(activation=nn.LeakyReLU()).to(device)
    PATH = 'mnist_cnn100.pt.bak'
    # model.load_state_dict(torch.load(PATH, map_location=device), strict=False)

    reconstruction_optimizer = optim.AdamW(model.autoenc_params(), lr=lr)
    discriminative_optimizer = optim.AdamW(model.disc_params(), lr=lr * 0.1)
    encoder_optimizer = optim.AdamW(model.enc_params(), lr=lr * 0.1)

    if train_adversarial:
        compute_loss = compute_loss_adversarial_enc
        optimizer = {'rec': reconstruction_optimizer,
                     'dis': discriminative_optimizer,
                     'enc': encoder_optimizer}
        tmp = [reconstruction_optimizer,
                discriminative_optimizer,
                encoder_optimizer]
        schedulers = [StepLR(x, step_size=5, gamma=0.9) for x in tmp]

    else:
        compute_loss = compute_loss_autoenc
        optimizer = {'rec': reconstruction_optimizer}
        schedulers = [StepLR(reconstruction_optimizer, step_size=5, gamma=0.9)]

    for epoch in range(1, epochs + 1):
        if epoch % 5 == 0:
            test(model, compute_loss, device, test_loader)
        train(model, compute_loss, device, train_loader, optimizer, epoch)
        for scheduler in schedulers:
            scheduler.step()
        if epoch % 10 == 0 and epoch:
            torch.save(model.state_dict(), "mnist_cnn{0}.pt".format(epoch))
        print('learning rate: {0}'.format(scheduler.get_lr()))


if __name__ == '__main__':
    main()
