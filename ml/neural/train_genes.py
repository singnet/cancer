import argparse
from collections import defaultdict
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from network import Net, compute_loss_adversarial_enc, compute_loss_autoenc
from training import train, test
from dataset import GeneDataset
from transform import DataLabelCompose, ToTensor
from data_util.util import load_merged_dataset, random_split



def get_datasets(label_columns=['study'], transform=None, balance_train=True):
    cancer_data_dir = '/home/noskill/projects/cancer.old/data'
    dataset_dict = load_merged_dataset(cancer_data_dir)
    merged = dataset_dict['merged']
    bmc = dataset_dict['bmc']
    genes_features = dataset_dict['genes_features']
    genes_columns = genes_features.columns.to_list()[1:]
    feature_columns = genes_columns
    if transform is None:
       to_tensor = ToTensor()
       transform = DataLabelCompose(to_tensor, to_tensor)

    train_data, train_labels, val_data, val_labels = random_split(merged,
            feature_columns, label_columns, balance_train=balance_train, balance_by_study=False)
    assert val_labels.mean() == 0.5
    train_set = GeneDataset(train_data, train_labels, transform)
    test_set = GeneDataset(val_data, val_labels, transform)
    return train_set, test_set


def main():
    # prepare dataset
    train_adversarial = 1
    use_cuda = True
    epochs = 2000
    lr = 0.0005

    train_set, test_set = get_datasets(balance_train=True)

    num_features = train_set[0][0].shape[0]
    batch_size = 400
    test_batch_size = 50
    torch.manual_seed(7347)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('using device {0}'.format(device))

    train_loader = torch.utils.data.DataLoader(train_set,
        batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(test_set,
        batch_size=test_batch_size, shuffle=True)

    model = Net(activation=nn.LeakyReLU(),
            num_features=num_features,
            embed_size=80).to(device)
    PATH = None
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
        schedulers = [StepLR(x, step_size=70, gamma=0.9) for x in tmp]
    else:
        compute_loss = compute_loss_autoenc
        optimizer = {'rec': reconstruction_optimizer}
        schedulers = [StepLR(reconstruction_optimizer, step_size=50, gamma=0.9)]

    for epoch in range(1, epochs + 1):
        if epoch % 50 == 0:
            test(model, compute_loss, device, test_loader)
        train(model, compute_loss, device, train_loader, optimizer, epoch)
        for scheduler in schedulers:
            scheduler.step()
        if epoch % 100 == 0 and epoch:
            torch.save(model.state_dict(), "mnist_cnn{0}.pt".format(epoch))
        print('learning rate: {0}'.format(scheduler.get_lr()))


if __name__ == '__main__':
    main()
