from collections import defaultdict
import numpy
import torch


def train(model, compute_loss, device, loader, optimizer, epoch):
    model.train()
    train_loss = defaultdict(list)
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        for opt in optimizer.values():
            opt.zero_grad()
        loss = compute_loss(model, data, target, optimizer)

        for key, value in  compute_loss(model, data, target).items():
            train_loss[key] += value  # sum up batch loss

    print("epoch {0}".format(epoch))
    for k, v in train_loss.items():
        print('\nTrain set: Average {}:'.format(k) + '{:.4f}', numpy.mean(v, axis=0))

def test(model, compute_loss, device, loader):
    model.eval()
    test_loss = defaultdict(list)
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            for key, value in  compute_loss(model, data, target).items():
                test_loss[key] += value  # sum up batch loss

    for k, v in test_loss.items():
        print('\nTest set: Average {}:'.format(k) + '{:.4f}', numpy.mean(v, axis=0))
