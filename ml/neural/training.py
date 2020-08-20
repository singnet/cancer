from collections import defaultdict
import numpy
from network import compute_classifier_loss
import torch
import sys


def loss_callback_with_extractor(model, data, target, optimizer=dict(), extractor=None):
    with torch.no_grad():
        shape = data.shape
        x = data.reshape(shape[0], numpy.prod(shape[1:]))
        out = extractor(x)
    # data = out['output'] - features
    data = out['code'] # encoder
    return compute_classifier_loss(model, data, target, optimizer)


def train(model, compute_loss, device, loader, optimizer, epoch):
    model.train()
    train_loss = defaultdict(list)
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        for opt in optimizer.values():
            opt.zero_grad()
        results = compute_loss(model, data, target, optimizer)

        for key, value in  results.items():
            train_loss[key] += value  # sum up batch loss

    print("epoch {0}".format(epoch))
    for k, v in train_loss.items():
        print('\nTrain set: Average {}:'.format(k) + '{:.4f}', numpy.mean(v, axis=0))
    sys.stdout.flush()


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
    sys.stdout.flush()
