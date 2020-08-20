import numpy
import functools
import torch
from torch import nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from training import train, test
from network import ClassifierNet
from network import Net
from network import compute_classifier_loss
from data_util.util import compute_metrics
from train_genes import get_datasets
from training import loss_callback_with_extractor as loss_callback


def main():
    embed_size=80
    use_embed = True
    use_cuda = True
    epochs = 2000
    lr = 0.0001

    train_set, test_set = get_datasets(label_columns=['posOutcome'])
    # process with feature extractor
    # create new test and train sets

    num_features = train_set[0][0].shape[0]
    batch_size = 100
    test_batch_size = 50
    torch.manual_seed(7347)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('using device {0}'.format(device))

    train_loader = torch.utils.data.DataLoader(train_set,
        batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(test_set,
        batch_size=test_batch_size, shuffle=True)

    extractor = Net(activation=nn.LeakyReLU(),
            num_features=num_features,
            embed_size=embed_size).to(device)
    extractor_path = 'extractor.pt'
    extractor.load_state_dict(torch.load(extractor_path, map_location=device), strict=False)
    model = ClassifierNet(num_features=embed_size if use_embed else num_features,
                                activation=nn.LeakyReLU()).to(device)
    classifier_optimizer = optim.AdamW(model.parameters(), lr=lr)
    optimizer = {'opt': classifier_optimizer}
    schedulers = [StepLR(classifier_optimizer, step_size=50, gamma=0.9)]
    callback = compute_classifier_loss
    if use_embed:
        callback = functools.partial(loss_callback, extractor=extractor)

    for epoch in range(1, epochs + 1):
        if epoch % 50 == 0 or epoch == 1:
            test(model, callback, device, test_loader)
        train(model, callback, device, train_loader, optimizer, epoch)
        for scheduler in schedulers:
            scheduler.step()
        if epoch % 100 == 0 and epoch:
            torch.save(model.state_dict(), "classifier{0}.pt".format(epoch))
        print('learning rate: {0}'.format(scheduler.get_lr()))

if __name__ == '__main__':
    main()
