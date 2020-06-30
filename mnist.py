import argparse
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR


class Net(nn.Module):
    def __init__(self,
            activation=lambda x: x,
            embed_size=2):
        super(Net, self).__init__()
        self.activation = activation
        mnist_size = 28 * 28
        self.enc1 = nn.Linear(mnist_size, 1000)
        self.enc2 = nn.Linear(1000, 1000)
        self.enc3 = nn.Linear(1000, embed_size)
        self.dec1 = nn.Linear(embed_size, 1000)
        self.dec2 = nn.Linear(1000, 1000)
        self.dec3 = nn.Linear(1000, mnist_size)

    def forward(self, batch):
        """
        batch: torch.Tensor
            batch_size x n
        """
        code = self.encode(batch)
        x = self.decode(code)
        return {'code': code, 'output': x}

    def encode(self, x):
        x = self.activation(self.enc1(x))
        x = self.activation(self.enc2(x))
        x = self.activation(self.enc3(x))
        return x

    def decode(self, x):
        x = self.activation(self.dec1(x))
        x = self.activation(self.dec2(x))
        x = torch.sigmoid(self.dec3(x))
        return x


def compute_loss(model, data, target):
    shape = data.shape
    x = data.reshape(shape[0], numpy.prod(shape[1:]))
    out  = model(x)
    output = out['output']
    loss = torch.mean((x - output) ** 2)
    return loss


def train(model, compute_loss, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        loss = compute_loss(model, data, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, compute_loss, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    i = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            test_loss += compute_loss(model, data, target).item()  # sum up batch loss
            i += 1
            # pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            #correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= i
    print('\nTest set: Average loss: {:.4f}', test_loss)
#    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#        test_loss, correct, len(test_loader.dataset),
#        100. * correct / len(test_loader.dataset)))


def main():
    use_cuda = True
    epochs = 100
    lr = 0.0005

    batch_size = 100
    test_batch_size = 100
    torch.manual_seed(7347)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('using device {0}'.format(device))

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                       ])),
        batch_size=batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                       ])),
        batch_size=test_batch_size, shuffle=True, **kwargs)

    model = Net(activation=nn.LeakyReLU()).to(device)
    PATH = 'mnist_cnn100.pt'
    model.load_state_dict(torch.load(PATH, map_location=device))

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.9)

    for epoch in range(1, epochs + 1):
        test(model, compute_loss, device, test_loader)
        train(model, compute_loss, device, train_loader, optimizer, epoch)
        scheduler.step()
        if epoch % 10 == 0 and epoch:
            torch.save(model.state_dict(), "mnist_cnn{0}.pt".format(epoch))
        print('learning rate: {0}'.format(scheduler.get_lr()))


if __name__ == '__main__':
    main()
