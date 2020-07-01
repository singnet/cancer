import argparse
from collections import defaultdict
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
        self.disc1 = nn.Linear(embed_size, 1000)
        self.disc2 = nn.Linear(1000, 1000)
        self.disc3 = nn.Linear(1000, 1)

    def _get_params_by_layers(self, *args):
        res = []
        for l in args:
            for param in l.parameters():
                yield param

    def enc_params(self):
        return self._get_params_by_layers(self.enc1,
                self.enc2, self.enc3)

    def dec_params(self):
        return self._get_params_by_layers(self.dec1,
                self.dec2, self.dec3)

    def disc_params(self):
        return self._get_params_by_layers(self.disc1,
                self.disc2, self.disc3)

    def autoenc_params(self):
        for p in self.enc_params():
            yield p
        for p in self.dec_params():
            yield p

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
        x = self.enc3(x)
        return x

    def decode(self, x):
        x = self.activation(self.dec1(x))
        x = self.activation(self.dec2(x))
        x = torch.sigmoid(self.dec3(x))
        return x

    def descriminate(self, code):
        x = self.activation(self.disc1(code))
        x = self.activation(self.disc2(x))
        x = torch.sigmoid(self.disc3(x))
        return x


def compute_loss_autoenc(model, data, target, optimizer=dict()):
    shape = data.shape
    x = data.reshape(shape[0], numpy.prod(shape[1:]))
    out  = model(x)
    output = out['output']
    loss = torch.mean((x - output) ** 2)
    for opt in optimizer.values():
        loss.backward()
        opt.step()
    return {'loss_reconst': loss}


def generate_code(code):
    return torch.normal(torch.zeros_like(code), torch.ones_like(code) * 5)


def compute_loss_adversarial_enc(model, data, target, optimizer=dict()):
    """
    Compute adversarial autoencoder loss

    Both, the adversarial network and the autoencoder are trained jointly with SGD
    in two phases – the reconstruction phase and the regularization phase –
    executed on each mini-batch.

    In the reconstruction phase, the autoencoder updates the encoder and the decoder
    to minimize the reconstruction error of the inputs.

    In the regularization phase, the adversarial network first updates its discriminative
    network to tell apart the true samples (generated using the prior)
    from the generated samples (the hidden codes computed by the autoencoder).
    The adversarial network then updates its generator (which is also the encoder of the autoencoder) to confuse the discriminative network.
    """
    shape = data.shape
    x = data.reshape(shape[0], numpy.prod(shape[1:]))
    out  = model(x)
    output = out['output']
    code = out['code']
    # reconstruction phase
    loss_reconst = torch.mean((x - output) ** 2)
    # regularization phase
    # 1) update discriminative network
    generated_code = generate_code(code)
    desc_real = model.descriminate(code)
    desc_fake = model.descriminate(generated_code)
    delta = 0.00000000001
    # expecting 0 for real and 1 for fake
    loss_desc = torch.mean(- 0.5 * torch.log(1 - desc_real + delta) - 0.5 * torch.log(desc_fake + delta))
    # 2) update the encoder using generator
    loss_gen = - torch.log(desc_real).mean()
    result = dict()
    result.update(dict(reconstruction_loss=loss_reconst))
    result.update(dict(discrimination_loss=loss_desc))
    if optimizer:
        for k, loss in result.items():
            opt = optimizer[k[:3]]
            loss.backward(retain_graph=True)
            opt.step()
    result.update(dict(encoder_loss=loss_gen))
    result.update(dict(desc_real=desc_real.mean(),
                  desc_fake=desc_fake.mean()))
    return result


def train(model, compute_loss, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        for opt in optimizer.values():
            opt.zero_grad()
        loss = compute_loss(model, data, target, optimizer)
        if batch_idx % 100 == 0:
            for k,v in loss.items():
                print(('Train Epoch: {} [{}/{} ({:.0f}%)]\t' + str(k) + ': {:.6f}').format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), v.item()))


def test(model, compute_loss, device, test_loader):
    model.eval()
    test_loss = defaultdict(float)
    correct = 0
    i = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            for key, value in  compute_loss(model, data, target).items():
                test_loss[key] += value.item()  # sum up batch loss
            i += 1
            # pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            #correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss = {k: v / i for (k, v) in test_loss.items()}
    for k, v in test_loss.items():
        print('\nTest set: Average {}:'.format(k) + '{:.4f}', v)
#    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#        test_loss, correct, len(test_loader.dataset),
#        100. * correct / len(test_loader.dataset)))


def main():
    train_adversarial = 1
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
    PATH = 'mnist_cnn100.pt.bak'
    # model.load_state_dict(torch.load(PATH, map_location=device), strict=False)

    reconstruction_optimizer = optim.AdamW(model.autoenc_params(), lr=lr)
    discriminative_optimizer = optim.AdamW(model.disc_params(), lr=lr)
    encoder_optimizer = optim.AdamW(model.enc_params(), lr=lr)

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
        test(model, compute_loss, device, test_loader)
        train(model, compute_loss, device, train_loader, optimizer, epoch)
        for scheduler in schedulers:
            scheduler.step()
        if epoch % 10 == 0 and epoch:
            torch.save(model.state_dict(), "mnist_cnn{0}.pt".format(epoch))
        print('learning rate: {0}'.format(scheduler.get_lr()))


if __name__ == '__main__':
    main()
