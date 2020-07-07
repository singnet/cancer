import torch
from collections import defaultdict
import numpy
from torch import nn
from metrics import compute_metrics


def init_weights(self):
    def init_weights(m):
        if hasattr(m, 'weight') and not isinstance(m, (torch.nn.BatchNorm2d,
                                                       torch.nn.BatchNorm1d)):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
    self.apply(init_weights)


class Net(nn.Module):
    def __init__(self, num_features,
            activation=lambda x: x,
            decoder_output_activation=lambda x: x,
            embed_size=2):
        super(Net, self).__init__()
        self.activation = activation
        self.dec_out_activation = decoder_output_activation
        # encoder
        self.enc1 = nn.Linear(num_features, 1000)
        self.enc2 = nn.Linear(1000, 1000)
        self.enc3 = nn.Linear(1000, embed_size)
        # decoder
        self.dec1 = nn.Linear(embed_size, 1000)
        self.dec2 = nn.Linear(1000, 1000)
        self.dec3 = nn.Linear(1000, num_features)
        # discriminator
        self.disc1 = nn.Linear(embed_size, 1000)
        self.disc2 = nn.Linear(1000, 1000)
        self.disc3 = nn.Linear(1000, 1)

        init_weights(self)

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
        batch = batch.to(next(self.parameters()))
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
        x = self.dec_out_activation(self.dec3(x))
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
    return {'loss_reconst': [loss]}


def compute_classifier_loss(model, data, target, optimizer=dict()):
    shape = data.shape
    x = data.reshape(shape[0], numpy.prod(shape[1:]))
    out = model(x)
    output = out['output']
    # cross entropy
    loss = - (target * torch.log(output) + (1 - target) * torch.log(1 - output)).mean()
    if optimizer:
        loss.backward()
    for opt in optimizer.values():
        opt.step()
    result = defaultdict(list)
    out_bin = output.detach().cpu().numpy() > 0.5
    compute_metrics(result, target.detach().cpu().numpy(), out_bin)
    result['loss'].append(loss.cpu().detach().numpy())
    return result


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
    loss_desc = 0.5 * torch.mean(- torch.log(1 - desc_real + delta)) + 0.5 * torch.mean(- torch.log(desc_fake + delta))
    # 2) update the encoder using generator
    loss_gen = (- torch.log(desc_real)).mean()
    result = dict()
    result.update(dict(reconstruction_loss=loss_reconst))
    result.update(dict(discrimination_loss=loss_desc))
    result.update(dict(encoder_loss=loss_gen))
    for k, loss in result.items():
        if k[:3] in optimizer:
            loss.backward(retain_graph=True)
    for opt in optimizer.values():
        opt.step()
    result.update(dict(desc_real=desc_real.mean(),
                  desc_fake=desc_fake.mean()))
    return {k: [v.detach().cpu().numpy()] for (k, v) in result.items()}


class MnistNet(Net):
    def __init__(self, activation):
        super(MnistNet, self).__init__(num_features=28*28,
                activation=activation,
                decoder_output_activation=torch.sigmoid,
                embed_size=2)


class ClassifierNet(nn.Module):
    def __init__(self, num_features, activation):
        super().__init__()
        self.activation = activation
        self.l1 = nn.Linear(num_features, 1000)
        self.l2 = nn.Linear(1000, 1000)
        self.l3 = nn.Linear(1000, 1)

    def forward(self, x):
        x = x.to(next(self.parameters()))
        x = self.activation(self.l1(x))
        x = self.activation(self.l2(x))
        x = torch.sigmoid(self.l3(x))
        result = dict(output=x)
        return result

