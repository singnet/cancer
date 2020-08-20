import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from network import init_weights

cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def to_categorical(y, num_columns):
    """Returns one-hot encoded Variable"""
    y_cat = np.zeros((y.shape[0], num_columns))
    if isinstance(y, torch.Tensor):
        y = y.cpu()
    y_cat[range(y.shape[0]), y] = 1.0

    return Variable(FloatTensor(y_cat))


class Generator(nn.Module):
    def __init__(self, opt, size, continious=0, binary=0):
        """
        Construct generator
        Parameters
        ---------------
        opt:  argparse.Namespace

        size: int
            number of outputs
        continious: int
            number of continious outputs in the total of size output neurons
        binary: int
            number of binary outputs in the total of size output neurons
        """
        super(Generator, self).__init__()
        input_dim = opt.latent_dim + opt.n_classes + opt.code_dim

        self.dense_block = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, size),
        )
        self.binary = binary
        self.continious = continious
        init_weights(self)

    def forward(self, noise, labels, code):
        gen_input = torch.cat((noise, labels, code), -1)
        img = self.dense_block(gen_input)
        result = img
        if self.binary:
            s = torch.tanh(img[:, -self.binary:])
            result = torch.cat([img[:, :-self.binary], s], dim=1)
        return result


class StudyGen(nn.Module):
    def __init__(self, opt, size, continious=0, binary=0):
        super().__init__()
        assert not binary
        assert not continious
        self.n_studies = opt.n_classes
        self.study_param_a = torch.tensor(np.random.random((n_studies, size - binary - continious)),
                requires_grad=True)
        self.study_param_b = torch.tensor(np.random.random((n_studies, size - binary - continious)),
                requires_grad=True)

        input_dim = opt.latent_dim + opt.code_dim

        self.dense_block = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, size),
        )
        self.binary = binary
        self.continious = continious
        init_weights(self)

    def forward(self, noise, labels, code):
        gen_input = torch.cat((noise, code), -1)
        img = self.dense_block(gen_input)
        result = img * self.study_param_a[labels] + self.study_param_b[labels]
        return result


class Discriminator(nn.Module):
    def __init__(self, opt, size, continious=0, binary=0):
        super().__init__()

        assert not binary
        self.dense_blocks = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(size - binary, 256)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Linear(256, 256)),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Output layers
        self.adv_layer = nn.Linear(opt.code_dim, 1)
        self.category_layer = nn.Sequential(nn.Linear(256, opt.n_classes), nn.Softmax(dim=1))
        self.latent_layer = nn.Sequential(nn.Linear(256, opt.code_dim))
        self.binary = binary
        if binary:
            self.adv_layer = nn.Sequential(nn.Linear(opt.code_dim + binary, 128),
                                           nn.LeakyReLU(),
                                           nn.Linear(128, 128),
                                           nn.LeakyReLU(),
                                           nn.Linear(128, 1),
                                           nn.Sigmoid())
        init_weights(self)

    def forward(self, img):
        genes = img[:, :-self.binary]
        binary = img[:, -self.binary:]
        out = self.dense_blocks(genes)
        #out = out.view(out.shape[0], -1)
        categorical_code = self.category_layer(out)
        latent_code = self.latent_layer(out)
        if self.binary:
            assert False # handle binary and continious
        validity = self.adv_layer(out)
        return validity, categorical_code, latent_code

    def hidden(self, img):
        out = self.dense_blocks(img)
        return out


class StudyDisc(Descriminator):
    def __init__(self, opt, size, continious=0, binary=0):
        super().__init__(self, opt, size, continious=0, binary=0)

    def forward(self, img):
        genes = img[:, :-self.binary]
        binary = img[:, -self.binary:]
        out = self.dense_blocks(genes)
        latent_code = self.latent_layer(out)
        if self.binary:
            assert False # handle binary and continious
        validity = self.adv_layer(out)
        return validity, None, latent_code

