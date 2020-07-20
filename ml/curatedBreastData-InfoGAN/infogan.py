import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

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
    y_cat[range(y.shape[0]), y] = 1.0

    return Variable(FloatTensor(y_cat))


class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()
        input_dim = opt.latent_dim + opt.n_classes + opt.code_dim

        self.init_size = opt.img_size // 4  # Initial size before upsampling
        #self.l1 = nn.Sequential(nn.Linear(input_dim, 128 * self.init_size ** 2))
        #self.l1 = nn.Sequential(nn.Linear(input_dim, 128), nn.LeakyReLU(0.2, inplace=True))

        self.dense_block = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 8832),
            #nn.Tanh(),
        )

    def forward(self, noise, labels, code):
        gen_input = torch.cat((noise, labels, code), -1)
        #out = self.l1(gen_input)
        #out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        #img = self.conv_blocks(out)
        img = self.dense_block(gen_input)
        return img


class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()

        self.dense_blocks = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(8832, 512)),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.BatchNorm1d(512),
            #nn.Dropout(0.25),
            nn.utils.spectral_norm(nn.Linear(512, 512)),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.BatchNorm1d(512, 0.8),
            #nn.Dropout(0.25),
            #nn.utils.spectral_norm(nn.Linear(512, 512)),
            #nn.LeakyReLU(0.2, inplace=True),
            #nn.BatchNorm1d(512, 0.8),
            #nn.Dropout(0.25),
        )

        # Output layers
        self.adv_layer = nn.Sequential(nn.Linear(512, 1))
        self.aux_layer = nn.Sequential(nn.Linear(512, opt.n_classes), nn.Softmax())
        self.latent_layer = nn.Sequential(nn.Linear(512, opt.code_dim))

    def forward(self, img):
        out = self.dense_blocks(img)
        #out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        label = self.aux_layer(out)
        latent_code = self.latent_layer(out)

        return validity, label, latent_code

    def hidden(self, img):
        out = self.dense_blocks(img)
        #out = out.view(out.shape[0], -1)
        return out
