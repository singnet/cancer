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


def init_weights_xavier(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


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
            nn.Linear(input_dim, 512),
            #nn.BatchNorm1d(512),
            nn.LeakyReLU(inplace=True),
            nn.Linear(512, 512),
            #nn.BatchNorm1d(512),
            nn.LeakyReLU(inplace=True),
            nn.Linear(512, size - binary - continious),
        )
        self.binary = binary
        self.continious = continious
        init_weights(self)

    def forward(self, noise, labels, code):
        gen_input = torch.cat((noise, labels, code), -1)
        img = self.dense_block(gen_input)
        result = img
        assert not self.continious
        assert not self.binary
        return result

    def get_params(self):
        return self.parameters()


class CovarGen(Generator):
    def __init__(self, opt, size, continious=0, binary=0):
        super().__init__(opt, size, continious=continious, binary=binary)
        inp_size = opt.latent_dim + opt.code_dim
        self.gen_covars = nn.Sequential(nn.Linear(inp_size, 256), nn.LeakyReLU(),
            # one for positive and one for negative outcome
            nn.Linear(256, (continious + binary) * 2))
        init_weights(self)

    def forward(self, noise, labels, code):
        gen_input = torch.cat((labels, code), -1)
        gexs = self.dense_block(gen_input)
        # duplicate gexs
        gexs = torch.stack([gexs, gexs], dim=0)
        gexs = torch.flatten(gexs, end_dim=1)
        covar_input = torch.cat((code, noise), dim=1)
        covars = self.gen_covars(covar_input)
        covars = torch.stack([covars[:, :self.binary + self.continious],
                              covars[:, self.binary + self.continious:]], dim=0)

        covars = torch.flatten(covars, end_dim=1)
        if self.binary:
            contin = covars[:, :self.continious]
            assert covars.shape[-1] - self.binary == self.continious
            bins = torch.tanh(covars[:, -self.binary:])
            outcomes = torch.ones(len(contin) // 2).to(contin)
            outcomes = torch.stack([outcomes, outcomes * 0], dim=0)
            outcomes = outcomes.flatten().unsqueeze(1)
        result = torch.cat([gexs, contin, bins, outcomes], dim=1)
        return result

    def get_params(self):
        return self.gen_covars.parameters()


class StudyGen(nn.Module):
    def __init__(self, opt, size, continious=0, binary=0):
        super().__init__()
        assert not binary
        assert not continious
        input_dim = opt.latent_dim + opt.code_dim
        self.dense_block = nn.Sequential(
            nn.Linear(input_dim, 512),
            #nn.BatchNorm1d(512),
            nn.LeakyReLU(inplace=True),
            nn.Linear(512, 512),
            #nn.BatchNorm1d(512),
            nn.LeakyReLU(inplace=True),
            nn.Linear(512, size),
        )
        self.binary = binary
        self.continious = continious

        self.degree = opt.degree
        self.n_studies = opt.n_classes
        self.poly_param = nn.Parameter(torch.tensor(np.random.random((self.degree + 1, self.n_studies, size - binary - continious)).astype(np.float32),  requires_grad=True))
        init_weights(self)

    def forward(self, noise, labels, code):
        gen_input = torch.cat((noise, code), -1)
        img = torch.abs(self.dense_block(gen_input))
        lab = labels.nonzero()[:,1]
        result = sum([img ** i * torch.abs(self.poly_param[i][lab]) for i in range(1, self.degree + 1)])
        result = result + self.poly_param[0][lab]
        return result

    def get_params(self):
        return self.parameters()


class Discriminator(nn.Module):
    def __init__(self, opt, size, continious=0, binary=0):
        super().__init__()

        self.dense_blocks = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(size - binary - continious, 512)),
            nn.LeakyReLU(inplace=True),
            nn.utils.spectral_norm(nn.Linear(512, 512)),
            nn.LeakyReLU(inplace=True),
            nn.utils.spectral_norm(nn.Linear(512, 512)),
            nn.LeakyReLU(inplace=True),
        )

        # Output layers
        self.adv_layer = nn.Sequential(nn.Linear(512, 1), nn.Sigmoid())
        self.category_layer = nn.Sequential(nn.Linear(512, opt.n_classes))
        tmp = (nn.Linear(512, 512), nn.LeakyReLU(), nn.Linear(512, opt.code_dim))
        if opt.distribution == 'uniform':
            tmp = tmp + (nn.Tanh(), )
        self.latent_layer = nn.Sequential(*tmp)
        self.binary = binary
        self.continious = continious
        if binary:
            self.adv_layer_covars = nn.Sequential(nn.Linear(opt.code_dim + binary + continious + 1, 256),
                                           nn.LeakyReLU(),
                                           nn.Linear(256, 256),
                                           nn.LeakyReLU(),
                                           nn.Linear(256, 1),
                                           nn.Sigmoid())
        init_weights(self)

    def forward(self, img):
        if self.binary:
            genes = img[:, :-self.binary - self.continious]
            # -1 for posOutcome
            covars = img[:, -self.binary - self.continious - 1:]
        else:
            genes = img
        out = self.dense_blocks(genes)
        categorical_code = self.category_layer(out)
        latent_code = self.latent_layer(out)
        validity = self.adv_layer(out)
        if self.binary or self.continious:
            validity = self.adv_layer_covars(torch.cat([latent_code, covars]))
        return validity, categorical_code, latent_code

    def extract_features(self, gexs):
        valid, categorical, latent_code = self(torch.as_tensor(gexs).to(next(self.parameters())))
        return latent_code
        # return torch.cat([latent_code, categorical, torch.as_tensor(gexs).to(latent_code)], dim=1)
        # return torch.cat([latent_code, torch.as_tensor(gexs).to(latent_code)],
        #        dim=1)

    def get_params(self):
        return self.parameters()


class CovarDisc(Discriminator):
    def get_params(self):
        return self.adv_layer_covars.parameters()


class CovarGenOutcome(nn.Module):
    def __init__(self, opt, size, continious=0, binary=0):
        super().__init__()
        self.apply(init_weights_xavier)
        input_dim = opt.latent_dim + opt.n_classes + opt.code_dim
        self.opt = opt

        self.dense_block = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(inplace=True),
            #nn.BatchNorm1d(512),
            nn.Linear(512, 512),
            nn.LeakyReLU(inplace=True),
            #nn.BatchNorm1d(512),
        )

        self.gene_layer = nn.Linear(512, size - binary - continious)
        self.outcome_layer = nn.Sequential(nn.Linear(512, 256),
                                           nn.LeakyReLU(inplace=True),
                                           #nn.BatchNorm1d(256),
                                           nn.Linear(256, 1),
                                           nn.LeakyReLU(inplace=True))

        assert(continious == 0)
        self.apply(init_weights_xavier)

    def forward(self, noise, labels, code):
        assert self.opt.latent_dim == noise.shape[1]
        assert self.opt.n_classes == labels.shape[1]
        assert self.opt.code_dim == code.shape[1]

        gen_input = torch.cat((noise, labels, code), -1)
        dense = self.dense_block(gen_input)
        genes = self.gene_layer(dense)
        outcomes = torch.sigmoid(self.outcome_layer(dense))
        result = dict()
        result['genes'] = genes
        result['outcomes'] = outcomes
        return result

    def get_params(self):
        return self.parameters()


class DiscriminatorOutcome(nn.Module):
    def __init__(self, opt, size, continious=0, binary=0):
        super().__init__()
        self.dense_blocks = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(size - binary - continious, 512),
            nn.LeakyReLU(inplace=False),
            #nn.BatchNorm1d(512),

            nn.Linear(512, 512),
            nn.Dropout(p=0.2),
            nn.LeakyReLU(inplace=False),
            #nn.BatchNorm1d(512),

            nn.Linear(512, 512),
            nn.Dropout(p=0.2),
            nn.LeakyReLU(inplace=False),
            #nn.BatchNorm1d(512),

            nn.Linear(512, 512),
            nn.LeakyReLU(inplace=False),
            #nn.BatchNorm1d(512),
        )

        self.disc_outcome_genes = nn.Sequential(
                nn.Linear(512 + 1, 512),
                nn.LeakyReLU(inplace=False),
                #nn.BatchNorm1d(512),

                nn.Linear(512, 256),
                nn.Dropout(p=0.2),
                nn.LeakyReLU(inplace=False),
                #nn.BatchNorm1d(256),

                nn.Linear(256, 256),
                nn.LeakyReLU(inplace=False),
                #nn.BatchNorm1d(256),

                nn.Linear(256, 1),
                torch.nn.Sigmoid())

        self.reconstruct = nn.Sequential(
                nn.Dropout(p=0.7),
                nn.Linear(512, 128),
                nn.LeakyReLU(inplace=False),
                #nn.BatchNorm1d(128),
                nn.Linear(128, 128),
                nn.Dropout(p=0.5),
                nn.LeakyReLU(inplace=False),
                #nn.BatchNorm1d(128),
                nn.Linear(128, opt.n_classes),
                nn.Softmax(dim=1))
        self.apply(init_weights_xavier)

    def forward(self, gen_out):
        genes = gen_out['genes']
        outcomes = gen_out['outcomes']
        dense = self.dense_blocks(genes)
        # stack dense with outcome
        sample = torch.cat([dense, outcomes], dim=1)
        fake_real = self.disc_outcome_genes(sample)
        categorical = self.reconstruct(dense)
        return fake_real, categorical, None

    def compute_code(self, genes):
        dense = self.dense_blocks(genes)
        categorical = self.reconstruct(dense)
        return categorical

    def get_params(self):
        return self.parameters()

