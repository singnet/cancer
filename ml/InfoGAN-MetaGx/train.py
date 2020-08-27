import os
import argparse
import itertools
import numpy as np

import torch
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

from metagxdataset import MetaGxDataset, metaGxConfigLoader
from infogan import cuda, FloatTensor, LongTensor, weights_init_normal, to_categorical, Generator, Discriminator

parser = argparse.ArgumentParser()
parser.add_argument("config", help="config file name")
args = parser.parse_args()

# load config
config = metaGxConfigLoader(args.config)
print(config)

# load dataset
dataset = MetaGxDataset(config.gexs_csv, config.treatments_csv, config.studies_csv, config.batch_size, config.normalization)

# Loss functions
adversarial_loss = torch.nn.MSELoss()
categorical_loss = torch.nn.CrossEntropyLoss()
continuous_loss = torch.nn.MSELoss()

# Loss weights
lambda_cat = 1
lambda_con = 0.1

# Initialize generator and discriminator
categorical_size = config.n_classes if config.n_classes else dataset.n_studies
print('categorical_size:', categorical_size)
#import ipdb; ipdb.set_trace()

generator = Generator(
    config.latent_dim,
    categorical_size,
    config.code_dim,
    dataset.n_genes
)
discriminator = Discriminator(
    categorical_size,
    config.code_dim,
    dataset.n_genes
)

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
    categorical_loss.cuda()
    continuous_loss.cuda()

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=config.lr, betas=(config.b1, config.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=config.lr, betas=(config.b1, config.b2))
optimizer_info = torch.optim.Adam(
    itertools.chain(generator.parameters(), discriminator.parameters()), lr=config.lr, betas=(config.b1, config.b2)
)

# ----------
#  Training
# ----------
summary = SummaryWriter('summary/s{0:02d}_c{1:02d}_{2}'.format(
        config.code_dim,
        categorical_size,
        os.path.split(config.gexs_csv)[-1].split('.')[0]
    )
)
for epoch in range(config.n_epochs):
    print('\repoch {0}'.format(epoch + 1), end='', flush=True)
    i = 0
    for batch_gexs in dataset.get_gexs_batch(config.batch_size):
        i += 1

        batch_size = batch_gexs.shape[0]

        # Adversarial ground truths
        valid = Variable(  # noisy labels
            FloatTensor(batch_size, 1).fill_(1.0) - FloatTensor(np.random.uniform(0.0, 0.05, (batch_size, 1))), requires_grad=False
        )
        fake = Variable(
            FloatTensor(batch_size, 1).fill_(0.0) + FloatTensor(np.random.uniform(0.0, 0.05, (batch_size, 1))), requires_grad=False
        )

        # Configure input
        real_gexs = FloatTensor(batch_gexs)

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise and labels as generator input
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, config.latent_dim))))
        label_input = to_categorical(np.random.randint(0, categorical_size, batch_size), num_columns=categorical_size)
        code_input = Variable(FloatTensor(np.random.uniform(-1, 1, (batch_size, config.code_dim))))

        # Generate a batch of images
        gen_gexs = generator(z, label_input, code_input)

        # Loss measures generator's ability to fool the discriminator
        validity, _, _ = discriminator(gen_gexs)
        g_loss = adversarial_loss(validity, valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Loss for real images
        real_pred, _, _ = discriminator(real_gexs)
        d_real_loss = adversarial_loss(real_pred, valid)

        # Loss for fake images
        fake_pred, _, _ = discriminator(gen_gexs.detach())
        d_fake_loss = adversarial_loss(fake_pred, fake)

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        # ------------------
        # Information Loss
        # ------------------

        optimizer_info.zero_grad()

        # Sample labels
        sampled_labels = np.random.randint(0, categorical_size, batch_size)

        # Ground truth labels
        gt_labels = Variable(LongTensor(sampled_labels), requires_grad=False)

        # Sample noise, labels and code as generator input
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, config.latent_dim))))
        label_input = to_categorical(sampled_labels, num_columns=categorical_size)
        code_input = Variable(FloatTensor(np.random.uniform(-1, 1, (batch_size, config.code_dim))))

        gen_gexs = generator(z, label_input, code_input)
        _, pred_label, pred_code = discriminator(gen_gexs)

        info_loss = lambda_cat * categorical_loss(pred_label, gt_labels) + lambda_con * continuous_loss(
            pred_code, code_input
        )

        info_loss.backward()
        optimizer_info.step()

        # --------------
        # Log Progress
        # --------------

        # print(
        #     "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [info loss: %f]"
        #     % (epoch, config.n_epochs, i, dataset.batches_per_epoch, d_loss.item(), g_loss.item(), info_loss.item())
        # )
        batches_done = epoch * dataset.batches_per_epoch + i
        summary.add_scalar('losses/discriminator', d_loss.item(), batches_done)
        summary.add_scalar('losses/generator', g_loss.item(), batches_done)
        summary.add_scalar('losses/info', info_loss.item(), batches_done)

print('\rdone', flush=True)

save_path = 'checkpoints/s{0:02d}_c{1:02d}_{2}'.format(
    config.code_dim,
    categorical_size,
    os.path.split(config.gexs_csv)[-1].split('.')[0]
)
os.makedirs(save_path, exist_ok=True)
torch.save(discriminator.state_dict(), os.path.join(save_path, 'discriminator.pth'))
torch.save(generator.state_dict(), os.path.join(save_path, 'generator.pth'))
