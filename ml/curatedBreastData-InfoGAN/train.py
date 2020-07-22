import argparse
import itertools
import numpy as np
from torch.autograd import Variable
import torch
import yaml
from data_loader import CuratedBreastCancerData
from infogan import cuda, FloatTensor, LongTensor, weights_init_normal, to_categorical, Generator, Discriminator

parser = argparse.ArgumentParser()
parser.add_argument("--config", help="config file name")
args = parser.parse_args()
print(args)
# load config
with open(args.config, 'r') as f:
    y = yaml.load(f, Loader=yaml.SafeLoader)
opt = argparse.Namespace(**y)

# Loss functions
adversarial_loss = torch.nn.MSELoss()
categorical_loss = torch.nn.CrossEntropyLoss()
continuous_loss = torch.nn.MSELoss()

# Loss weights
lambda_cat = 1
lambda_con = 0.1

# Initialize generator and discriminator
generator = Generator(opt)
discriminator = Discriminator(opt)

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
    categorical_loss.cuda()
    continuous_loss.cuda()

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Configure data loader
data = CuratedBreastCancerData(opt.batch_size, test_split=0.0)  # train this unsupervised model on all data

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_info = torch.optim.Adam(
    itertools.chain(generator.parameters(), discriminator.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2)
)

# Static generator inputs for sampling
static_z = Variable(FloatTensor(np.zeros((opt.n_classes ** 2, opt.latent_dim))))
static_label = to_categorical(
    np.array([num for _ in range(opt.n_classes) for num in range(opt.n_classes)]), num_columns=opt.n_classes
)
static_code = Variable(FloatTensor(np.zeros((opt.n_classes ** 2, opt.code_dim))))

# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
    i = 0
    for batch_gexs in data.get_unsupervised_batch():
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
        #real_imgs = Variable(batch_gex.type(FloatTensor)).view(-1, 1024)
        real_gexs = FloatTensor(batch_gexs)
        #labels = to_categorical(labels.numpy(), num_columns=opt.n_classes)

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise and labels as generator input
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
        label_input = to_categorical(np.random.randint(0, opt.n_classes, batch_size), num_columns=opt.n_classes)
        code_input = Variable(FloatTensor(np.random.uniform(-1, 1, (batch_size, opt.code_dim))))

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
        sampled_labels = np.random.randint(0, opt.n_classes, batch_size)

        # Ground truth labels
        gt_labels = Variable(LongTensor(sampled_labels), requires_grad=False)

        # Sample noise, labels and code as generator input
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
        label_input = to_categorical(sampled_labels, num_columns=opt.n_classes)
        code_input = Variable(FloatTensor(np.random.uniform(-1, 1, (batch_size, opt.code_dim))))

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

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [info loss: %f]"
            % (epoch, opt.n_epochs, i, data.batches_in_epoch, d_loss.item(), g_loss.item(), info_loss.item())
        )
        batches_done = epoch * data.batches_in_epoch + i
        #if batches_done % opt.sample_interval == 0:
        #    sample_image(n_row=10, batches_done=batches_done)

torch.save(discriminator.state_dict(), 'discriminator.pth')
torch.save(generator.state_dict(), 'generator.pth')