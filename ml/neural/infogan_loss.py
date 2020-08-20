import torch
from sklearn.metrics import accuracy_score
import numpy
from infogan import to_categorical
from metrics import compute_metrics, compute_auc
from collections import defaultdict
from torch.nn import functional as F
np = numpy


def cross_entropy(y, y_pred, eps=0.0000000001):
    result = -(y * torch.log(y_pred + eps) + (1 - y) * torch.log(1 - y_pred + eps))
    return result


def invert_half_outcomes(real_gexs, valid):
    orig_outcome = real_gexs[:, -1].clone()
    half = len(real_gexs) // 2
    real_gexs[:half, -1] *= -1
    new_outcome = real_gexs[:, -1]
    # valid is in range (0.95, 1)
    # substract 0.95 from modified outcomes
    new_valid = (valid.flatten() - (0.95 * (orig_outcome != new_outcome)).flatten()).reshape(valid.shape)
    assert all(new_valid[half:] == valid[half:])
    return real_gexs, new_valid, orig_outcome


def predictor_loss(model: 'InfoGAN', batch_gexs, labels, optimizers, opt):
    """
    Train discriminator on mix of real and generated data for outcome prediction
    """
    categorical_loss = torch.nn.CrossEntropyLoss()
    continuous_loss = torch.nn.MSELoss()
    param = next(model.parameters())

    batch_size = len(batch_gexs)
    # Sample noise and labels as generator input
    z = torch.as_tensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))).to(param)
    random_category = torch.randint(0, opt.n_classes, (batch_size, )).to(param.device)
    label_input = to_categorical(random_category, num_columns=opt.n_classes)
    code_input = torch.as_tensor(np.random.uniform(-1, 1, (batch_size, opt.code_dim))).to(param)


    # ---------------------
    #  Train Discriminator
    # ---------------------

    if optimizers:
        optimizer_D = optimizers['D']
        optimizer_D.zero_grad()

    fake_gexs = model.generator(z, label_input, code_input)
    fake_pred, fake_label, fake_code, new_valid = invert_discriminate(model.discriminator, fake_gexs)

    d_fake_loss = cross_entropy(new_valid, fake_pred).mean()
    real_gexs = batch_gexs.to(param)
    real_pred, _, _, new_valid  = invert_discriminate(model.discriminator, real_gexs.clone())

    d_real_loss = cross_entropy(new_valid, real_pred).mean()
    d_loss = (d_real_loss + d_fake_loss) * 0.5


    if optimizers:
        d_loss.backward(retain_graph=True)
        optimizer_D.step()
    # ------------------
    # Information Loss
    # ------------------
        optimizer_info = optimizers['info']
        optimizer_info.zero_grad()


    info_loss = opt.lambda_cat * categorical_loss(fake_label, random_category) + \
            opt.lambda_con * continuous_loss(fake_code, code_input)

    if optimizers:
       info_loss.backward()
       optimizer_info.step()
    result = defaultdict(list)
    evaluate_real(result, model.discriminator, batch_gexs)
    losses = dict(info_loss=[detach_numpy(info_loss)],
                  d_loss=[detach_numpy(d_loss)])
    result.update(losses)

    return result


def evaluate_real(result, discriminator, real_gexs):
    orig_outcome = real_gexs[:, -1]
    nonzero = orig_outcome.nonzero().flatten()
    valid_gexs = real_gexs[nonzero]
    valid_gexs[:, -1] = -0.985
    out_neg = discriminator(valid_gexs)[0]
    valid_gexs[:, -1] = 0.985
    out_pos = discriminator(valid_gexs)[0]
    p_pos = F.softmax(torch.cat([out_neg, out_pos], dim=1))[:, 1]

    orig_outcome = orig_outcome[nonzero]
    compute_metrics(result, detach_numpy(orig_outcome > 0), detach_numpy(p_pos > 0.5))
    compute_auc(result, detach_numpy(orig_outcome > 0), detach_numpy(p_pos))
    return result


def invert_discriminate(discriminator, real_gexs):
    size = len(real_gexs)
    # Adversarial ground truths
    valid = torch.ones((size, 1)).to(real_gexs) - torch.as_tensor(np.random.uniform(0.0, 0.05, (size, 1))).to(real_gexs)

    real_gexs, new_valid, _ = invert_half_outcomes(real_gexs, valid)
    pred, label, code = discriminator(real_gexs)
    return pred, label, code, new_valid


def infogan_loss(model: 'InfoGAN', batch_gexs, labels, optimizers, opt):

    # Loss functions
    categorical_loss = torch.nn.CrossEntropyLoss()
    continuous_loss = torch.nn.MSELoss()
    param = next(model.parameters())

    batch_size = batch_gexs.shape[0]

    # Adversarial ground truths
    valid = torch.ones((batch_size, 1)).to(param) - torch.as_tensor(np.random.uniform(0.0, 0.05, (batch_size, 1))).to(param)
    fake = torch.zeros((batch_size, 1)).to(param) + torch.as_tensor(np.random.uniform(0.0, 0.05, (batch_size, 1))).to(param)

    # Configure input
    real_gexs = batch_gexs.to(param)

    # Sample noise and labels as generator input
    z = torch.as_tensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))).to(param)
    random_category = torch.randint(0, opt.n_classes, (batch_size, )).to(param.device)
    label_input = to_categorical(random_category, num_columns=opt.n_classes)
    code_input = torch.as_tensor(np.random.uniform(-1, 1, (batch_size, opt.code_dim))).to(param)

    # -----------------
    #  Train Generator
    # -----------------
    if optimizers:
        optimizer_G = optimizers['G']
        optimizer_G.zero_grad()

    out_fake = model(z, label_input, code_input)

    g_loss = cross_entropy(valid, out_fake['validity']).mean()

    if optimizers:
        g_loss.backward(retain_graph=True)
        optimizer_G.step()

    # ---------------------
    #  Train Discriminator
    # ---------------------

    if optimizers:
        optimizer_D = optimizers['D']
        optimizer_D.zero_grad()

    # Loss for real images
    # invert some outcomes
    real_pred, _, _, new_valid = invert_discriminate(model.discriminator, real_gexs.clone())
    d_real_loss = cross_entropy(new_valid, real_pred)
#    weights = torch.ones_like(d_real_loss)
#    weights[half:] = 4
    d_real_loss = (d_real_loss).mean()

    # Loss for fake images
    d_fake_loss = cross_entropy(fake, out_fake['validity']).mean()
    acc = accuracy_score(detach_numpy(fake > 0.5), detach_numpy(out_fake['validity'] > 0.5))

    # Total discriminator loss
    d_loss = (d_real_loss + d_fake_loss) / 2

    if optimizers:
        d_loss.backward(retain_graph=True)
        optimizer_D.step()
    # ------------------
    # Information Loss
    # ------------------
        optimizer_info = optimizers['info']
        optimizer_info.zero_grad()

    info_loss = opt.lambda_con * continuous_loss(out_fake['S'], code_input)
    if out_fake['categorical'] is not None:
        info_loss = info_loss + opt.lambda_cat * categorical_loss(out_fake['categorical'], random_category)

    if optimizers:
        info_loss.backward()
        optimizer_info.step()
    result = defaultdict(list)
    evaluate_real(result, model.discriminator, batch_gexs)
    result['desc_acc_fake'].append(acc)
    losses = dict(info_loss=[detach_numpy(info_loss)],
                d_loss=[detach_numpy(d_loss)],
                g_loss=[detach_numpy(g_loss)])
    result.update(losses)
    return result


def detach_numpy(tensor):
    return tensor.cpu().detach().numpy()
