import torch
from collections import Counter
from sklearn.metrics import accuracy_score
import numpy
from infogan import to_categorical
from metrics import compute_metrics, compute_auc
from collections import defaultdict
from torch.nn import functional as F
from common import mutual_information
np = numpy


def cross_entropy(y, y_pred, eps=0.0000000001):
    result = -(y * torch.log(y_pred + eps) + (1 - y) * torch.log(1 - y_pred + eps))
    return result


def tanh_loss(y, y_pred, eps=0.0000000001):
    result = torch.exp(torch.abs(y - y_pred)) - 1
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


def sample_random_variables(batch_size, opt, param):
    z = torch.as_tensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))).to(param)
    random_category = torch.randint(0, opt.n_classes, (batch_size, )).to(param.device)
    label_input = to_categorical(random_category, num_columns=opt.n_classes)
    if opt.distribution == 'uniform':
        code_tmp = np.random.uniform(-1, 1, (batch_size, opt.code_dim))
    elif opt.distribution == 'normal':
        code_tmp = torch.normal(torch.zeros(batch_size, opt.code_dim),
                torch.ones(batch_size, opt.code_dim))
    code_input = torch.as_tensor(code_tmp).to(param)
    return z, label_input, code_input, random_category


def predictor_loss(model: 'InfoGAN', batch_gexs, labels, optimizers, opt):
    """
    Train discriminator on mix of real and generated data for outcome prediction
    """
    categorical_loss = torch.nn.CrossEntropyLoss()
    continuous_loss = torch.nn.MSELoss()
    param = next(model.parameters())

    batch_size = len(batch_gexs)
    # Sample noise and labels as generator input
    z, label_input, code_input, _ = sample_random_variables(batch_size, opt, param)


    # ---------------------
    #  Train Discriminator
    # ---------------------

    if optimizers:
        optimizer_D = optimizers['D']
        optimizer_D.zero_grad()

    assert opt.invert_outcome
    fake_gexs = model.generator(z, label_input, code_input)
    fake_pred, fake_label, fake_code, new_valid = invert_discriminate(model.discriminator, fake_gexs, opt.invert_outcome)

    d_fake_loss = cross_entropy(new_valid, fake_pred).mean()
    real_gexs = batch_gexs.to(param)
    real_pred, _, _, new_valid  = invert_discriminate(model.discriminator, real_gexs.clone(), opt.invert_outcome)

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
            opt.lambda_con * continuous_loss(fake_code, code_input).mean()

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


def invert_discriminate(discriminator, real_gexs, invert):
    size = len(real_gexs)
    # Adversarial ground truths
    valid = torch.ones((size, 1)).to(real_gexs) - torch.as_tensor(np.random.uniform(0.0, 0.05, (size, 1))).to(real_gexs)
    if invert:
        real_gexs, new_valid, _ = invert_half_outcomes(real_gexs, valid)
    else:
        new_valid = valid
    pred, label, code = discriminator(real_gexs)
    return pred, label, code, new_valid


def infogan_loss(model: 'InfoGAN', batch_gexs, labels, optimizers, opt):

    # Loss functions
    categorical_loss = torch.nn.CrossEntropyLoss()
    if opt.continious_loss == 'mse':
        continuous_loss = torch.nn.MSELoss()
    elif opt.continious_loss == 'tanh':
        continuous_loss = tanh_loss
    else:
        assert False
    param = next(model.parameters())

    batch_size = batch_gexs.shape[0]

    # Adversarial ground truths
    valid = torch.ones((batch_size, 1)).to(param) - torch.as_tensor(np.random.uniform(0.0, 0.05, (batch_size, 1))).to(param)
    fake = torch.zeros((batch_size, 1)).to(param) + torch.as_tensor(np.random.uniform(0.0, 0.05, (batch_size, 1))).to(param)

    # Configure input
    real_gexs = batch_gexs.to(param)

    # Sample noise and labels as generator input
    z, label_input, code_input, random_category = sample_random_variables(batch_size, opt, param)
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
    real_category = labels[:, 0].long()
    real_pred, cat_pred, real_code, new_valid = invert_discriminate(model.discriminator, real_gexs.clone(), opt.invert_outcome)
    real_cat_loss = opt.lambda_cat * categorical_loss(cat_pred, real_category)
    d_real_loss = cross_entropy(new_valid, real_pred) + real_cat_loss


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

    result = defaultdict(list)
    real_labels = labels[:, 0].long()
    info_loss = (opt.lambda_con * continuous_loss(out_fake['S'], code_input)).mean()
    result['info_cont_loss'] = [detach_numpy(info_loss)]
    result['real_cat_loss'] = [detach_numpy(real_cat_loss)]
    if out_fake['categorical'] is not None:
        cat_loss = opt.lambda_cat * categorical_loss(out_fake['categorical'], random_category)
        result['info_cat_loss'] = [detach_numpy(cat_loss)]
        info_loss = info_loss + cat_loss

    if optimizers:
        info_loss.backward()
        optimizer_info.step()

    if opt.invert_outcome:
        evaluate_real(result, model.discriminator, batch_gexs)

    result['desc_acc_fake'].append(acc)
    losses = dict(info_loss=[detach_numpy(info_loss)],
                d_loss=[detach_numpy(d_loss)],
                g_loss=[detach_numpy(g_loss)])
    result.update(losses)
    return result


def evaluate_predictor(model, code, labels):
    nonzero = labels.nonzero().flatten()
    labels = labels[nonzero]
    batch_gexs = code[nonzero]
    p_pos = model(code[nonzero])
    if not len(labels):
        return None, None
    loss = cross_entropy(labels, p_pos).mean()
    result = defaultdict(list)
    auc = compute_auc(result, detach_numpy(labels > 0), detach_numpy(p_pos))
    return loss, result['auc']


def detach_numpy(tensor):
    return tensor.cpu().detach().numpy()


def infogan_loss_outcome(model: 'InfoGAN', batch_gexs, labels, optimizers, opt):
    train = optimizers is not None
    train_only_D = False
    param = next(model.parameters())

    batch_size = len(batch_gexs)
    # Sample noise and labels as generator input
    z, label_input, code_input, random_category = sample_random_variables(batch_size, opt, param)
    # label_input and code input are to be reconstructed

    # Adversarial ground truths
    valid = torch.ones((batch_size, 1)).to(param) - torch.as_tensor(np.random.uniform(0.0, 0.05, (batch_size, 1))).to(param)
    fake = torch.zeros((batch_size, 1)).to(param) + torch.as_tensor(np.random.uniform(0.0, 0.05, (batch_size, 1))).to(param)


    # Configure input
    real_gexs = batch_gexs.to(param)
    outcome_noise = torch.as_tensor(np.random.uniform(0.0, 0.05, (batch_size))).to(param)
    outcome_real = (labels - outcome_noise) * (labels > 0.5) + (labels + outcome_noise) * (labels <= 0.5)
    real_data = {'genes': real_gexs, 'outcomes': outcome_real.to(param).unsqueeze(1)}
    inverted_data = {'genes': real_gexs, 'outcomes': 1 - outcome_real.to(param).unsqueeze(1)}


    concat_real = torch.cat([real_data['genes'], real_data['outcomes']], dim=1)
    concat_fake = torch.cat([inverted_data['genes'], inverted_data['outcomes']], dim=1)
    xdata = torch.cat([concat_real, concat_fake])
    xlabels = numpy.ones(len(xdata))
    xlabels[len(xdata) // 2: ] = 0
    categorical_loss = torch.nn.CrossEntropyLoss()

    if not train_only_D:
        # output of generator and descriminator on fake data
        out_fake = model(z, label_input, code_input)
        info_loss = 0

        # infogan loss including categorical loss
        if out_fake['categorical'] is not None:
            cat_loss = opt.lambda_cat * categorical_loss(out_fake['categorical'], random_category)
            info_loss = cat_loss + info_loss

    # real output
    real_pred, code_real, _ = model.discriminator(real_data)
    code_real = code_real.argmax(dim=1)
    # ------------------
    # Mutual Information
    # ------------------
    counter = Counter()
    counter.update(code_real.cpu().numpy())
    p_c = {k: v / len(code_real) for (k, v) in counter.items()}
    counter = Counter()
    counter.update(((labels > 0.5) * 1).cpu().numpy())
    p_outcome = {k: v / len(code_real) for (k, v) in counter.items()}
    counter = Counter()
    counter.update(zip(code_real.cpu().numpy(), ((labels > 0.5) * 1).cpu().numpy()))
    p_c_outcome = defaultdict(int)
    p_c_outcome.update({k: v / len(code_real) for (k, v) in counter.items()})
    mi = mutual_information(p_c_outcome, p_c, p_outcome)
    real_inverted_pred, _, _ = model.discriminator(inverted_data)

    stacked_pred = torch.cat([real_pred, real_inverted_pred])

    # optimizers
    if train:
        optimizer_info = optimizers['info']
        optimizer_D = optimizers['D']
        optimizer_G = optimizers['G']
        optimizer_info.zero_grad()
        optimizer_G.zero_grad()
        optimizer_D.zero_grad()

    # -----------------
    # Discriminator Loss
    # -----------------
    d_real_loss = cross_entropy(valid, real_pred)
    d_real_inverted_loss = cross_entropy(fake, real_inverted_pred)
    if not train_only_D:
        d_fake_loss = cross_entropy(fake, out_fake['validity']).mean()
        d_loss = (d_fake_loss / 3 + d_real_loss / 3 + d_real_inverted_loss / 3).mean()
    else:
        d_loss = (d_real_loss / 2 + d_real_inverted_loss / 2).mean()

    if train:
        d_loss.backward(retain_graph=True)


    result = dict()
    # -----------------
    # Generator Loss
    # -----------------
    if train and not train_only_D:
        g_loss = cross_entropy(valid, out_fake['validity']).mean()
        g_loss.backward(retain_graph=True)

    # ------------------
    # Information Loss
    # ------------------
        info_loss.backward()

    if train:
        optimizer_D.step()

    if train and not train_only_D:
        optimizer_G.step()
        optimizer_info.step()

        losses = dict(info_loss=[detach_numpy(info_loss)],
                      g_loss=[detach_numpy(g_loss)])
        result['fake_pred'] = [detach_numpy(out_fake['validity'].mean())]
        result['info_accuracy'] = [accuracy_score(random_category.cpu(), out_fake['categorical'].argmax(dim=1).detach().cpu())]
        result.update(losses)

    result['MI(c, outcome)'] = [mi]
    result['d_loss'] = [detach_numpy(d_loss)]
    result['real_pred'] = [detach_numpy(real_pred.mean()) ]
    result['inverted_pred'] = [detach_numpy(real_inverted_pred.mean())]
    result['accuracy real inverted'] = [accuracy_score(xlabels, stacked_pred.cpu().detach().numpy() >= 0.5)]
    return result
