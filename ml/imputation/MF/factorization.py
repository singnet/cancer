import torch
from torch import nn
from torch import optim
import numpy

torch.autograd.set_detect_anomaly(True)


def numpy_nonzero(tensor):
    if isinstance(tensor, numpy.ndarray):
        select = tensor.nonzero()
    return torch.unbind(tensor.nonzero(), 1)


def loss(A, U, V, missing, median, w):
    tmp = U @ V.T
    select1 = numpy_nonzero(1 - missing)
    loss_present = ((A[select1] - tmp[select1]) ** 2)
    select2 = numpy_nonzero(missing)
    loss_missing = (median[select2] - tmp[select2]) ** 2
    if any(torch.isnan(loss_missing).flatten()):
        import pdb;pdb.set_trace()
    return loss_present.mean() + loss_missing.mean() * w


def main():
    A = numpy.asarray([[1, 0, 1, 1, 0],
                       [0, 1, 0, 0, 1],
                       [1, 1, 1, 0, 0],
                       [0, 0, 0, 1, 1]]).astype(numpy.float16)
    idx = numpy.where(A == 0)
    A[idx] = numpy.nan
    train_embedding(A, 2)


def train_embedding(A, embed_size, device='cpu'):
    print('using device {0}'.format(device))
    A = torch.from_numpy(A).to(device)
    weight_missing = 0.01
    num_patients = A.shape[0]
    num_variables = A.shape[1]

    U = nn.Embedding(num_patients, embed_size).to(device)
    V = nn.Embedding(num_variables, embed_size).to(device)

    missing = numpy.isnan(A.cpu())  * 1.0
    median = torch.ones_like(A) * torch.from_numpy(numpy.nanmedian(A.cpu(), axis=0)).to(A)
    median = torch.zeros_like(A)
    # todo: median or zeros might be an
    # obvious solution and wrong solution
    optimizer = optim.Adam([{'params': U.parameters()},
                            {'params': V.parameters()}], lr=0.0001)
    i = 0
    thresh = 0.000005
    patience = 3000
    prev_loss = 100
    while True:
        optimizer.zero_grad()
        l = loss(A, U.weight, V.weight, missing, median, weight_missing)
        l.backward()
        optimizer.step()
        current = l.detach().item()
        if (prev_loss - current) < thresh:
            patience -= 1
        else:
            patience = 3000
        if current < prev_loss:
            prev_loss = current
            patience += 1
        i += 1
        if patience <= 0:
            print('No improvement {0}\'th iteration'.format(i))
            break
        if i % 200 == 0:
           print(l)
        if i == 30000:
            print('stopping at {0}\'th iteration'.format(i))
            break
    return U.weight, V.weight


def unique(x):
    x = x[~numpy.isnan(x)]
    return list(set(x))


def match_to_nearest(X, X_new, missing):
    per_col = dict()
    for row, col in zip(*missing.nonzero()):
        if col not in per_col:
            per_col[col] = numpy.asarray(unique(X[:, col].flatten()))
        a0 = X_new[row, col]
        a = per_col[col]
        idx = numpy.abs(a - a0).argmin()
        X_new[row, col] = a.flat[idx]
    return X_new


class MFImputer:
    def __init__(self, size):
        self.size = size
        self.device = 'cpu'

    def fit_transform(self, X):
        X = numpy.asarray(X)
        U, V = train_embedding(X, self.size, device=self.device)
        X_new = X.copy()
        missing = numpy.isnan(X)
        P = U @ V.T
        X_new[missing.nonzero()] = P[missing.nonzero()].detach().cpu()
        result = match_to_nearest(X, X_new.copy(), missing)
        return result

    def to(self, device):
        self.device = device
        return self


if __name__ == '__main__':
    main()
