import abc
import numpy
import torch


class ToTensor:
    def __call__(self, arg):
        return torch.as_tensor(arg)


class TransformCompose:

    @abc.abstractmethod
    def __call__(self, data=None, target=None):
        pass


class ToType:
    def __init__(self, typename):
        self.typename = typename

    def __call__(self, x):
        return getattr(x, self.typename)()


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x


class AdditiveUniform:
    """
    Adding noise sampled from uniform distribution to a given column
    zero entries are not modified
    """
    def __init__(self, low, high, colname):
        self.low = low
        self.high = high
        self.colname = colname

    def __call__(self, x):
        x = x.copy()
        if x[self.colname] > 0:
            x[self.colname] += numpy.random.uniform(low=self.low, high=self.high)
        if x[self.colname] < 0:
            x[self.colname] = 0
        return x


class AdditiveUniformTriary(AdditiveUniform):
    def __call__(self, x):
        x = x.copy()
        if x[self.colname] > 0:
            x[self.colname] -= numpy.random.uniform(low=self.low, high=self.high)
        if x[self.colname] < 0:
            x[self.colname] += numpy.random.uniform(low=self.low, high=self.high)
        return x


class DataLabelCompose(TransformCompose):
    def __init__(self, data_transform, label_transform):
        self.data_transform = data_transform
        self.label_transform = label_transform

    def __call__(self, data=None, target=None):
        return self.data_transform(data), self.label_transform(target)
