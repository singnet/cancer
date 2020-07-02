import abc
import torch


class ToTensor:
    def __call__(self, arg):
        return torch.as_tensor(arg)


class TransformCompose:

    @abc.abstractmethod
    def __call__(self, data=None, target=None):
        pass


class DataLabelCompose(TransformCompose):
    def __init__(self, data_transform, label_transform):
        self.data_transform = data_transform
        self.label_transform = label_transform

    def __call__(self, data=None, target=None):
        return self.data_transform(data), self.label_transform(target)



