import torch


class GeneDataset(torch.utils.data.Dataset):
    def __init__(self, features, labels, transform=None, binary=0):
        self.features = features
        self.labels = labels
        assert len(features) == len(labels)
        self.transform = transform
        self.binary = binary

    def __getitem__(self, idx):
        return self.transform(data=self.features.iloc[idx],
            target=self.labels.iloc[idx])

    def __len__(self):
        return len(self.features)
