import torch


class GeneDataset(torch.utils.data.Dataset):
    def __init__(self, features, labels, transform=None):
        self.features = features
        self.labels = labels
        assert len(features) == len(labels)
        self.transform = transform

    def __getitem__(self, idx):
        return self.transform(data=self.features[idx],
            target=self.labels[idx])

    def __len__(self):
        return len(self.features)

