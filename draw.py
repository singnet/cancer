from mnist import Net
import numpy
import torch
from torchvision import datasets, transforms
import cv2


def main():
    model = Net(activation=torch.nn.LeakyReLU())
    use_cuda = True
    device = 'cuda'

    test_batch_size = 20
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, transform=transforms.Compose([
                       transforms.ToTensor(),
                   ])),
    batch_size=test_batch_size, shuffle=True, **kwargs)
    PATH = 'mnist_cnn100.pt'
    model.load_state_dict(torch.load(PATH, map_location=device))
    model = model.eval().to(device)
    with torch.no_grad():
       for data, target in test_loader:
           data, target = data.to(device), target.to(device)
           shape = data.shape
           x = data.reshape(shape[0], numpy.prod(shape[1:]))
           output = model(x).reshape(shape)
           cv2.imshow('source', data[3][0].cpu().numpy())
           cv2.imshow('target', output[3][0].cpu().numpy())
           q = cv2.waitKey()
           if q == 113: # q
               return
main()
