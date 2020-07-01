from mnist import Net
import matplotlib.pyplot as plt
import matplotlib
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
    PATH = 'mnist_cnn100.pt.bak'
    model.load_state_dict(torch.load(PATH, map_location=device))
    model = model.eval().to(device)
    with torch.no_grad():
       for data, target in test_loader:
           data, target = data.to(device), target.to(device)
           shape = data.shape
           x = data.reshape(shape[0], numpy.prod(shape[1:]))
           output = model(x)['output'].reshape(shape)
           cv2.imshow('source', data[3][0].cpu().numpy())
           cv2.imshow('target', output[3][0].cpu().numpy())
           output1 = model.decode(torch.normal(
               torch.zeros((100, 2)), torch.ones((100, 2))).to(data)).reshape((100, 28, 28))
           fig, axes = plt.subplots(10, 10)
           plt.figure(1)
           for i in range(100):
               img = output1[i].cpu().numpy()
               k = i // 10
               j = i % 10
               axes[k, j].imshow((img * 255).astype(numpy.int32), cmap='gray', vmin=0, vmax=255)
           plt.subplots_adjust(wspace=0, hspace=0)
           plt.show()
           q = cv2.waitKey()
           if q == 113: # q
               return
main()
