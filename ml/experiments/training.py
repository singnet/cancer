import torch


def train(model, compute_loss, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        for opt in optimizer.values():
            opt.zero_grad()
        loss = compute_loss(model, data, target, optimizer)
        if batch_idx % 100 == 0:
            for k,v in loss.items():
                print(('Train Epoch: {} [{}/{} ({:.0f}%)]\t' + str(k) + ': {:.6f}').format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), v.item()))


def test(model, compute_loss, device, test_loader):
    model.eval()
    test_loss = defaultdict(float)
    correct = 0
    i = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            for key, value in  compute_loss(model, data, target).items():
                test_loss[key] += value.item()  # sum up batch loss
            i += 1
            # pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            #correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss = {k: v / i for (k, v) in test_loss.items()}
    for k, v in test_loss.items():
        print('\nTest set: Average {}:'.format(k) + '{:.4f}', v)
#    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#        test_loss, correct, len(test_loader.dataset),
#        100. * correct / len(test_loader.dataset)))


