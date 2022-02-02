import os
import time
import sys
from glob import glob
import numpy as np
from tqdm import tqdm
import pdb
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import torch.nn.functional as F
from torchvision import datasets, transforms
from _dct import LinearDCT


class CCL_(nn.Module):
    def __init__(self, c_in, c_out, kernel, stride):
        super(CCL_, self).__init__()

        if np.isscalar(kernel):
            kernel = np.array([kernel, kernel])
        if np.isscalar(stride):
            stride = np.array([stride, stride])

        K = np.sqrt(1 / (c_in * kernel.prod()))

        init = (2 * torch.rand(c_out, c_in, kernel[0], kernel[1]) - 1) * K
        self.weight = nn.Parameter(init)

        init = (2 * torch.rand(c_out) - 1) * K
        self.bias = nn.Parameter(init)
        self.stride = stride

    def forward(self, x):  # x: N x C_in x H x W

        dims = x.shape[-2:]
        dct = LinearDCT(dims[0], 'dct')
        idct = LinearDCT(dims[0], 'idct')

        x = dct(x.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        x = torch.fft.rfft(x).unsqueeze(1)
        w = dct(F.pad(self.weight.permute(0, 1, 3, 2),
                      [0, dims[0] - self.weight.shape[-2]])).permute(0, 1, 3, 2)
        w = torch.fft.rfft(w, dims[1])

        h = (x * w).sum(2)
        h = torch.fft.irfft(h, dims[1])
        h = idct(h.permute(0, 1, 3, 2)).permute(0, 1, 3, 2) + self.bias.reshape(-1, 1, 1)

        return h[:, :, ::self.stride[0], ::self.stride[1]]


class CCL__(nn.Module):
    def __init__(self, c_in, c_out, kernel, stride):
        super(CCL__, self).__init__()

        if np.isscalar(kernel):
            kernel = np.array([kernel, kernel])
        if np.isscalar(stride):
            stride = np.array([stride, stride])

        K = np.sqrt(1 / (c_in * kernel.prod()))

        init = (2 * torch.rand(c_out, c_in, kernel[0], kernel[1]) - 1) * K
        self.weight = nn.Parameter(init)

        init = (2 * torch.rand(c_out) - 1) * K
        self.bias = nn.Parameter(init)
        self.stride = stride

    def forward(self, x):  # x: N x C_in x H x W

        dims = x.shape[-2:]
        dct = LinearDCT(dims[0], 'dct')
        idct = LinearDCT(dims[0], 'idct')

        x = dct(x.permute(0, 1, 3, 2)).permute(0, 1, 3, 2).unsqueeze(1)
        w = dct(F.pad(self.weight.permute(0, 1, 3, 2),
                      [0, dims[0] - self.weight.shape[-2],
                       0, dims[1] - self.weight.shape[-1]]
                      )).permute(0, 1, 3, 2)
        h = (x * w)
        h = idct(h.permute(0, 1, 2, 4, 3)).permute(0, 1, 2, 4, 3)
        h = torch.fft.rfft(h)
        w = torch.fft.rfft(F.pad(self.weight,
                                 [0, 0, 0, dims[0] - self.weight.shape[-2]
                                  ]), dims[1])
        h = (h * w).sum(2)
        h = torch.fft.irfft(h, dims[1]) + self.bias.reshape(-1, 1, 1)

        return h[:, :, ::self.stride[0], ::self.stride[1]]


class CCL(nn.Module):
    def __init__(self, c_in, c_out, kernel, stride):
        super(CCL, self).__init__()

        if np.isscalar(kernel):
            kernel = np.array([kernel, kernel])
        if np.isscalar(stride):
            stride = np.array([stride, stride])

        K = np.sqrt(1 / (c_in * kernel.prod()))

        init = (2 * torch.rand(c_out, c_in, kernel[0], kernel[1]) - 1) * K
        self.weight = nn.Parameter(init)

        init = (2 * torch.rand(c_out) - 1) * K
        self.bias = nn.Parameter(init)
        self.stride = stride

    def forward(self, x):  # x: N x C_in x H x W

        dims = x.shape[-2:]

        x = torch.fft.rfft2(x).unsqueeze(1)
        w = torch.fft.rfft2(self.weight, dims)

        h = (x * w).sum(2)
        h = torch.fft.irfft2(h, dims) + self.bias.reshape(-1, 1, 1)

        return h[:, :, ::self.stride[0], ::self.stride[1]]


class Net_CNN(nn.Module):
    def __init__(self):
        super(Net_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, 1)
        self.conv2 = nn.Conv2d(8, 8, 3, 2)
        self.conv3 = nn.Conv2d(8, 8, 3, 1)
        self.conv4 = nn.Conv2d(8, 8, 3, 2)
        self.conv5 = nn.Conv2d(8, 10, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv5(x).mean([-1, -2])
        output = F.log_softmax(x, dim=-1)
        return output


class Net_CCL(nn.Module):
    def __init__(self):
        super(Net_CCL, self).__init__()
        self.conv1 = CCL(1, 8, 3, 1)
        self.conv2 = CCL(8, 8, 3, 2)
        self.conv3 = CCL(8, 8, 3, 1)
        self.conv4 = CCL(8, 8, 3, 2)
        self.conv5 = CCL(8, 10, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv5(x).mean([-1, -2])
        output = F.log_softmax(x, dim=-1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        #data = data.roll(np.random.randint(0,15,1)[0],-1)
        #pdb.set_trace()
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader, roll=0):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            # data = data.flip(-2)
            #data = data.roll(np.random.randint(0, 15, 1)[0], -1)
            data = data.roll(roll, -1)
            # data = data.roll(roll,-2)
            # data[:,:,:roll]=0
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_false', default=True,
                        help='For Saving the current Model')
    parser.add_argument('--train', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--which', type=str, default='CCL',
                        help='Choose CCL or CNN layer')
    
    
    
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    #args.train=False
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                              transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                              transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
    
    if args.which == 'CNN':
        model = Net_CNN().to(device)
    else:
        model = Net_CCL().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    if args.train:
        for epoch in range(1, args.epochs + 1):
            train(args, model, device, train_loader, optimizer, epoch)
            test(model, device, test_loader, roll=0)

            if args.save_model:
                torch.save(model.state_dict(), "mnist_%s.pt"%args.which)
    else:
        model.load_state_dict(torch.load("mnist_%s.pt"%args.which))
        for roll in range(28):
            test(model, device, test_loader, roll)


if __name__ == '__main__':
    main()

