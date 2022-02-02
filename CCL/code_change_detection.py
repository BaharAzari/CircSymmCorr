

import os
import time
import sys
from glob import glob
import numpy as np
from tqdm import tqdm
import pdb
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import torch.nn.functional as F
from torchvision import datasets, transforms
from _dct import LinearDCT
import cv2


class CCL_(nn.Module):
    def __init__(self, c_in, c_out, kernel, stride):
        super(CCL_, self).__init__()
        
        if np.isscalar(kernel):
            kernel = np.array([kernel, kernel])
        if np.isscalar(stride):
            stride = np.array([stride, stride])
            
        K = np.sqrt(1/(c_in*kernel.prod()))
        
        init = (2*torch.rand(c_out,c_in, kernel[0],kernel[1])-1)*K
        self.weight = nn.Parameter(init)
        
        init=(2*torch.rand(c_out)-1)*K
        self.bias = nn.Parameter(init)
        self.stride = stride

    def forward(self, x): # x: N x C_in x H x W
        
        dims = x.shape[-2:]
        dct = LinearDCT(dims[0], 'dct')
        idct = LinearDCT(dims[0], 'idct')

        x = dct(x.permute(0,1,3,2)).permute(0,1,3,2)
        x = torch.fft.rfft(x).unsqueeze(1)
        w = dct(F.pad(self.weight.permute(0,1,3,2),
                      [0,dims[0]-self.weight.shape[-2]])).permute(0,1,3,2)
        w = torch.fft.rfft(w, dims[1])
        
        h = (x * w).sum(2)
        h = torch.fft.irfft(h, dims[1]) 
        h = idct(h.permute(0,1,3,2)).permute(0,1,3,2) + self.bias.reshape(-1, 1, 1)
        
        return h[:,:,::self.stride[0],::self.stride[1]]
    

class CCL(nn.Module):
    def __init__(self, c_in, c_out, kernel, stride):
        super(CCL, self).__init__()
        
        if np.isscalar(kernel):
            kernel = np.array([kernel, kernel])
        if np.isscalar(stride):
            stride = np.array([stride, stride])
            
        K = np.sqrt(1/(c_in*kernel[0]*kernel[1]))
        
        init = (2*torch.rand(c_out,c_in, kernel[0],kernel[1])-1)*K
        self.weight = nn.Parameter(init)
        
        init=(2*torch.rand(c_out)-1)*K
        self.bias = nn.Parameter(init)
        self.stride = stride

    def forward(self, x): # x: N x C_in x H x W
        
        dims = x.shape[-2:]
        
        x = torch.fft.rfft2(x).unsqueeze(1)
        w = torch.fft.rfft2(self.weight, dims)
        
        h = (x * w).sum(2)
        h = torch.fft.irfft2(h, dims) + self.bias.reshape(-1, 1, 1)
        
        return h[:,:,::self.stride[0],::self.stride[1]]



class Net_CNN(nn.Module):
    def __init__(self):
        super(Net_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 50, (3, 7), 1, (1,3))
        self.conv2 = nn.Conv2d(50, 50, (3,7), 1, (1,3))
        self.pool = nn.MaxPool2d((2,4),(2,4))
        self.conv3 = nn.ConvTranspose2d(50, 50, (3,5), (2,4),(1,1),(1,1))
        self.conv4 = nn.ConvTranspose2d(50, 50, (3,7), (2,4),(1,2),(1,1))
        #self.conv3 = nn.ConvTranspose2d(20, 20, (3,5), (1,1),(1,2),(0,0))
        #self.conv4 = nn.ConvTranspose2d(20, 20, (3,7), (1,1),(1,3),(0,0))
        self.conv5 = nn.ConvTranspose2d(50, 1, (3,7), 1,(1,3))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.reshape(-1, 1, x.shape[-2], x.shape[-1])
        x = self.pool(self.conv1(x))
        x = F.relu(x)
        x = self.pool(self.conv2(x))
        x = F.relu(x)
        x = x.reshape(-1, 2, x.shape[-3], x.shape[-2], x.shape[-1])
        x = x[:, 0]+x[:, 1].mean([-1,-2]).unsqueeze(-1).unsqueeze(-1)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.sigmoid(self.conv5(x)).squeeze(1)
        return x
    

class Net_CCL(nn.Module):
    def __init__(self):
        super(Net_CCL, self).__init__()
        self.conv1 = CCL(1, 50, (3, 7), 1)
        self.conv2 = CCL(50, 50, (3,7), 1)
        self.pool = nn.MaxPool2d((2,4),(2,4))
        self.conv3 = CCL(50, 50, (3,5), 1)
        self.conv4 = CCL(50, 50, (3,7), 1)
        self.conv5 = CCL(50, 1, (3,7), 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = x.reshape(-1, 1, x.shape[-2], x.shape[-1])
        x = self.pool(self.conv1(x))
        x = F.relu(x)
        x = self.pool(self.conv2(x))
        x = F.relu(x)
        x = x.reshape(-1, 2, x.shape[-3], x.shape[-2], x.shape[-1])
        x = x[:, 0]+x[:, 1].mean([-1,-2]).unsqueeze(-1).unsqueeze(-1)
        y = torch.zeros(x.shape[0],x.shape[1],x.shape[2]*2,x.shape[3]*4)
        y[:,:,::2,::4] = x
        x = self.conv3(y)
        x = F.relu(x)
        y = torch.zeros(x.shape[0],x.shape[1],x.shape[2]*2,x.shape[3]*4)
        y[:,:,::2,::4] = x
        x = self.conv4(y)
        x = F.relu(x)
        x = self.sigmoid(self.conv5(x)).squeeze(1)
        return x


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    correct = 0
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data[:,:2].unsqueeze(2)*2-1)
        loss = BCE(output, data[:,2])
        loss.backward()
        optimizer.step()
        correct += ((output > 0.5).type(torch.bool) == (data[:,2]>0.5).type(torch.bool)).sum().item()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break
    print('train accuracy: %.2f' %(100. * correct / (len(train_loader.dataset)*data.shape[-1]*data.shape[-2])))


def test(model, device, test_loader, roll=0, flag=False):
    model.eval()
    test_loss = 0
    correct = 0
    conf = np.zeros(4) #[tp,tn,fp,fn]
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            data = data.to(device)
            #data = data.flip(-2)
            #data = data.roll(roll, -1)
            data[:,[0,2]] = data[:,[0,2]].roll(roll,-1)
            #data = data.roll(roll,-2)
            #data[:,:,:roll]=0
            output = model(data[:,:2].unsqueeze(2)*2-1)
            test_loss += BCE(output, data[:,2]).item()  # sum up batch loss
            correct += ((output[:,:,:10] > 0.5).type(torch.bool) == (data[:,2,:,:10]>0.5).type(torch.bool)).sum().item()
            #pdb.set_trace()
            #tp = (((output[:,:,:10]>0.5)==data[:,2,:,:10])*(data[:,2,:,:10]==1)).sum().item()
            #tn = (((output[:,:,:10]>0.5)==data[:,2,:,:10])*(data[:,2,:,:10]==0)).sum().item()
            if flag:
                output = torch.cat((data[0,2],output[0]>0.5)).numpy()*255
                #pdb.set_trace()
                output1 = torch.cat((data[0,0],data[0,1])).numpy()*255
                cv2.imwrite(save_path + '%.3d.jpg'%i, output)
                cv2.imwrite(save_path + '%.3d_r.jpg'%i, output1)
    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / (len(test_loader.dataset)*10*data.shape[-2])))#data.shape[-1]*data.shape[-2])))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
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

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)


    data_path = './TSUNAMI/'
    images = np.zeros((100, 3, 28, 128)) 
    for i in range(100):
        r_0 = np.random.randint(0,128//2)
        r_1 = np.random.randint(0,128//2)
#        if i==34:
#            cv2.imwrite(save_path+'orig34.jpg',np.concatenate((np.roll(cv2.imread(data_path+'t0/%.8d.jpg'%i),19*8,-2),
#            np.roll(cv2.imread(data_path+'t1/%.8d.jpg'%i),24*8,-2))))
#            pdb.set_trace()
        images[i,0] = np.roll(cv2.imread(data_path+'t0/%.8d.jpg'%i, 0)[::8,::8],r_0,-1) 
        images[i,1] = np.roll(cv2.imread(data_path+'t1/%.8d.jpg'%i, 0)[::8,::8],r_1,-1)
        images[i,2] = np.roll(cv2.imread(data_path+'mask/%.8d.png'%i, 0)[::8,::8],r_0,-1)

    images = torch.FloatTensor(images)/255

    idxs = torch.randperm(len(images))
    train_idxs = idxs[:int(len(images)*0.8)]
    test_idxs = idxs[int(len(images)*0.8):]
    dataset1 = [images[i] for i in train_idxs]
    dataset2 = [images[i] for i in test_idxs]
    del images
    
    train_loader = torch.utils.data.DataLoader(dataset1, batch_size=args.batch_size,
                                         shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(dataset2, batch_size=args.test_batch_size,
                                         shuffle=False, num_workers=0)
    if args.which == 'CNN':
        model = Net_CNN().to(device)
    else:
        model = Net_CCL().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    if args.train:
        #model.load_state_dict(torch.load("change_cnn_%s.pt" %which))
        for epoch in range(1, args.epochs + 1):
            train(args, model, device, train_loader, optimizer, epoch)
            test(model, device, test_loader,roll=63)
    
            if args.save_model:
                torch.save(model.state_dict(), "change_%s.pt"%args.which)
    else:
        model.load_state_dict(torch.load("change_%s.pt"%args.which))
        for roll in [15]:#range(128//2):
            test(model, device, train_loader, roll, flag=False)


if __name__ == '__main__':
    #save_path = './results%s/'%args.which
    BCE = nn.BCELoss()
    main()

