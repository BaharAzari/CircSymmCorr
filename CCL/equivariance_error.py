import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from _dct import LinearDCT

# adjust matplotlib parameters
plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 20
plt.rcParams['font.family'] = "serif"
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)
#plt.rc('xtick', labelsize='x-small')
#plt.rc('ytick', labelsize='x-small')

if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"  
device = torch.device(dev)


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


class CCL(nn.Module):
    def __init__(self, c_in, c_out, kernel, stride):
        super(CCL, self).__init__()
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
        x = torch.fft.rfft2(x).unsqueeze(1)
        w = torch.fft.rfft2(self.weight, dims)
        h = (x * w).sum(2)
        h = torch.fft.irfft2(h, dims) + self.bias.reshape(-1, 1, 1)
        return h[:,:,::self.stride[0],::self.stride[1]]





class Net_ours(nn.Module):
    def __init__(self):
        super(Net_ours, self).__init__()
        self.conv1 = CCL(1, 8, 3, 1)
        self.conv2 = CCL(8, 8, 3, 2)
        self.conv3 = CCL(8, 8, 3, 3)
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
        x = self.conv5(x).mean([-1,-2])
        output = F.log_softmax(x, dim=-1)
        return output


class PHI_1(nn.Module):
    def __init__(self, relu=True):
        super(PHI_1, self).__init__()
        self.conv1 = CCL(10, 8, K, 1)
        self.conv2 = nn.ModuleList([CCL(8, 8, K, 1) for _ in range(L)])
        self.isRelu = relu
    def forward(self, x):
        x = self.conv1(x)
        if self.isRelu:
            x = F.relu(x)
        for i in range(L):
            x = self.conv2[i](x)
            if self.isRelu:
                x = F.relu(x)
        return x



if __name__ == '__main__':
    fig = plt.figure(figsize=(4, 3))
    ax = fig.add_subplot(111)
    markers = ['navy','r','g']
    for L in [1, 3, 5]:
        K = 3
        res_step = 10
        resolutions = range(1, 500, res_step)
        N_out = 10
        N = 10
        isRelu = False

        phi_1 = PHI_1(relu=isRelu).to(device)
        for p in phi_1.parameters():
            p.requires_grad = False
        #bigImg = torch.randn(N, 1, 100, 1000)
        deltas = []
        for res in resolutions:
            delta = []
            for n in range(N_out):
                Roll = 2 * np.pi * np.random.rand(1).item()
                #ind = np.linspace(0, 999, res, dtype=int)
                #sampImg = bigImg[:, :, :,ind].to(device)
                sampImg = torch.randn(N, 10, 100, res).to(device)
                in_w = sampImg.shape[2]
                LR_sampImg = sampImg.roll(int((in_w * Roll) // (2 * np.pi)), -1)
                out_LR = phi_1(LR_sampImg)
                out = phi_1(sampImg)
                out_w = out.shape[2]
                LR_out = out.roll(int((out_w * Roll) // (2 * np.pi)), -1)
                delta.append((LR_out - out_LR).std().detach().cpu() / sampImg.std().detach().cpu())
            deltas.append(np.array(delta).mean())
        figpath = 'act_' + str(isRelu) + '_layer_' + str(L) + '.npy'
        np.save(figpath,{'deltas':deltas,'resolutions':resolutions})



        ax.plot(resolutions,deltas, c=markers[L//2], label = '$\#$ layers = '+ str(L))
        # figure info
        ax.set_xlabel("Resolution")
        ax.set_ylabel(r'Equivariance error $\varepsilon$')
        ax.set_xlim((0, 500))
        ax.set_ylim((0,5e-7))
        ax.set_xticks([0,100,200,300,400,500])

    ax.legend(fontsize=10)
    # save the plots
    figpath = 'results\\act_'+str(isRelu)+'_layer_'+str(L)+'_K_'+str(K)+'.pdf'
    plt.savefig(figpath, dpi=600, pad_inches=.1, bbox_inches='tight')
    ax.legend()
# deltass = []
# for f in range(N_out_out):
#     bigImg = torch.randn(N, 1, 100, 1000)
#     deltas = []
#     for n in range(N_out):
#         Roll = 2 * np.pi * np.random.rand(1).item()
#         delta = []
#         for res in resolutions:
#             # ind = np.linspace(0, 999, res, dtype=int)
#             # sampImg = bigImg[:, :, :,ind].to(device)
#             sampImg = torch.randn(N, 1, 100, res).to(device)
#             in_w = sampImg.shape[2]
#             LR_sampImg = sampImg.roll(int((in_w * Roll) // (2 * np.pi)), -1)
#             out_LR = phi_1(LR_sampImg)
#             out = phi_1(sampImg)
#             out_w = out.shape[2]
#             LR_out = out.roll(int((out_w * Roll) // (2 * np.pi)), -1)
#             delta.append((LR_out - out_LR).std().detach().cpu() / sampImg.std().detach().cpu())
#         deltas.append(delta)
#     deltass.append(np.array(deltas).mean(axis=0).tolist())
# Del = np.array(deltass).mean(axis=0)
# del phi_1
# ax.plot(resolutions, Del)
# # figure info
# ax.set_xlabel("Resolution")
# ax.set_ylabel("$\Delta$")
# ax.set_xlim((0, 500))
# ax.set_ylim((0, 3e-7))
#
# # save the plots
# plt.savefig(figpath, dpi=600, pad_inches=.1, bbox_inches='tight')