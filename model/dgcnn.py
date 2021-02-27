import os
import sys
import copy
import math
from contextlib import contextmanager
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from sklearn.cluster import KMeans

class Conv(nn.Module):
    def __init__(self,input_channels,output_channels):
        super().__init__()
        self.bn = nn.BatchNorm1d(output_channels)
        self.conv = nn.Conv1d(input_channels,output_channels,kernel_size=1,bias=False)
        self.rl = nn.ReLU()
        self.dp = nn.Dropout(p=0)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.rl(x)
        x = self.dp(x)
        return x


class ConvStack(nn.Module):
    def __init__(self, input_channel):
        super(ConvStack, self).__init__()
        self.conv1 = Conv(input_channel,64).to("cuda")
        self.conv2 = Conv(64,64).to("cuda")
        self.conv3 = MLP(128,[64]).to("cuda")

    def forward(self, x):
        x = self.conv1(x)                       # (batch_size, input_channel, k) -> (batch_size, 32, k)
        x = self.conv2(x)                       # (batch_size, 32, k) -> (batch_size, 64, k)
        x_max = x.max(dim=-1, keepdim=False)[0] # (batch_size, 64, k) -> (batch_size, 64)
        x_mean = x.mean(dim=-1, keepdim=False)  # (batch_size, 64, k) -> (batch_size, 64)
        x = torch.cat((x_max,x_mean),1)         # (batch_size, 64) -> (batch_size, 128)
        x = self.conv3(x)                       # (batch_size, 128) -> (batch_size, 64)
        return x

class MLP(nn.Module):
    def __init__(self, input_channel, channels = []):
        super().__init__()
        self.ln = []
        self.bn = []
        self.num_layers = len(channels)
        for i,c in enumerate(channels):
            if i==0:
                self.ln.append(nn.Linear(input_channel,c,bias=False).to("cuda"))
            else:
                self.ln.append(nn.Linear(last_c,c,bias=False).to("cuda"))
            self.bn.append(nn.BatchNorm1d(c).to("cuda"))
            last_c = c

    def forward(self, x):
        for i in range(self.num_layers):
            x = self.ln[i](x)
            x = self.bn[i](x)
            x = F.relu(x)
            x = F.dropout(x,p=0)
        return x

class Encoder(nn.Module):
    def __init__(self,input_channel):
        super().__init__()
        self.mlp = MLP(256,[128,64,32])
        self.cs0 = MLP(input_channel,[64])
        self.cs1 = ConvStack(input_channel)
        self.cs2 = ConvStack(input_channel)
        self.cs3 = ConvStack(input_channel)
    
    def forward(self, input_data:list):
        x0,x1,x2,x3 = input_data
        batch_size = x0.size(0)
        dim = x0.size(1)
        x1 = self.cs1(x1)
        x2 = self.cs2(x2)
        x3 = self.cs3(x3)
        x0 = self.cs0(x0)
        x = torch.cat((x0,x1,x2,x3),dim=1)      # (batch_size, 64) -> (batch_size, 256)
        x = self.mlp(x)
        return x

class SSAE(nn.Module):
    def __init__(self, input_channel,num_clu,center = None):
        super().__init__()
        self.enc = Encoder(input_channel)
        self.dec_cls = nn.Sequential(
            MLP(32,[64,128]),
            nn.Linear(128,num_clu)
        ) 
        self.dec_rec = nn.Sequential(
            MLP(32,[64,128]),
            nn.Linear(128,input_channel)
        )
        self.num_clu = num_clu
        self.z_acc = torch.zeros((0,32)).cuda()
        self.centers = torch.zeros((num_clu,32)).cuda()

    def set_center(self,center):
        self.centers = center
        self.fitter = KMeans(self.num_clu,init=self.centers,n_init=1)
        self.fitter.cluster_centers_ = center.cpu().numpy()

    def init_center(self,sample):
        with torch.no_grad():
            z = self.enc(sample)
            self.fitter = KMeans(self.num_clu,n_init=1)
            self.fitter.fit(z.cpu())
            self.centers = torch.Tensor(self.fitter.cluster_centers_)

    def update_center(self):
        with torch.no_grad():
            self.fitter = KMeans(self.num_clu,init=self.centers,n_init=1)
            self.fitter.fit(self.z_acc.cpu())
            self.centers = torch.Tensor(self.fitter.cluster_centers_)
            self.z_acc = torch.zeros((0,32)).cuda()

    def forward(self, input_data:list):
        z = self.enc(input_data)
        y_cls = self.dec_cls(z)
        y_rec = self.dec_rec(z)
        self.z_acc = torch.cat((self.z_acc,z))
        return z, y_cls, y_rec

    def loss(self, x, z, y_cls, y_rec):
        rec_loss = torch.mean(torch.sum((y_rec - x) ** 2,dim=-1),dim=0)
        clu_id = torch.LongTensor(self.fitter.predict(z.detach().cpu())).cuda()
        cls_loss = F.cross_entropy(y_cls,clu_id)
        clu_loss = torch.mean(torch.sum((torch.index_select(self.centers.cuda(),0,clu_id) - z) ** 2,dim=-1),dim=0)
        return (rec_loss, clu_loss, cls_loss)


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout

class Transform_Net(nn.Module):
    def __init__(self, args):
        super(Transform_Net, self).__init__()
        self.args = args
        self.k = 3
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv1d(128, 1024, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn3 = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, 256, bias=False)
        self.bn4 = nn.BatchNorm1d(256)

        self.transform = nn.Linear(256, 3*3)
        init.constant_(self.transform.weight, 0)
        init.eye_(self.transform.bias.view(3, 3))

    def forward(self, x):
        batch_size = x.size(0)

        x = self.conv1(x)                       # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 128, num_points, k)
        x = x.max(dim=-1, keepdim=False)[0]     # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x = self.conv3(x)                       # (batch_size, 128, num_points) -> (batch_size, 1024, num_points)
        x = x.max(dim=-1, keepdim=False)[0]     # (batch_size, 1024, num_points) -> (batch_size, 1024)

        x = F.leaky_relu(self.bn3(self.linear1(x)), negative_slope=0.2)     # (batch_size, 1024) -> (batch_size, 512)
        x = F.leaky_relu(self.bn4(self.linear2(x)), negative_slope=0.2)     # (batch_size, 512) -> (batch_size, 256)

        x = self.transform(x)                   # (batch_size, 256) -> (batch_size, 3*3)
        x = x.view(batch_size, 3, 3)            # (batch_size, 3*3) -> (batch_size, 3, 3)

        return x


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None, dim9=False):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False:
            idx = knn(x, k=k)   # (batch_size, num_points, k)
        else:
            idx = knn(x[:, 6:], k=k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2)
  
    return feature      # (batch_size, 2*num_dims, num_points, k)