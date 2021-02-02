from torch import nn
import torch
from torch.nn import functional as F
import numpy as np
import random

class AE(nn.Module):
    def __init__(self,args):
        """
        input size: B * N * C
        """
        super(AE, self).__init__()
        self.vector_length = args.vector_length
        self.num_channel = args.dim
        self.prediction_num = args.k
        self.mode = args.mode
        self.have_label = args.have_label
        self.pointnet = nn.Sequential(
            nn.Linear(self.num_channel,64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),

            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),

            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),

            nn.Linear(128, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
        )
        self.fc01 = nn.Linear(1024,self.vector_length)

        self.fc_decode = nn.Sequential(
            nn.Linear(self.vector_length,1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),

            nn.Linear(1024,1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),

            nn.Linear(1024,self.prediction_num * self.num_channel),
        )

        self.cls = nn.Sequential(
            nn.BatchNorm1d(self.vector_length),
            nn.LeakyReLU(),

            nn.Linear(self.vector_length,self.vector_length),
            nn.BatchNorm1d(self.vector_length),
            nn.LeakyReLU(),

            nn.Linear(self.vector_length,self.vector_length//2),
            nn.BatchNorm1d(self.vector_length//2),
            nn.LeakyReLU(),

            nn.Linear(self.vector_length//2,2),
        )

    def encode(self,x,mask=None):
        batch_size = len(x)
        n_points = x.shape[1]
        n_channel = self.num_channel
        if self.mode == "ball":
            x = x.view(-1,n_channel)
            x = self.pointnet(x)
            n_dim = x.shape[-1]
            x = x.view(batch_size, n_points, -1)
            new_tensor = torch.zeros((batch_size,n_dim),dtype=torch.float,device="cuda")
            for i,pc in enumerate(x):
                pc = pc[:mask[i]]
                new_tensor[i] = torch.max(pc,dim=-2)[0]
            x = new_tensor
        elif self.mode == "knn":
            n_points = x.shape[1]
            n_channel = self.num_channel
            x = x.view(-1,n_channel)
            x = self.pointnet(x)
            x = x.view(batch_size, n_points, -1)
            x = torch.max(x,dim=-2)[0]
        x = self.fc01(x)
        x = F.leaky_relu(x)
        return x
    
    def decode(self,z):
        y = self.fc_decode(z)
        y = y.view(-1,self.prediction_num,self.num_channel)
        return y

    def loss(self,output, target):
        recon_loss = 0
        for i in range(len(output)):
            pc1 = output[i]
            pc2 = target[i]
            recon_loss += nn_distance(pc1,pc2)
        recon_loss /= len(output)

        return recon_loss

    def forward(self, x, mask=None):
        z = self.encode(x, mask)
        if self.have_label:
            y = self.cls(z)
        else:
            y = self.decode(z)
        return y

def nn_distance(pc1,pc2):
    np1 = pc1.shape[0]
    np2 = pc2.shape[0]
    pc1 = pc1[:,None,:].repeat(1,np2,1)
    pc2 = pc2[None,:,:].repeat(np1,1,1)
    d = (pc1 - pc2)**2
    d = torch.sum(d,dim=-1)

    d1 = torch.min(d,dim=-1)[0]
    d2 = torch.min(d,dim=-2)[0]
    dis = torch.cat((d1,d2),dim=0)
    dis = torch.mean(dis)
    # print(dis)
    return dis

