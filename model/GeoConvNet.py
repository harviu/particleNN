from torch import nn
import torch
from torch.nn import functional as F

class GeoConvNet(nn.Module):
    def __init__(self, lat_dim, input_dim, ball, neuron_num, r):
        """
        input size: B * N * C
        """
        super(GeoConvNet, self).__init__()
        self.vector_length = lat_dim
        self.num_channel = input_dim
        self.ball = ball
        self.geoconv = GeoConv(self.num_channel,neuron_num,neuron_num//4,r * 0.5,r)
        self.fc01 = nn.Linear(neuron_num, self.vector_length)
        self.geodeconv = GeoDeConv(self.vector_length + 3,[neuron_num, neuron_num//4])
        self.fc02 = nn.Linear(neuron_num//4 ,self.num_channel-3)

    def encode(self,x, mask=None):
        xyz = x[:,:,:3]
        points = x[:,:,3:]
        x = self.geoconv(xyz,points)
        x = self.fc01(x)
        return x
    
    def decode(self,z,xyz):
        y = self.geodeconv(xyz,z)
        B, N, D = y.shape
        y = y.view(B*N,D)
        y = self.fc02(y)
        y = torch.sigmoid(y)
        y = y.view(B,N,-1)
        return y

    def forward(self, x, mask=None):
        xyz = x[:,:,:3]
        z = self.encode(x, mask)
        y = self.decode(z, xyz) # no need to add mask to GeoDeConv, mask in loss function instead
        return y

    
class GeoDeConv(nn.Module):
    ''' GeoCNN Deconvolution
        Input:
            points: (B, D) center signal
            xyz: (B, N, 3) recontruction locations
            channels: middle channels
            inner_radius
            outer_radius
        output:
            output: (B,N,D)
        If we choose to predict coordinates, coordinates are not concatenated to the latent.
    '''
    def __init__(self,in_channel,channels):
        super().__init__()
        self.center_mlp = nn.Linear(in_channel,channels[-1])
        self.DeConv = nn.Linear(in_channel,channels[0]*6,bias=False)
        self.mlp = nn.ModuleList()
        self.bn = nn.ModuleList()
        for i,c in enumerate(channels):
            self.mlp.append(nn.Linear(c,channels[i+1],bias=False))
            self.bn.append(nn.BatchNorm1d(channels[i+1]))
            if i+1 == len(channels)-1:
                break

    def forward(self,xyz,signal):
        B, N, _ = xyz.shape
        _, D = signal.shape
        center_input = torch.clone(signal)
        center_xyz = xyz[:,0,:]
        center_input = torch.cat([center_input,center_xyz],dim=-1)
        center = self.center_mlp(center_input) #output size (B,out_channel)
        signal = signal.view(B,1,D).repeat(1,N,1)
        signal = torch.cat([xyz,signal],dim=-1)
        signal = signal.view(B*N,-1)
        signal = self.DeConv(signal)
        signal = signal.view(B,N,-1)
        signal = disperse(signal,xyz)
        signal = signal.view(B*N,-1)
        for i, (m,b) in enumerate(zip(self.mlp,self.bn)):
            if i != len(self.mlp) -1:
                signal = F.relu(b(m(signal)))
            else: 
                signal = m(signal)
                signal = signal.view(B, N, -1)
                signal[:,0,:] = center
                signal = signal.view(B * N, -1)
                signal = F.relu(b(signal))
        signal = signal.view(B, N, -1)
        return signal


def disperse(signal,xyz):
    device = xyz.device
    B, N, _ = xyz.shape
    _, _, D = signal.shape
    D //= 6
    vector_norm = torch.norm(xyz,dim=-1)  # (B,N)
    axis = [[0,0,1],[0,0,-1],[0,1,0],[0,-1,0],[1,0,0],[-1,0,0]]
    axis = torch.FloatTensor(axis).to(device).permute(1,0) #(3,6)
    cos = torch.matmul(xyz,axis) #(B,N,6)
    cos[cos<0] = 0
    cos /= vector_norm.view(B,N,1) + 1e-10
    cos = cos ** 2
    signal = signal.view(B,N,D,6) * cos.view(B,N,1,6)
    signal = signal.sum(dim=-1) #(B,N,D)
    return signal



class GeoConv(nn.Module):
    ''' GeoCNN Geo-Conv
        Input:
            points: (B, N, D) 
            xyz: (B, N, 3)
            out_channel: the count of output channels
            mid_channel: the count of output channels of bypass
            radius: the inner radius of local ball of each point.
            decay radius: the outer radius of local ball of each point
        Output:
            output: (B,N,D)
    '''
    def __init__(self,in_channel,out_channel,mid_channel,inner_radius,outer_radius):
        super().__init__()
        self.center_mlp = nn.Linear(in_channel,out_channel)
        self.direction_mlp = nn.Linear(in_channel,mid_channel * 6,bias=False)
        self.direction_bn = nn.BatchNorm1d(mid_channel)
        self.direction_mlp2 = nn.Linear(mid_channel,out_channel,bias=False)
        self.last_bn = nn.BatchNorm1d(out_channel)
        self.outer_radius = outer_radius
        self.inner_radius = inner_radius

    def forward(self,xyz,points,mask=None):
        B,N,_ = points.shape
        center_point = points[:,0,:]
        center_xyz = xyz[:,0,:]
        center_input = torch.cat([center_point,center_xyz],dim=-1)
        center = self.center_mlp(center_input) #output size (B,out_channel)
        dir_input = torch.cat([points,xyz],dim=-1)
        dir_input = dir_input.view(B*N,-1)
        direction = self.direction_mlp(dir_input)
        direction = direction.view(B,N,-1)
        direction = aggregate(direction,xyz,self.inner_radius,self.outer_radius,mask)
        direction = direction.view(B,-1) # (B,mid_channel)
        direction = self.direction_bn(direction)
        direction = F.relu(direction)
        direction = self.direction_mlp2(direction)
        output = center + direction
        output = self.last_bn(direction)
        output = F.relu(output)
        return output

def aggregate(direction,xyz,inner_radius,outer_radius,mask=None):
    """
        Input:
            direction: (B, N, D * 6) 
            xyz: (B, N, 3)
            mask: (B, N)
            radius: the inner radius of local ball of each point.
            decay radius: the outer radius of local ball of each point
        Return:
            direction:(B,D)
    """
    device = xyz.device
    B, N, _ = xyz.shape
    _, _, D = direction.shape
    D //= 6
    eps = 1e-10

    vector_norm = torch.norm(xyz,dim=-1)  # (B,N)
    axis = [[0,0,1],[0,0,-1],[0,1,0],[0,-1,0],[1,0,0],[-1,0,0]]
    axis = torch.FloatTensor(axis).to(device).permute(1,0) #(3,6)
    cos = torch.matmul(xyz,axis) #(B,N,6)
    cos[cos<0] = 0
    cos /= vector_norm.view(B,N,1) + 1e-10
    cos = cos ** 2
    dist_weight = 1 - (vector_norm**2 - inner_radius**2)/(outer_radius**2-inner_radius**2)
    dist_weight[dist_weight<0] = eps  # assign 0 if d(p,q) is less than 0
    dist_weight[vector_norm<=0] = eps # basically remove center node (or node that is too close)
    if mask is not None: # if the neighbor is ball
        dist_weight[torch.logical_not(mask)] = eps
    dist_weight /= torch.sum(dist_weight,dim = -1,keepdim=True) #(B,N)
    cos = cos * dist_weight.view(B,N,1) #(B,N,6)
    direction = direction.view(B,N,D,6) * cos.view(B,N,1,6)
    direction = direction.sum(dim=-1) #(B,N,D)
    direction = direction.sum(dim=-2) #(B,D)
    return direction