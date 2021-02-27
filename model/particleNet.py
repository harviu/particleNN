from torch import nn
import torch
from torch.nn import functional as F

class ParticleNet(nn.Module):
    """
    ParticleNet based on GeoConv
    input size: 1 * N * C
    Make batch size 1 and change the patch size to fit the GRAM
    """
    def __init__(self,args):
        super().__init__()
        r = args.r
        self.sa1 = PointNetSetAbstraction(1024, r, 64, 4 , [64, 128], [64, 64], False)
        self.sa2 = PointNetSetAbstraction(256, r/0.4, 64, 128 , [128, 256], [64, 64], False)
        self.sa3 = PointNetSetAbstraction(64, r/(0.4*0.4), 64, 256 , [256, 512], [64, 64], False)
        self.fp3 = PointNetFeaturePropagation(r/(0.4*0.4), 64, 512, [256, 256], [64,64])
        self.fp2 = PointNetFeaturePropagation(r/0.4, 64, 256, [256, 128], [64,64])
        self.fp1 = PointNetFeaturePropagation(r, 64, 128, [64, 64], [64,64])
        self.fc1 = nn.Linear(64,32)
        self.bn1 = nn.BatchNorm1d(32)
        self.fc2 = nn.Linear(32,4,1)

    def forward(self, xyz):
        l0_points = xyz[:,:,3:]
        l0_xyz = xyz[:,:,:3]

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        l2_points = self.fp3(l2_xyz, l3_xyz, None, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, None, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points) # (B, N,C)

        B, N, C = l0_points.shape
        x = self.fc1(l0_points.view(B*N,-1))
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        x = x.view(B,N,-1)
        return x


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def largest_variance_sample(xyz, points, npoint,radius,nsample):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        points: input signal [B, N, D]
        radius: variance calculation radius
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, _, D = points.shape
    sqrdists = square_distance(xyz,xyz) #（B, N, N)
    mask = sqrdists < radius ** 2
    var_array = torch.zeros((B,N),dtype=torch.float).to(device)
    for nn in range(N):
        mm = mask[:,:,nn]
        count = torch.sum(mm,dim=-1) # (B)
        zero_signal = torch.zeros((B,N,D),dtype=torch.float).to(device)
        masked_points = torch.where(mm.view(B,N,1).repeat(1,1,D), points, zero_signal) # (B, N, D)
        mean_signal = torch.sum(masked_points,dim=-2,keepdim=True) / count.view(B,1,1) # (B, 1, D)
        var = (masked_points - mean_signal) ** 2 # (B, N, D)
        var = torch.where(mm.view(B,N,1).repeat(1,1,D),var,zero_signal)
        var = torch.sum(var,dim=-2) / count
        var = torch.sum(var,dim=-1) # (B)
        var_array[:,nn] = var
    idx = torch.argsort(var_array,dim=-1,descending=True)
    idx = idx[:,:npoint]
    return idx

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx

def sample_and_group (npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint: output number of points
        radius: query radius
        nsample: max number of neighbor
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: output position
        new_signal: output attribute
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz,npoint) 
    torch.cuda.empty_cache()
    new_xyz = index_points(xyz, fps_idx) # [B, npoint, C]
    torch.cuda.empty_cache()
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    torch.cuda.empty_cache()
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    torch.cuda.empty_cache()
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    torch.cuda.empty_cache()

    # if points is not None:
    #     grouped_points = index_points(points, idx)
    #     new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    # else:
    #     new_points = grouped_xyz_norm
    new_points = index_points(points, idx)
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points

def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    # if points is not None:
    #     new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    # else:
        # new_points = grouped_xyz
    new_points = points.view(B, 1, N, -1)
    return new_xyz, new_points

class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, out_chnls, mid_chnls , group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        # self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel, mid_channel in zip(out_chnls,mid_chnls):
            self.mlp_convs.append(GeoConv(last_channel,out_channel,mid_channel,radius,2*radius,nsample))
            # self.mlp_convs.append(nn.Conv1d(last_channel,out_channel,1))
            # self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, N,C]
            points: input points data, [B, N,D]
        Return:
            new_xyz: sampled points position data, [B, S,C]
            new_points_concat: sample points feature data, [B, S,D']
        """
        # for conv,bn in zip(self.mlp_convs,self.mlp_bns):
        for conv in self.mlp_convs:
            points = conv(xyz,points)
            # points = F.relu(bn(conv(points)))

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, D]
        new_points = torch.max(new_points, 2)[0]
        return new_xyz, new_points


class PointNetFeaturePropagation(nn.Module):
    def __init__(self,radius,nsample, in_channel, out_chnls, mid_chnls):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        # self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel, mid_channel in zip(out_chnls,mid_chnls):
            self.mlp_convs.append(GeoConv(last_channel,out_channel,mid_channel,radius,radius * 2,nsample))
            # self.mlp_convs.append(nn.Conv1d(last_channel,out_channel,1))
            # self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, N,C]
            xyz2: sampled input points position data, [B, S,C]
            points1: input points data, [B, N,D]
                points1 is used to concatenate the embedding from encoder (like U-Net)
            points2: input points data, [B, N,S]
        Return:
            new_points: upsampled points data, [B, N,D']
        """
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2) #(B,N,D)

        if points1 is not None:
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        # for conv,bn in zip(self.mlp_convs,self.mlp_bns):
        for conv in self.mlp_convs:
            new_points = conv(xyz1,new_points)
            # new_points = F.relu(bn(conv(new_points)))
        return new_points


def aggregate(direction,xyz,radius,decay_radius):
    """
        Input:
            direction: (B, N, D * 6) 
            xyz: (B, 3, N)
            radius: the inner radius of local ball of each point.
            decay radius: the outer radius of local ball of each point
        Return:
            direction:(B,D,N)
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, _, D = direction.shape
    D //= 6

    sqdist = square_distance(xyz,xyz)
    sqdist, idx = torch.sort(sqdist,dim=-1)
    mask = sqdist < decay_radius ** 2
    count = torch.sum(mask,dim=-1)
    S = torch.max(count)
    idx = idx[:,:,:S]
    direction = index_points(direction,idx)

    vector = index_points(xyz,idx) #(B,N,S,3)
    vector = vector-vector[:,:,0,:].view(B,N,1,3)
    vector_norm = torch.norm(vector,dim=-1)  # (B,N,S)
    axis = [[0,0,1],[0,0,-1],[0,1,0],[0,-1,0],[1,0,0],[-1,0,0]]
    axis = torch.FloatTensor(axis).to(device).permute(1,0) #(3,6)
    cos = torch.matmul(vector,axis) #(B,N,S,6)
    cos[cos<0] = 0
    cos /= vector_norm.view(B,N,S,1) + 1e-8
    cos = cos ** 2
    dist_weight = 1 - (vector_norm**2 - radius**2)/(decay_radius**2-radius**2)
    dist_weight[dist_weight < 0] = 0
    dist_weight /= torch.sum(dist_weight,dim = -1,keepdim=True) #(B,N)
    cos = cos * dist_weight.view(B,N,S,1) #(B,N,S,6)
    direction = direction.view(B,N,S,D,6) * cos.view(B,N,S,1,6)
    direction = direction.sum(dim=-1) #(B,N,S,D)
    direction = direction.sum(dim=-2) #(B,N,D)
    return direction


class GeoConv(nn.Module):
    ''' GeoCNN Geo-Conv
        Input:
            points: (B, N,D) 
            xyz: (B, N,C)
            out_channel: the count of output channels
            mid_channel: the count of output channels of bypass
            radius: the inner radius of local ball of each point.
            decay radius: the outer radius of local ball of each point
        Output:
            output: (B,N,D)
    '''
    def __init__(self,in_channel,out_channel,mid_channel,radius,decay_radius,nsample):
        super().__init__()
        self.center_mlp = nn.Linear(in_channel,out_channel)
        self.direction_mlp = nn.Linear(in_channel,mid_channel * 6)
        self.direction_bn = nn.BatchNorm1d(mid_channel)
        self.direction_mlp2 = nn.Linear(mid_channel,out_channel)
        self.last_bn = nn.BatchNorm1d(out_channel)
        self.radius = radius
        self.decay_radius = decay_radius
        self.nsample = nsample

    def forward(self,xyz,points):
        B,N,_ = points.shape
        points = points.view(B*N,-1)
        center = self.center_mlp(points)
        direction = self.direction_mlp(points)
        direction = direction.view(B,N,-1)
        direction = aggregate(direction,xyz,self.radius,self.decay_radius)
        direction = direction.view(B*N,-1)
        torch.cuda.empty_cache()
        direction = self.direction_bn(direction)
        direction = F.relu(direction)
        direction = self.direction_mlp2(direction)
        output = center + direction
        output = self.last_bn(output)
        output = F.relu(output)
        return output.view(B,N,-1)

