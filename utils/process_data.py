import random
import os
import math

from vtkmodules import all as vtk
from vtkmodules.util import numpy_support
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import KDTree
from multiprocessing import Pool
import torch
from torch.utils.data import Dataset
from utils.sdf import SDFRead
from thingking import loadtxt
from utils.attr_kdtree import KDTree as AKDTree


class PointData(Dataset):
    #dataset for individual point
    def __init__(self,data ,k ,r ,ball ,sampler ):
        self.k = int(k)
        self.r = r
        self.ball = ball
        # assume data alreade normalized coord [0, 1] attr [0.1, 0.9]
        coord = data[:,:3]
        attr = data[:,3:]
        
        kd = KDTree(coord,leafsize=100)

        if isinstance(sampler,int):
            sample_id = uniform_sample(sampler,attr)
            self.center = data[sample_id,:3]
        else:
            # use input index as samples
            sampler = np.array(sampler)
            if len(sampler.shape) == 1:
                # sample index
                sample_id = sampler
                self.center = data[sample_id,:3]
            else:
                # sample centers (normalized)
                self.center = sampler
        self.sample_id = sample_id
        
        if self.ball:
            self.nn = kd.query_ball_point(self.center,self.r,workers=8, return_sorted=False)
            # return_sorted means sort by index but not distance
        else:
            self.dist, self.nn = kd.query(self.center,self.k,workers=8)
            # already assume ordered by distance
        self.data = data
        self.center = self.center[:,None,:]

    def __getitem__(self, index):
        # renormalize the point cloud
        nn_id = self.nn[index]
        center = self.center[index]
        pc = self.data[nn_id]
        pc[...,:3] -= center[...,:3]
        if self.ball:
            # reorder the point cloud according to distance
            dist = np.sum(pc[:,:3]**2,axis=1)
            order = np.argsort(dist)
            pc = pc[order]
        return pc
    def __len__(self):
        return len(self.nn)

def uniform_sample(sample_size,attr):
    # uniform samples of attributes
    max_level = round(math.log2(sample_size))
    attr_kd = AKDTree(attr, leafsize=1, max_level=max_level)
    leaf = all_leaf_nodes_at_level(attr_kd.tree, max_level)
    rand = np.random.rand((len(leaf)))
    idx = []
    for i,l in enumerate(leaf):
        indices = l.idx
        idx.append(indices[int(len(indices)*rand[i])])
    return idx

def halo_reader(filename):
    try:
        ID, DescID, Mvir, Vmax, Vrms, Rvir, Rs, Np, x, y, z, VX, VY, VZ, JX, JY, JZ, Spin, rs_klypin, Mvir_all, M200b, M200c, M500c, M2500c, Xoff, Voff, spin_bullock, b_to_a, c_to_a, A_x_, A_y_, A_z_, b_to_a_500c_, c_to_a_500c_, A_x__500c_, A_y__500c_, A_z__500c_, TU, M_pe_Behroozi, M_pe_Diemer = \
            loadtxt(filename, unpack=True)
        r = Rvir/1000
        try:
            halo_num = len(x)
            return np.stack([x,y,z],axis=1),r
        except TypeError:
            return np.array([[x,y,z]]),np.array([r])
    except ValueError:
        return [],[]

def IoU(predict,target):
    assert len(predict) == len(target)
    predict = np.array(predict)
    target = np.array(target)
    union = np.logical_or(predict,target)
    inter = np.logical_and(predict,target)
    return np.sum(inter)/np.sum(union)

def scatter_3d(array,vmin=None,vmax=None,threshold = -1e10,center=None,save=False,fname=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    array = array[array[:,3] > threshold]
    ax.scatter(array[:,0],array[:,1],array[:,2],c=array[:,3],marker='.',vmin=vmin,vmax=vmax)
    if center is not None:
        ax.scatter(center[0],center[1],center[2],c="red",marker='o')
    # ax2 = fig.add_subplot(122,projection='3d',sharex=ax,sharey=ax,sharez=ax)
    # ax2.scatter(array2[:,0],array2[:,1],array2[:,2],c=array2[:,3],marker='^',vmin=-1,vmax=1)
    if save:
        plt.savefig(fname)
    else:
        plt.show()


def data_reader(filename, type):
    if type == 'cos':
        data = sdf_reader(filename)
        attr_min = np.array([-2466, -2761, -2589, -17135.6, -20040, -20096, -6928022])
        attr_max = np.array([2.7808181e+03, 2.9791230e+03, 2.6991892e+03, 1.9324572e+04, 2.0033873e+04, 1.7973633e+04, 6.3844562e+05])
        # data_min=  np.array([0,0,0, -2466, -2761, -2589, -17135.6, -20040, -20096, -6928022])
        # data_max=  np.array([6.2500008e+01, 6.2500000e+01, 6.2500000e+01, 2.7808181e+03, 2.9791230e+03, 2.6991892e+03, 1.9324572e+04, 2.0033873e+04, 1.7973633e+04, 6.3844562e+05])
        # mean = [30.4, 32.8, 32.58, 0, 0, 0, 0, 0, 0, -732720]
        # std = [18.767, 16.76, 17.62, 197.9, 247.2, 193.54, 420.92, 429, 422.3, 888474]
    elif type == 'fpm':
        data = fpm_reader(filename)
        attr_min = np.array([0, -5.63886223e+01, -3.69567909e+01, -7.22953186e+01])
        attr_max = np.array([357.19000244, 38.62746811, 48.47133255, 50.60621262])
        # data_min = np.array([-5, -5, 0, 0, -5.63886223e+01, -3.69567909e+01, -7.22953186e+01])
        # data_max = np.array([ 5, 5, 10.00022221, 357.19000244, 38.62746811, 48.47133255, 50.60621262])
        # mean = [0, 0, 5, 23.9, 0, 0, 0.034]
        # std = [2.68, 2.68, 3.09, 55.08, 0.3246, 0.3233, 0.6973]
    elif type =='jet3b':
        data = jet3b_reader(filename)
        attr_min = np.array([-1.50166025e+01, 1.47756422e+00])
        attr_max = np.array([1.24838667e+01, 1.00606432e+01])
    # normalize coordinates
    coord = data[:,:3]
    coord_min = coord.min(0)
    coord_max = coord.max(0)
    coord = (coord - coord_min) / (coord_max-coord_min)
    coord = np.clip(coord,0,1)
    # normalize attr to [0.1,0.9]
    attr = data[:,3:]
    attr = (attr - attr_min) / (attr_max - attr_min)
    attr = attr * 0.8 + 0.1
    attr = np.clip(attr,0.1,0.9)

    data = np.concatenate([coord,attr], axis=1)
    return np.float32(data) #convert all data to float32

def load_helper(args):
    f = args[0]
    type = args[1]
    return data_reader(f,type)

def all_file_loader(file_list, data_type):
    if data_type == 'fpm':
        data = []
        for f in file_list:
            data.append(data_reader(f,data_type))
    elif data_type == 'jet3b' or data_type == 'cos':
        pool = Pool(8)
        data = pool.map(load_helper,zip(file_list,[data_type]*len(file_list)))
    return data
    
def collate_ball(data):
    dim = data[0].shape[1]
    len_list = list(d.shape[0] for d in data)
    max_len = max(len_list)
    out_tensor = torch.zeros((len(data),max_len,dim),dtype=torch.float32)
    mask_tensor = torch.ones((len(data),max_len,1),dtype=torch.bool)
    for i, d in enumerate(data):
        out_tensor[i,:len(d),:] = torch.from_numpy(d)
        mask_tensor[i,len(d):,:] = False
    return out_tensor, mask_tensor


def jet3b_reader(filename):
    ds = SDFRead(filename)
    rho = np.array(ds['rho'])
    temp = np.array(ds['temp'])
    rho = np.log10(rho)
    temp = np.log10(temp)
    data = np.stack([ds['x'],ds['y'],ds['z'],rho,temp],axis=1)
    return data
    

def fpm_reader(filename):
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(filename)
    reader.Update()
    vtk_data = reader.GetOutput()
    coord = numpy_support.vtk_to_numpy(vtk_data.GetPoints().GetData())
    concen = numpy_support.vtk_to_numpy(vtk_data.GetPointData().GetArray(0))[:,None]
    velocity = numpy_support.vtk_to_numpy(vtk_data.GetPointData().GetArray(1))
    point_data = np.concatenate((coord,concen,velocity),axis=-1)
    return point_data

def sdf_reader(filename):
    particles = SDFRead(filename)
    h_100 = particles.parameters['h_100']
    cosmo_a = particles.parameters['a']
    kpc_to_Mpc = 1./1000
    convert_to_cMpc = lambda proper: (proper ) * h_100 * kpc_to_Mpc / cosmo_a + 31.25
    numpy_data = np.array(list(particles.values())[2:-1]).T # coord, velocity, acceleration, phi
    numpy_data[:,:3] = convert_to_cMpc(numpy_data[:,:3])

    return numpy_data


def all_leaf_nodes_at_level(node, max_level):
    if isinstance(node,AKDTree.leafnode):
        return [node]
    elif node.level == max_level:
        return [node]
    else:
        return(all_leaf_nodes_at_level(node.less, max_level)+all_leaf_nodes_at_level(node.greater, max_level))

def collect_file(directory,mode,shuffle=True):
    file_list = []
    for (dirpath, dirnames, filenames) in os.walk(directory):
        for filename in filenames:
            if mode == "fpm":
                if filename.endswith(".vtu") and filename != "000.vtu":
                    inp = os.sep.join([dirpath, filename])
                    file_list.append(inp)
            elif mode == "cos":
                if "ds14" in filename.split('_'):
                    inp = os.sep.join([dirpath, filename])
                    file_list.append(inp)
            elif mode == "jet3b":
                if "run3g" in filename.split('_'):
                    inp = os.sep.join([dirpath, filename])
                    file_list.append(inp)
    if shuffle:
        random.shuffle(file_list)
    else:
        file_list.sort()
    return file_list
    
def min_max(file_list,mode):
    all_min = None
    all_max = None
    for i,f in enumerate(file_list):
        print("processing file {}/{}".format(i,len(file_list)),end='\r')
        data = data_reader(f, mode)
        f_min = np.min(data,axis=0)
        f_max = np.max(data,axis=0)
        if all_min is None:
            all_min = f_min
            all_max = f_max
            total = np.zeros((data.shape[1]))
            length = 0
        else:
            all_min = np.where(all_min < f_min,all_min,f_min)
            all_max = np.where(all_max > f_max,all_max,f_max)
        total += np.sum(data,axis = 0)
        length += len(data)
    mean = total/length
    print("mean: ", mean)
    print("min: ", all_min)
    print("max: ", all_max)
    return mean,all_min,all_max

def std(file_list,mean,mode="fpm"):
    for i,f in enumerate(file_list):
        print("processing file {}/{}".format(i,len(file_list)),end='\r')
        data = data_reader(f, mode)
        if i==0:
            total = np.zeros((data.shape[1]))
            length = 0
        data = (data - mean) ** 2
        total += np.sum(data,axis = 0)
        length += len(data)
    std = np.sqrt(total/length)
    print("std: ", std)
    return std

def numpy_to_vtp(position:np.array,array_dict:dict):
    vtk_position = numpy_support.numpy_to_vtk(position)
    points = vtk.vtkPoints()
    points.SetData(vtk_position)
    data_save = vtk.vtkPolyData()
    vertices = vtk.vtkPolyVertex()
    length = len(position)
    vertices.GetPointIds().SetNumberOfIds(length)
    for i in range(0, length):
        vertices.GetPointIds().SetId(i, i)
    vert_cell = vtk.vtkCellArray()
    vert_cell.InsertNextCell(vertices)
    data_save.SetVerts(vert_cell)
    data_save.SetPoints(points)
    pd = data_save.GetPointData()
    for k, v in array_dict.items():
        vtk_array = numpy_support.numpy_to_vtk(v)
        vtk_array.SetName(k)
        pd.AddArray(vtk_array)
    return data_save

def numpy_to_vts(data,nx,ny,nz,array_dict):
    assert data.shape[0] == nx*ny*nz
    vol = vtk.vtkStructuredPoints()
    vol.SetDimensions(nx,ny,nz)
    vol.SetOrigin(*list(data[:,:3].min(0)))
    sx,sy,sz = data[:,:3].max(0) - data[:,:3].min(0)
    sx /= nx-1
    sy /= ny-1
    sz /= nz-1
    vol.SetSpacing(sx,sy,sz)
    for k, v in array_dict.items():
        vtk_array = numpy_support.numpy_to_vtk(v)
        vtk_array.SetName(k)
        vol.GetPointData().AddArray(vtk_array)

    return vol

def vtk_write(data_save,filename:str):
    writer = vtk.vtkXMLDataSetWriter()
    writer.SetFileName(filename)
    writer.SetInputData(data_save)
    writer.Write()
    


def resample(data,nx,ny,nz,h):
    '''
    Resample particle data to regular grids
    '''
    kernel = lambda p: 0 if p > 3 else 1/(math.pow(math.pi, 3/2) * math.pow(h,3)) * math.pow(math.e, -p**2)
    pos = data[:,:3]
    kd = KDTree(pos,leafsize=100)
    x = np.linspace(min(data[:,0]), max(data[:,0]),nx)
    y = np.linspace(min(data[:,1]), max(data[:,1]),ny)
    z = np.linspace(min(data[:,2]), max(data[:,2]),nz)
    zz,yy,xx = np.meshgrid(z,y,x,indexing='ij')
    grid = np.stack((xx.flatten(),yy.flatten(),zz.flatten()),axis=1)
    new_data = np.zeros((len(grid),data.shape[1]))
    new_data[:,:3] = grid
    neighbor = kd.query_ball_point(grid, h,workers=8)
    for j,nn in enumerate(neighbor):
        print("%d/%d"%(j,len(new_data)),end='\r')
        if len(nn) > 0:
            nn_data = data[nn]
            nn_pos = nn_data[:,:3]
            r = np.sum((nn_pos - grid[j]) ** 2,axis=1) ** 0.5
            p = r/h
            averaged = np.average(nn_data[:,3:],axis=0,weights=list(map(kernel,p)))
            new_data[j,3:] = averaged
    return new_data


def halo_writer(center,Rvir,outputname):
    haloData = vtk.vtkAppendPolyData()
    for i in range(len(center)):
        print(i,"/",len(center),end='\r')
        s = vtk.vtkSphereSource()
        s.SetCenter(*center[i])
        s.SetRadius(Rvir[i])
        s.Update()
        input1 = vtk.vtkPolyData()
        input1.ShallowCopy(s.GetOutput())
        haloData.AddInputData(input1)
    haloData.Update()
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetInputConnection(haloData.GetOutputPort())
    writer.SetFileName(outputname)
    writer.Write()

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
    return dis

def chamfer_distance_loss(output, target):
    recon_loss = 0
    for i in range(len(output)):
        pc1 = output[i]
        pc2 = target[i]
        recon_loss += nn_distance(pc1,pc2)
    recon_loss /= len(output)

    return recon_loss

def masked_mse_loss(output, target, mask):
    '''
        output: (B,N,D)
        target: (B,N,D)
        mask: (B,N,1)
    '''
    _,_,D = output.shape
    out = torch.sum(((output-target)*mask)**2.0)  / (torch.sum(mask)*D)
    return out
    

def plot_loss(filename):
    epoch = 1
    loss_sum = 0.0
    loss_list = []
    count = 0
    with open(filename,"r") as f:
        for line in f:
            if "====>" in line:
                line_list = line.split(' ')
                loss = float(line_list[5])
                if epoch == int(line_list[2]):
                    loss_sum += loss
                    count += 1
                else:
                    epoch = int(line_list[2])
                    loss_list.append(loss_sum/count)
                    loss_sum = loss
                    count = 1
    print(loss_list)
