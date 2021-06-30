import random
import os
import sys
import math
import time

from vtkmodules import all as vtk
from vtkmodules.util import numpy_support
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.ckdtree import cKDTree,cKDTreeNode
from multiprocessing import Pool
import torch
from torch.utils.data import Dataset,DataLoader
from yt.utilities.sdf import SDFRead
from thingking import loadtxt

class PointDataFPM(Dataset):
    # dataset in blocks
    def __init__(self,data,args):
        self.r = args.r
        self.p = args.p
        # normalize data to [0, 1]
        data_max = np.array([ 5, 5, 10.00022221, 357.19000244, 38.62746811, 48.47133255, 50.60621262])
        data_min = np.array([-5, -5, 0, 0, -5.63886223e+01, -3.69567909e+01, -7.22953186e+01])
        data = (data - data_min) / (data_max-data_min)
        data = np.clip(data,0,1)
        patch_list = []
        p_each_dim = round(math.pow(args.p,1.0/3))
        ran = 1.0/p_each_dim
        num = 0
        for x  in range(p_each_dim):
            min_x = max(x*ran-self.r,0)
            max_x = min((x+1)*ran+self.r,1)
            filter_x = np.logical_and(data[:,0] > min_x, data[:,0] < max_x)
            for y in range(p_each_dim):
                min_y = max(y*ran-self.r,0)
                max_y = min((y+1)*ran+self.r,1)
                filter_y = np.logical_and(data[:,1] > min_y, data[:,1] < max_y)
                fil_xy = np.logical_and(filter_x,filter_y)
                for z in range(p_each_dim):
                    min_z = max(z*ran-self.r,0)
                    max_z = min((z+1)*ran+self.r,1)
                    filter_z = np.logical_and(data[:,2] > min_z, data[:,2] < max_z)
                    fil = np.logical_and(fil_xy,filter_z)
                    patch_list.append(data[fil])
                    num += len(patch_list[-1])
        self.patch_list = patch_list
    
    def __getitem__(self, index):
        return self.patch_list[index]
    
    def __len__(self):
        return len(self.patch_list)
                    


class PointData(Dataset):
    #dataset for individual point
    def __init__(self,data,args,sampler=None):
        mode = args.mode
        k = args.k
        self.k = int(k)
        self.r = args.r
        self.mode = mode
        source = args.source
        if source == "fpm":
            data_max = np.array([ 5, 5, 10.00022221, 357.19000244, 38.62746811, 48.47133255, 50.60621262])
            data_min = np.array([-5, -5, 0, 0, -5.63886223e+01, -3.69567909e+01, -7.22953186e+01])
            # mean = [0, 0, 5, 23.9, 0, 0, 0.034]
            # std = [2.68, 2.68, 3.09, 55.08, 0.3246, 0.3233, 0.6973]
        elif source == "cos":
            data_min=  np.array([0,0,0, -2466, -2761, -2589, -17135.6, -20040, -20096, -6928022])
            data_max=  np.array([6.2500008e+01, 6.2500000e+01, 6.2500000e+01, 2.7808181e+03, 2.9791230e+03, 2.6991892e+03, 1.9324572e+04, 2.0033873e+04, 1.7973633e+04, 6.3844562e+05])
            # mean = [30.4, 32.8, 32.58, 0, 0, 0, 0, 0, 0, -732720]
            # std = [18.767, 16.76, 17.62, 197.9, 247.2, 193.54, 420.92, 429, 422.3, 888474]
        elif source == 'eth':
            data_min = np.array([ 0., 0., 0., -4.4875424, -1.914121])
            data_max = np.array([1., 1., 1., 4.22590344, 1.70973541])
        data = (data - data_min) / (data_max-data_min)
        data = np.clip(data,0,1)
        coord = data[:,:3]
        kd = cKDTree(coord,leafsize=100)

        if sampler is None:
            # sample the data according to args.sample_size
            #balanced sample of attributes
            attr = data[:,3:]
            leaf_size = len(data)//args.sample_size
            # for some input files, enforce balanced tree will cause bug.
            attr_kd = cKDTree(attr,leaf_size,balanced_tree=False)
            leaf = all_leaf_nodes(attr_kd.tree)
            rand = np.random.rand((len(leaf)))
            idx = []
            for i,l in enumerate(leaf):
                indices = l.indices
                idx.append(indices[int(len(indices)*rand[i])])
            sample_id = idx
            self.center = data[sample_id,:3]
        else:
            sampler = np.array(sampler)
            if len(sampler.shape) == 1:
                # sample index
                sample_id = sampler
                self.center = data[sample_id,:3]
            else:
                # sample center
                sampler = sampler/10
                sampler[:,0] += 0.5
                sampler[:,1] += 0.5
                self.center = sampler
        
        if mode == "ball":
            self.nn = kd.query_ball_point(self.center,self.r,workers=8)
        elif mode == "knn":
            _, self.nn = kd.query(self.center,self.k,workers=8)
        self.data = data

    def __getitem__(self, index):
        nn_id = self.nn[index]
        center = self.center[index]
        if self.mode == "ball":
            #cutting
            if len(nn_id) >= self.k:
                nn_id = nn_id[:self.k]
                pc_length = self.k
            #point cloud and center point
            pc = self.data[nn_id]
            pc[:,:3] -= center[:3]
            #padding
            if len(nn_id) < self.k:
                dim = pc.shape[1]
                remaining = np.zeros((self.k-len(nn_id),dim))
                pc = np.concatenate((pc,remaining),axis=0)
                pc_length = len(nn_id)
            return pc, pc_length
        elif self.mode =="knn":
            pc = self.data[nn_id]
            pc[:,:3] -= center[:3]
            return pc
    def __len__(self):
        return len(self.nn)

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

def data_reader(filename, type):
    if type == 'cos':
        return sdf_reader(filename)
    elif type == 'fpm':
        return vtk_reader(filename)
    elif type == 'eth':
        return eth_reader(filename)

def vtk_reader(filename):
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
    width = particles.parameters['L0']
    cosmo_a = particles.parameters['a']
    kpc_to_Mpc = 1./1000
    convert_to_cMpc = lambda proper: (proper ) * h_100 * kpc_to_Mpc / cosmo_a + 31.25
    numpy_data = np.array(list(particles.values())[2:-1]).T # coord, velocity, acceleration, phi
    numpy_data[:,:3] = convert_to_cMpc(numpy_data[:,:3])
    return numpy_data

def eth_reader(filename):
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(filename)
    reader.Update()
    vtk_data = reader.GetOutput()
    _,x,_,y,_,z = vtk_data.GetBounds()
    nx=int(x+1)
    ny=int(y+1)
    nz=int(z+1)
    log_Rho = numpy_support.vtk_to_numpy(vtk_data.GetPointData().GetArray(0))
    log_s = numpy_support.vtk_to_numpy(vtk_data.GetPointData().GetArray(1))
    x = np.linspace(0,1,nx)
    y = np.linspace(0,1,ny)
    z = np.linspace(0,1,nz)
    z, y, x = np.meshgrid(x,y,z,indexing='xy')
    data = np.stack((x.flatten(),y.flatten(),z.flatten(),log_Rho,log_s),axis=-1)
    return data 

    

    
def normalize(data,mean,std):
    return (data - mean)/std

def all_leaf_nodes(node):
    if node.greater==None and node.lesser==None:
        return [node]
    else:
        return(all_leaf_nodes(node.lesser)+all_leaf_nodes(node.greater))

def collect_file(directory,mode,shuffle=True):
    file_list = []
    for (dirpath, dirnames, filenames) in os.walk(directory):
        for filename in filenames:
            if mode == "fpm":
                if filename.endswith(".vtu") and filename != "000.vtu":
                    inp = os.sep.join([dirpath, filename])
                    file_list.append(inp)
            if mode == "cos":
                if "ds14" in filename.split('_'):
                    inp = os.sep.join([dirpath, filename])
                    file_list.append(inp)
    if shuffle:
        random.shuffle(file_list)
    return file_list
    
def min_max(file_list,mode="fpm"):
    all_min = None
    all_max = None
    for i,f in enumerate(file_list):
        print("processing file {}/{}".format(i,len(file_list)),end='\r')
        data = vtk_reader(f) if mode == "fpm" else sdf_reader(f)
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
        data = vtk_reader(f) if mode == "fpm" else sdf_reader(f)
        if i==0:
            total = np.zeros((data.shape[1]))
            length = 0
        data = (data - mean) ** 2
        total += np.sum(data,axis = 0)
        length += len(data)
    std = np.sqrt(total/length)
    print("std: ", std)
    return std

def numpy_to_vtk(position:np.array,array_dict:dict):
    vtk_position = numpy_support.numpy_to_vtk(position)
    points = vtk.vtkPoints()
    points.SetData(vtk_position)
    data_save = vtk.vtkUnstructuredGrid()
    data_save.SetPoints(points)
    pd = data_save.GetPointData()
    for k, v in array_dict.items():
        vtk_array = numpy_support.numpy_to_vtk(v)
        vtk_array.SetName(k)
        pd.AddArray(vtk_array)
    return data_save

def vtk_write(data_save,filename:str):
    writer = vtk.vtkXMLDataSetWriter()
    writer.SetFileName(filename)
    writer.SetInputData(data_save)
    writer.Write()

def vtk_write_image(x_dim,y_dim,z_dim,data,filename):
    # data flattened in xyz order
    assert len(data) == x_dim * y_dim * z_dim
    imageData = vtk.vtkImageData()
    imageData.SetDimensions(x_dim,y_dim,z_dim)
    imageData.AllocateScalars(vtk.VTK_DOUBLE, 1)

    dims = imageData.GetDimensions()
    vtk_data = numpy_support.numpy_to_vtk(data)
    imageData.GetPointData().AddArray(vtk_data)

    # i = 0
    # for z in range(dims[2]):
    #     for y in range(dims[1]):
    #         for x in range(dims[0]):
    #             imageData.SetScalarComponentFromDouble(z, y, x, 0, data[i])
    #             i+=1


    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(imageData)
    writer.Write()

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

if __name__ == "__main__":
    eth_reader('F:\\OneDrive - The Ohio State University\\data/ethanediol.vti')