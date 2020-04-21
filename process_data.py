import random
import os
import sys
import math
from datetime import datetime

from vtk import *
from vtk.util import numpy_support
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
from scipy.spatial import KDTree
import torch


class Generator():
    """
    generate dataset from the raw data directory
    """
    def __init__(self,directory,mean=[2.39460057e+01, -4.29336209e-03, 9.68809421e-04, 3.44706680e-02],std=[55.08245731,  0.32457581,  0.32332313,  0.6972805]):
        """
        generate random file list
        """
        file_list = []
        for (dirpath, dirnames, filenames) in os.walk(directory):
            for filename in filenames:
                if filename.endswith('.vtu'):
                    inp = os.sep.join([dirpath, filename])
                    file_list.append(inp)
        random.shuffle(file_list)
        self.file_list = file_list
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.file_list)

    def mean(self):
        length = 0
        total = np.zeros((4,))
        i = 0
        for filename in self.file_list:
            data = data_reader(filename)
            data = data_to_numpy(data)
            total += np.sum(data[:,3:],axis = 0)
            length += len(data)
            i+=1
            print("{}/{}".format(i,len(self)),end='\r')
        print("mean: ",total/length)

    def std(self,mean=[2.39460057e+01, -4.29336209e-03, 9.68809421e-04, 3.44706680e-02]):
        length = 0
        total = np.zeros((4,))
        i = 0
        for filename in self.file_list:
            data = data_reader(filename)
            data = data_to_numpy(data)
            data = data[:,3:]
            data = (data - mean) ** 2
            total += np.sum(data,axis = 0)
            length += len(data)
            i+=1
            print("{}/{}".format(i,len(self)),end='\r')
        print("std: ",np.sqrt(total/length))

    def generate_new(self,filename):
        data = []
        for i,file in enumerate(self.file_list):
            batch = self._sample_one_file(file)
            data += batch
            print("{}/{}".format(i+1,len(self.file_list)),end='\r')
            # print()
            # print(len(data))
        data = np.array(data)
        np.save(filename,data)
        print()
        print("Process complete, {} samples generated".format(len(data)))
        
    def _sample_one_file(self,filename,cut_level=4,num_sample=1, r=0.7):
        data = data_reader(filename)
        data = data_to_numpy(data)
        rmin = np.min(data,axis = 0)
        rmax = np.max(data,axis = 0)
        coord = data[:,:3]
        attr = data[:,3:]
        attr = (attr - self.mean)/self.std
        attr_dim = attr.shape[1]

        sys.setrecursionlimit(10000)
        attr_kd = KDTree(attr,10000)
        nodes = nodes_at_level(attr_kd.tree,cut_level)
        sample = []
        # print("number_of_sample:",len(nodes))
        for node in nodes:
            children = collect_children(node)
            node_sample = []
            s = np.random.permutation(children)
            for ss in s:
                c = coord[ss]
                # check c not in the padding area
                if c[2] > r and c[2]<(10-r) and (c[0]**2 + c[1] **2) < (5-r) ** 2:
                    node_sample.append(ss)
                    if len(node_sample) == num_sample:
                        sample = sample + node_sample
                        break
        sample = np.array(sample).reshape(-1)
        coord_kd = KDTree(coord)
        
        ball = coord_kd.query_ball_point(coord[sample],r=r)
        batch = []
        for b in ball:
            a = attr[b]
            c = coord[b]
            batch.append(np.concatenate((c,a),axis=-1))
        return batch

            # scatter_3d(np.concatenate((c,a),axis=-1),vmin=-2,vmax=2)
        # for i in range(len(ball)):
        #     points = point_data[ball[i]]
        #     points = points - center[i]
        #     points[:,:3] /= r #normalize dim0,1,2 to [-1,1]
        #     batch.append(points)

        
        # sign = [0,0,0,0]
        # acond = np.full((attr.shape[1],split,attr.shape[0]),True)
        # for d in range(attr.shape[1]):
        #     dmax = rmax[3+d]
        #     dmin = rmin[3+d]
        #     for s in range(split):
        #         start = s*(dmax-dmin)/split + dmin
        #         end = (s+1)*(dmax-dmin)/split + dmin
        #         acond[d,s] = (attr[:,d] >= start) & (attr[:,d] <= end)
        # total_l = 0
        # for s in range(split ** attr.shape[1]):
        #     for d in range(attr.shape[1]):
        #         sign[d] = s % split
        #         s //= split
        #     cond = np.full((attr.shape[0]),True)
        #     for j in range(len(sign)):
        #         cond = cond & acond[j,sign[j]]

            # print(len(data[cond]))

        # print(total_l,len(data))
                

        # for d in range(attr.shape[1]):
        #     print(d)
        #     for s in range(split):
        #         print(s)
        #         start = s*(dmax-dmin)/split + dmin
        #         end = (s+1)*(dmax-dmin)/split + dmin
        #         acond = (attr[:,d] > start) & (attr[:,d] < end)
        #         print(len(data[acond & ccond]))


    def generate(self,filename, n_sample=60000 ,k = 128, r=0.5, dim = 4, mode='ball'):
        """
        generate dataset
        """
        point_per_file = n_sample // len(self.file_list)
        overflow = n_sample - point_per_file * len(self.file_list)
        
        file_id = 0
        batch = []
        
        for file in self.file_list:
            data = data_reader(file)
            num_points = data.GetNumberOfPoints()
            if file_id < overflow:
                idx = np.random.rand(point_per_file+1)
            else:
                idx = np.random.rand(point_per_file)
            idx = (idx * num_points).astype(np.int)
            batch+=list(sample_around(idx,data,k,r,dim,mode))
            print("{}/{} processed".format(file_id+1,len(self.file_list)),end="\r")

            file_id += 1

        batch = np.array(batch)
        np.save(filename,batch)
        print()
        print("Process complete, {} samples generated".format(len(batch)))

class Loader():
    """
    load from npy file
    """
    def __init__(self,filename, batch_size):
        self.data = np.load(filename, allow_pickle=True)
        # print(len(self.data))
        self.batch_size =batch_size

    def __len__(self):
        return math.ceil(len(self.data) / self.batch_size)

    def load_data(self,start=0,end=None):
        data = self.data[start:end]
        batch_size = self.batch_size
        for i in range(0,len(data),batch_size):
            yield data[i:i+batch_size]


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

def prepare_for_model(data: list,device,coord_dim=3,kept_dim=4,mean=[2.39460057e+01, -4.29336209e-03, 9.68809421e-04, 3.44706680e-02],std=[55.08245731,  0.32457581,  0.32332313,  0.6972805]):
    tensor_list = []
    mean = mean[:kept_dim-coord_dim]
    std = std[:kept_dim-coord_dim]
    for datum in data:
        numpy_datum = datum[:,:kept_dim]
        numpy_datum[:,coord_dim:] = (numpy_datum[:,coord_dim:] - mean)/std
        numpy_datum[:,:coord_dim] = mean_sub(numpy_datum[:,:coord_dim],coord_dim)
        tensor = torch.from_numpy(numpy_datum).float().to(device)
        tensor_list.append(tensor)
    return tensor_list

def to_tensor_list(data: list,device,kept_dim=4):
    tensor_list = []
    for datum in data:
        numpy_datum = datum[:,:kept_dim]
        tensor = torch.from_numpy(numpy_datum).float().to(device)
        tensor_list.append(tensor)
    return tensor_list

def image_to_pointcloud(img, point_size = 1024):
    ndim = len(img.shape)-1
    points = np.zeros((point_size,ndim))
    img = img.numpy()[:,0,:,:]
    rand_points = np.random.rand(point_size*10,2)
    rand_points[:,0] = rand_points[:,0] * img.shape[-2]
    rand_points[:,1] = rand_points[:,1] * img.shape[-1]
    prob = np.random.rand(img.shape[0],point_size*10)
    point_prob = img[:,rand_points[:,0].astype(np.int),rand_points[:,1].astype(np.int)]
    points = []
    for i in range(img.shape[0]):
        lab = (point_prob>prob)[i]
        points.append(rand_points[lab])
    return points


def data_reader(filename):
    reader = vtkXMLUnstructuredGridReader()
    reader.SetFileName(filename)
    reader.Update()
    data = reader.GetOutput()
    return data

def sample_around(idx, data,k = 128, r=0.5, dim = 4, mode='ball'):
    """
    sample around idx(num_center * point_dim) in data with parameters
    data: vtk data
    dimensions 0,1,2 are xyz coordinates
    dimension 3 is concentration value max = 357.19, min = 0.0 (normalized)
    dimensions 4,5,6 are velocity value
    """
    coord = numpy_support.vtk_to_numpy(data.GetPoints().GetData())
    if dim == 3:
        point_data = coord
    else:
        concen = numpy_support.vtk_to_numpy(data.GetPointData().GetArray(0))[:,None]
        concen = (concen/357.19)*2-1 #normalize concentration value t0 [-1,1]
        if dim == 4:
            point_data = np.concatenate((coord,concen),axis=-1)
        else:
            velocity = numpy_support.vtk_to_numpy(data.GetPointData().GetArray(1))
            if dim == 7:
                point_data = np.concatenate((coord,concen,velocity),axis=-1)
            else:
                raise ValueError

    center=np.expand_dims(point_data[idx],1)
    center[:,:,3:] = 0

    sys.setrecursionlimit(10000)
    kd = KDTree(coord,leafsize=100)
    if mode == 'knn':
        # no normalization yet...
        knn = kd.query(coord[idx],k=k)
        points = point_data[knn[1]]
        points = points - center
        return points
    elif mode == 'ball':
        ball = kd.query_ball_point(coord[idx],r=r)
        batch = []
        for i in range(len(ball)):
            points = point_data[ball[i]]
            points = points - center[i]
            points[:,:3] /= r #normalize dim0,1,2 to [-1,1]
            batch.append(points)
        return batch

def sample_all_from_file(filename,k = 128, r=0.5, dim = 4, mode='ball'):
    data = data_reader(filename)
    num_points = data.GetNumberOfPoints()

    idx = np.arange(num_points)
    sample = sample_around(idx,data,k,r,dim,mode)

    return sample
    
def data_to_numpy(vtk_data):
    coord = numpy_support.vtk_to_numpy(vtk_data.GetPoints().GetData())
    concen = numpy_support.vtk_to_numpy(vtk_data.GetPointData().GetArray(0))[:,None]
    velocity = numpy_support.vtk_to_numpy(vtk_data.GetPointData().GetArray(1))
    point_data = np.concatenate((coord,concen,velocity),axis=-1)
    return point_data


def nodes_at_level(root,level):
    # finding all nodes on the level
    current_node = [root]
    next_node = []
    for i in range(level):
        for node in current_node:
            if isinstance(node,KDTree.innernode):
                next_node.append(node.less)
                next_node.append(node.greater)
            else:
                next_node.append(node)
        current_node =next_node
        next_node = []
    return current_node

def all_leaf_nodes(node):
    if isinstance(node,KDTree.leafnode):
        return [node]
    else:
        return(all_leaf_nodes(node.less)+all_leaf_nodes(node.greater))

def collect_children(node):
    if isinstance(node,KDTree.leafnode):
        return node.idx
    elif isinstance(node,KDTree.innernode):
        return np.concatenate((collect_children(node.less),collect_children(node.greater)))

def mean_sub(data,dim=3):
    mean = np.mean(data[:,:dim],axis = 0)
    data[:,:dim] -= mean
    return data

if __name__ == "__main__":
    # generate data
    try:
        data_dir = os.environ['data']
        data_dir = data_dir + "\\2016_scivis_fpm\\0.44"
    except KeyError:
        data_dir = "../data/0.44"
    # print(data_reader(data_dir+"/run01/010.vtu"))
    # generator = Generator(data_dir)
    # generator.generate_new("data_sample")
    # t1 = datetime.now()

    # generator._sample_one_file(generator.file_list[1])

    # t2 = datetime.now()
    # print(t2-t1)
    # dir = r'C:\Users\aide0\OneDrive - The Ohio State University\data\2016_scivis_fpm\0.44\run34\002.vtu'
    # data = data_reader(dir)
    # data = data_to_numpy(data)
    # coord = data[:,:3]
    # attr = data[:,3:]
    # print(len(coord))
    # # attr = (attr - self.mean)/self.std
    # # attr_dim = attr.shape[1]

    # # sys.setrecursionlimit(10000000)
    # attr_kd = KDTree(attr,30000)
    # print(attr_kd.tree)
    # children = all_leaf_nodes(attr_kd.tree)
    # for child in children:
    #     print(len(collect_children(child)))
    # # nodes = nodes_at_level(attr_kd.tree,4)
    # # print("number_of_sample:",len(nodes))
    # # print(np.min(attr[:,1]),np.max(attr[:,1]))
    # # nodes = nodes_at_level(attr_kd.tree,cut_level)
    # data = np.load("data/new_sample.npy", allow_pickle=True)
    # print(data[0][0])
    # i = 0
    # for datum in data:
    #     center = np.mean(datum[:,:3],axis=0)
    #     datum[:,:3] -= center
    #     i+=1
    #     print("{}/{}".format(i,len(data)),end='\r')
    # np.save("new_sample",data)
    