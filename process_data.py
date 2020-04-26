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
from yt.utilities.sdf import SDFRead


class Generator():
    """
    generate dataset from the raw data directory
    """
    def __init__(self,directory):
        """
        generate random file list
        """
        file_list = []
        for (dirpath, dirnames, filenames) in os.walk(directory):
            for filename in filenames:
                print()
                if "ds14" in filename.split('_'):
                    inp = os.sep.join([dirpath, filename])
                    file_list.append(inp)
        random.shuffle(file_list)
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def calc_mean(self):
        length = 0
        total = np.zeros((4,))
        i = 0
        max = np.zeros((10,))
        min = np.full((10,),1000)
        for filename in self.file_list:
            data = data_reader(filename)
            data = data_to_numpy(data)
            batch_max = np.max(data,axis = 0)
            batch_min = np.min(data,axis = 0)
            max = np.where(batch_max>max,batch_max,max)
            min = np.where(batch_min<min,batch_min,min)
            i+=1
            print("{}/{}".format(i,len(self)),end='\r')
        print("max: ",max)
        print("min: ",min)

    def generate_new(self,filename):
        data = []
        for i,file in enumerate(self.file_list):
            batch = self._sample_one_file(file)
            data += batch
            # print(len(data),data[0].shape)
            print("{}/{}".format(i+1,len(self.file_list)),end='\r')
            # print()
            # print(len(data))
        data = np.array(data)
        np.save(filename,data)
        print()
        print("Process complete, {} samples generated".format(len(data)))
        
    def _sample_one_file(self,filename,cut_level=4,num_sample=50, r=0.6):
        data = data_reader(filename)
        data = data_to_numpy(data)
        data = normalize(data)
        rmin = np.min(data,axis = 0)
        rmax = np.max(data,axis = 0)
        coord = data[:,:3]
        attr = data[:,3:]
        attr_dim = attr.shape[1]

        sys.setrecursionlimit(10000)
        attr_kd = KDTree(attr,1000)
        nodes = nodes_at_level(attr_kd.tree,cut_level)
        sample = []
        # print("number_of_sample:",len(nodes))
        for node in nodes:
            children = collect_children(node)
            node_sample = []
            s = np.random.permutation(children)

            # sample = sample +list(s[:num_sample])

            for ss in s:
                c = coord[ss]
                # check c not in the padding area
                if c[0] > r and c[0]<(62.5-r) and c[1] > r and c[1]<(62.5-r) and c[2] > r and c[2]<(62.5-r):
                    node_sample.append(ss)
                    if len(node_sample) == num_sample:
                        sample = sample + node_sample
                        break
        sample = np.array(sample).reshape(-1)
        coord_kd = KDTree(coord,1000)
        # print(len(sample))

        # knn = coord_kd.query(coord[sample],r)
        # knn = list(data[knn[1]])
        
        ball = coord_kd.query_ball_point(coord[sample],r=r)
        batch = []
        i=0
        for b in ball:
            a = attr[b]
            c = coord[b]
            c -= coord[sample[i]]
            batch.append(np.concatenate((c,a),axis=-1))
            i+=1
        return batch

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

def normalize(data,coord_dim=3,kept_dim=4,mean=[ -2.46606519e+03, -2.76064209e+03, -2.58863428e+03, -1.71356348e+04, -2.00398145e+04, -2.00960469e+04, -6.92802200e+06] \
    ,std=[ 2.78081812e+03, 2.97912305e+03, 2.69918921e+03, 1.93245723e+04, 2.00338730e+04, 1.79736328e+04, 6.38445625e+05]):
    #treat std as max, mean as min
    mean = np.array(mean)
    std = np.array(std)
    tensor_list = []
    assert len(mean) == kept_dim-coord_dim
    data[:,coord_dim:kept_dim] = (data[:,coord_dim:kept_dim] - mean)/(std-mean)
    return data

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
    data = SDFRead(filename)
    return data


    
def data_to_numpy(particles):
    h_100 = particles.parameters['h_100']
    width = particles.parameters['L0']
    cosmo_a = particles.parameters['a']
    kpc_to_Mpc = 1./1000

    numpy_data = np.array(list(particles.values())[2:-1]).T

    convert_to_cMpc = h_100 * kpc_to_Mpc / cosmo_a 
    numpy_data[:,:3] = numpy_data[:,:3] * convert_to_cMpc + 31.25
    return numpy_data


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
    data_env = os.environ['data']
    data_dir = data_env + "\\ds14_scivis_0128\\raw"
    # except KeyError:
    #     data_dir = "../data/0.44"
    generator = Generator(data_dir)
    generator.generate_new(data_env+"\cosmology_ball_more")
    # t1 = datetime.now()
    # data = np.load(data_env+"/datanew.npy",allow_pickle=True)
    # print(data[0][3])
    # for d in data:
    #     # d[:,:3] /= 62.5
    #     mean_position = np.mean(d[:,:3],axis=0)
    #     d[:,:3] -= mean_position
    # np.save(data_env+"/new",data)
    

    # generator._sample_one_file(generator.file_list[1])

    # t2 = datetime.now()
    # print(t2-t1)
    # dir = r'C:\Users\aide0\OneDrive - The Ohio State University\data\2016_scivis_fpm\0.44\run34\002.vtu'
    # data = data_reader(dir)
    # data = data_to_numpy(data)
    # coord = data[:,:3]
    # attr = data[:,3:]
    # print(len(coord))
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
    