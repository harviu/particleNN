import cv2
import os
from datetime import datetime
import numpy as np
from matplotlib import pyplot as plt
import pickle

from process_data import data_reader,data_to_numpy,scatter_3d
from scipy.spatial import KDTree
from sklearn.decomposition import PCA

MAX = 360
eps = 1e-30
# def weighted_hist(center,h,coord,samples,bins,ranges):
#     weights = np.sum(((coord-center)/h)**2,axis=-1)
#     return np.histogramdd(samples,bins,range=ranges,weights = weights,density = True)[0]


# def calc_weights(samples,target,hist):
#     pass


class data_frame():
    def __init__(self,data,n_channel,center,h,bins=None,ranges=None):
        self.coord = data[:,:n_channel]
        self.attr = data[:,n_channel:]
        self.kd = KDTree(self.coord)
        self.bins = bins
        self.h = h
        self.center = center
        self.ranges = ranges
        self.update()

    def update(self):
        """
        update the hist to new center
        """
        near = self.kd.query_ball_point(self.center,self.h)
        self.near_coord = self.coord[near]
        self.near_attr = self.attr[near]
        self.hist = None
        if self.ranges is None:
            rmax = np.max(self.near_attr,axis=0)
            rmin = np.min(self.near_attr,axis=0)
            ranges = np.stack((rmin,rmax),axis=-1)
        else:
            ranges = self.ranges
        self._hist()

    def _hist(self):
        weights = 1 - np.sum(((self.near_coord-self.center)/self.h)**2,axis=-1)
        hist = np.histogramdd(
            self.near_attr,
            bins = self.bins,
            range = self.ranges,
            weights = weights,
            density = True)
        self.hist = hist[0]
        

class mean_shift():
    def __init__(self,target,data,ite=None,dis=None):
        """
        data: data_frame object to search
        target: target histgram
        """
        self.target = target
        self.data = data
        self.ite = ite
        self.dis = dis

    def next_center(self):
        data = self.data
        target = self.target
        
        ### adaptive range 
        data_rmax = np.max(data.near_attr,axis=0)
        data_rmin = np.min(data.near_attr,axis=0)
        target_rmax = np.max(target.near_attr,axis=0)
        target_rmin = np.min(target.near_attr,axis=0)
        new_rmin = np.where(data_rmin<target_rmin,data_rmin,target_rmin)
        new_rmax = np.where(data_rmax>target_rmax,data_rmax,target_rmax)
        new_ranges = np.stack((new_rmin,new_rmax),axis=-1)
        # new_ranges = ((0,MAX),(-1,1),(-1,1),(-1,1))
        data.ranges = new_ranges
        target.ranges = new_ranges
        data.update()
        target.update()

        weights = np.sqrt(target.hist/(data.hist+eps))
        # print(data.hist[0,:])
        # print(target.hist)
        # print(data.hist)
        # os.system("pause")
        near_bins = self._get_bins(data.near_attr,data.ranges,data.bins)
        
        new_center = np.zeros((len(data.center),))
        w_sum = eps
        near_w = np.zeros((len(near_bins)))
        for i in range(len(near_bins)):
            b = near_bins[i]
            w = weights[tuple(b)]
            new_center += w * data.near_coord[i]
            w_sum += w
            near_w[i]=w

        new_center /= w_sum
        return new_center

    def _get_bins(self,samples, ranges, bins):
        n_dims = samples.shape[1]
        sample_bins = np.zeros_like(samples,dtype=np.int)
        for d in range(n_dims):
            if ranges is None:
                max = np.max(samples[:,d])
                min = np.min(samples[:,d])
            else:
                max = ranges[d][1]
                min = ranges[d][0]
            if isinstance(bins,int):
                N=bins
            else:
                N = bins[d]
            step = (max-min) / N
            sample_bins[:,d] = (samples[:,d]-min) // step
            # make bin = N to bin = N-1
            idx = np.where(sample_bins[:,d] == N)
            sample_bins[:,d][idx] = N-1
        return sample_bins

    def shift(self):
        if self.ite == None and self.dis == None:
            mode = "ite"
            self.ite = 10
        elif self.ite != None:
            mode = "ite"
        else:
            mode = "dis"
        i = 0
        while(True):
            # print(i)
            center = self.data.center
            # print("hsit", hist)
            # print("target",self.target)
            next_center = self.next_center()
            self.data.center = (next_center+self.data.center)/2
            self.data.update()
            # print("new_hist",self.data.hist)
            # print(hist_similarity(hist,self.target))
            i+=1
            if mode == "ite":
                if i == self.ite:
                    break
            else:
                if np.sqrt(np.sum((center-next_center)**2))<self.dis:
                    break
        print("next_center",self.data.center)
        return self.data

def hist_similarity(h1,h2):
    return np.sum(np.sqrt(h1*h2))

def nn_distance(pc1,pc2):
    pc1 = pc1[:,None,:]
    pc2 = pc2[None,:,:]
    d = (pc1 - pc2)**2
    d = np.sum(d,axis=-1)

    d1 = np.min(d,axis=-1)
    d2 = np.min(d,axis=-2)
    # print(d1.shape,d2.shape)
    # d1 = np.sum(d1,axis=-1)
    # d2 = np.sum(d2,axis=-1)
    # dis = (d1+d2)/2
    dis = np.concatenate((d1,d2))
    dis = np.mean(dis)
    return dis

if __name__ == "__main__":
    data_dir = os.environ["data"]
    ############ work on artificial data ############

    # data0 = np.random.rand(100000,3)
    # data0[:,2] = np.sqrt((data0[:,0]-0.5) **2 + (data0[:,1]-0.5)**2)
    # # print(data0[:,2])
    # tar = data_frame(data0,2,(0.5,0.5),0.05,bins=20)
    # pc1 = np.concatenate((tar.near_coord,tar.near_attr),axis = 1)
    # plt.scatter(pc1[:,0],pc1[:,1],c=pc1[:,2])
    # plt.show()
    
    # data1 = np.random.rand(100000,3)
    # data1[:,2] = np.sqrt((data1[:,0]-0.3) **2 + (data1[:,1]-0.3)**2)
    # aim = data_frame(data1,2,(0.5,0.5),0.05,bins=20)
    # pc2 = np.concatenate((aim.near_coord,aim.near_attr),axis = 1)
    # plt.scatter(pc2[:,0],pc2[:,1],c=pc2[:,2])
    # plt.show()

    # ms = mean_shift(tar,aim,ite=50)
    # ms.shift()
    # pc3 = np.concatenate((aim.near_coord,aim.near_attr),axis = 1)
    # plt.scatter(pc3[:,0],pc3[:,1],c=pc3[:,2])
    # plt.show()

    # pc1[:,0] -= np.mean(pc1[:,0])
    # pc1[:,1] -= np.mean(pc1[:,1])
    # # pc1[:,2] -= np.mean(pc1[:,2])
    # pc2[:,0] -= np.mean(pc2[:,0])
    # pc2[:,1] -= np.mean(pc2[:,1])
    # # pc2[:,2] -= np.mean(pc2[:,2])
    # pc3[:,0] -= np.mean(pc3[:,0])
    # pc3[:,1] -= np.mean(pc3[:,1])
    # # pc3[:,2] -= np.mean(pc3[:,2])

    # print("original distance:",nn_distance(pc1,pc2))
    # print("after meanshift:",nn_distance(pc1,pc3))
    
    #################### latent

    data = data_reader(data_dir+r"\2016_scivis_fpm\0.44\run41\024.vtu")
    data = data_to_numpy(data)
    data = data[:,:4]
    
    with open("data/latent_024","rb") as file:
        d = pickle.load(file).cpu()
    
    with open("data/latent_025","rb") as file:
        d2 = pickle.load(file).cpu()

    pca = PCA(n_components=1)
    dd = np.concatenate((d,d2),axis=0)
    pca.fit(d)
    d_embedded = pca.transform(d)
    d_embedded2 = pca.transform(d2)

    center = (1.5,-1,6.25)
    new_data = np.concatenate((data,d_embedded),axis=1)
    model = data_frame(new_data,3,center,1,bins=12)
    pc1 = np.concatenate((model.near_coord,model.near_attr),axis = 1)
    pc1[:,0] -= np.mean(pc1[:,0])
    pc1[:,1] -= np.mean(pc1[:,1])
    pc1[:,2] -= np.mean(pc1[:,2])
    # scatter_3d(pc1)

    data2 = data_reader(data_dir+r"\2016_scivis_fpm\0.44\run41\025.vtu")
    data2 = data_to_numpy(data2)
    data2 = data2[:,:4]

    # center2 = (1.2,-1.3,5.95)
    new_aim = np.concatenate((data2,d_embedded2),axis=1)
    target = data_frame(new_aim,3,center,1,bins=12)
    pc2 = np.concatenate((target.near_coord,target.near_attr),axis = 1)
    pc2[:,0] -= np.mean(pc2[:,0])
    pc2[:,1] -= np.mean(pc2[:,1])
    pc2[:,2] -= np.mean(pc2[:,2])
    # scatter_3d(pc2)

    ms = mean_shift(model,target,ite=100)
    ms.shift()
    pc3 = np.concatenate((target.near_coord,target.near_attr),axis = 1)
    pc3[:,0] -= np.mean(pc3[:,0])
    pc3[:,1] -= np.mean(pc3[:,1])
    pc3[:,2] -= np.mean(pc3[:,2])
    # scatter_3d(pc3)

    print(pc1.shape)
    print("original distance:",nn_distance(pc1[:,:4],pc2[:,:4]))
    print("after meanshift:",nn_distance(pc1[:,:4],pc3[:,:4]))

    ############
    # t1 = datetime.now()

    # center = (1.5,-1,6.25)
    # di1 = data_dir+"\\2016_scivis_fpm\\0.44\\run41\\024.vtu"
    # di2 = data_dir+"\\2016_scivis_fpm\\0.44\\run41\\025.vtu"

    # data = data_reader(di1)
    # data = data_to_numpy(data)
    # data = data[:,:4]

    # data2 = data_reader(di2)
    # data2 = data_to_numpy(data2)
    # data2 = data2[:,:4]

    # model = data_frame(data,3,center,1,bins=12)
    # pc1 = np.concatenate((model.near_coord,model.near_attr),axis = 1)
    # pc1[:,0] -= np.mean(pc1[:,0])
    # pc1[:,1] -= np.mean(pc1[:,1])
    # pc1[:,2] -= np.mean(pc1[:,2])
    # # scatter_3d(pc1)

    # # center2 = (1.2,-1.3,5.95)

    # target = data_frame(data2,3,center,1,bins=12)
    # pc2 = np.concatenate((target.near_coord,target.near_attr),axis = 1)
    # pc2[:,0] -= np.mean(pc2[:,0])
    # pc2[:,1] -= np.mean(pc2[:,1])
    # pc2[:,2] -= np.mean(pc2[:,2])
    # # scatter_3d(pc2)

    # ms = mean_shift(model,target,ite=100)
    # ms.shift()
    # pc3 = np.concatenate((target.near_coord,target.near_attr),axis = 1)
    # pc3[:,0] -= np.mean(pc3[:,0])
    # pc3[:,1] -= np.mean(pc3[:,1])
    # pc3[:,2] -= np.mean(pc3[:,2])
    # # scatter_3d(pc3)

    # center = target.center

    # print(pc1.shape)
    # print("original distance:",nn_distance(pc1,pc2))
    # print("after meanshift:",nn_distance(pc1,pc3))

    # t2 = datetime.now()
    # print(t2-t1)
    # ###############
