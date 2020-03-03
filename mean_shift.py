import os
from datetime import datetime
import numpy as np
from matplotlib import pyplot as plt
import pickle

from scipy.spatial import KDTree
from sklearn.decomposition import PCA

from process_data import *
from latent_max import LatentMax

MAX = 360
eps = 1e-10

class data_frame():
    def __init__(self,data,n_channel,center,h,bins=None,ranges=None):
        self.data = data
        self.coord = data[:,:n_channel]
        self.attr = data[:,n_channel:]
        self.kd = KDTree(self.coord,1000)
        self.bins = bins
        self.h = h
        self.coord_dim = n_channel
        self.set_center(center)
        self.set_range(ranges)
        self.update_hist()

    def set_range(self,ranges):
        self.ranges = ranges
        if self.ranges is None:
            rmax = np.max(self.near_attr,axis=0)
            rmin = np.min(self.near_attr,axis=0)
            self.ranges = np.stack((rmin,rmax),axis=-1)

    def set_center(self,center):
        self.center = center
        self.near = self.kd.query_ball_point(self.center,self.h)
        self.near_coord = self.coord[self.near]
        self.near_attr = self.attr[self.near]
        self.near_pc = self.data[self.near]

    def update_hist(self):
        self.hist = weighted_hist(self.near_coord,self.near_attr, self.center,self.h,self.bins,self.ranges)


    
class latent_df(data_frame):
    def __init__(self,data,n_channel,center,h,bins,ranges,model,device,dim,pca=None):
        self.model = model
        self.device = device
        self.dim = dim
        self.pca = pca

        latent = np.zeros((len(data),2))
        self.data = np.concatenate((data,latent),axis=1)
        self.coord = self.data[:,:n_channel]
        self.attr = self.data[:,n_channel:]
        self.kd = KDTree(self.coord,1000)
        self.bins = bins
        self.h = h
        self.coord_dim = n_channel

        self.set_center(center)
        self.set_range(ranges)
        self.update_hist()

    def set_center(self, center):
        super().set_center(center)
        self.calc_latent(self.model,self.device,self.dim,self.pca)

    def calc_latent(self,model,device,dim,pca):
        """
        calculte the latent vectors of points around center
        """
        t = self.kd.query_ball_point(self.near_coord,0.7)
        x = []
        # print(self.data.shape)
        for tt in t:
            x.append(self.data[tt][:,:-2])
        x = prepare_for_model(x,device,3,dim)
        # print(x[0].shape)
        with torch.no_grad():
            y = model.encode(x)
        latent = y.detach().cpu().numpy()   
        # print(latent.shape) 
        self.data[self.near,-2:] = latent
        self.near_coord = self.coord[self.near]
        self.near_attr = self.attr[self.near]
        self.near_pc = self.data[self.near]


            
    
def get_latent(model,x:np.ndarray,device,dim):
    x = prepare_for_model([x],device,3,dim)
    # scatter_3d(x[0])
    with torch.no_grad():
        y = model.encode(x)
    return y.detach().cpu().numpy()        

def weighted_hist(near_coord,near_attr, center, h,bins,ranges):
    weights = 1 - np.sum(((near_coord-center)/h)**2,axis=-1)
    hist = np.histogramdd(
        near_attr,
        bins = bins,
        range = ranges,
        weights = weights,
        density = True)
    hist[0][hist[0]<0] = 0
    # print(hist[0])
    return hist[0]

class guided_shift():
    def __init__(self, target_pc, init_df: data_frame, guide: LatentMax):
        self.target_pc = target_pc
        self.init_df = init_df
        # guide is a latent optimizer
        self.guide = guide
        # init mean shifter
        self.ms = mean_shift(None,self.init_df,ite=30)
        # how about stopping criteria?

    def shift(self):
        while True:
            # update init point cloud
            init_pc = self.init_df.near_pc
            mean =  np.mean(init_pc[:,:3],axis=0)
            init_pc[:,:3] -= mean[None,:]
            # scatter_3d(init_pc)
            # copy = init_pc.copy()
            # get the next pc through latent optimizer
            # notice the init_pc is changed inplace
            next_pc = self.guide.take_one_step_to_target(init_pc)
            next_pc[:,:3] += mean[None,:]
            # print(next_pc)
            # mean shift to next pc
            self.ms.target = next_pc
            self.ms.shift()
            # when to stop?
            # break
        

class mean_shift():
    def __init__(self,target,data,ite=20,dis=0.01):
        """
        data: data_frame object to search
        target: target point cloud
        """
        self.target = target
        self.data = data
        self.ite = ite
        self.dis = dis

    def next_center(self):

        data = self.data
        target = self.target
        coord_dim = self.data.coord_dim
        
        ### adaptive range 
        data_rmax = np.max(data.near_attr,axis=0)
        data_rmin = np.min(data.near_attr,axis=0)
        target_rmax = np.max(target[:,coord_dim:],axis=0)
        target_rmin = np.min(target[:,coord_dim:],axis=0)
        new_rmin = np.where(data_rmin<target_rmin,data_rmin,target_rmin)
        new_rmax = np.where(data_rmax>target_rmax,data_rmax,target_rmax)
        new_ranges = np.stack((new_rmin,new_rmax),axis=-1)
        # new_ranges = ((0,MAX),(-1,1),(-1,1),(-1,1))
        data.set_range(new_ranges)
        data.update_hist()
        target_center = np.mean(target[:,:coord_dim],axis=0)
        target_hist = weighted_hist(target[:,:coord_dim],target[:,coord_dim:],target_center,data.h,data.bins,new_ranges)

        weights = np.sqrt(target_hist/(data.hist+eps))

        near_bins = self._get_bins(data.near_attr,new_ranges,data.bins)
        
        new_center = np.zeros((len(data.center),))
        w_sum = eps
        near_w = np.zeros((len(near_bins)))
        for i in range(len(near_bins)):
            b = near_bins[i]
            w = weights[tuple(b)]
            new_center += w * data.near_coord[i]
            w_sum += w
            near_w[i]=w
        # print(near_w)

        new_center /= w_sum
        
        print(new_center)
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
        i = 0
        while(True):
            # print(i)
            center = self.data.center
            next_center = self.next_center()
            # t1 = datetime.now()
            self.data.set_center((next_center+self.data.center)/2)
            # self.data.set_center(next_center)
            # self.data.set_range(None)
            self.data.update_hist()
            i+=1
            if i == self.ite or np.sqrt(np.sum((center-next_center)**2))<self.dis:
                break
        print("Mean_shift_next_center",self.data.center)
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
    dis = np.concatenate((d1,d2))
    dis = np.mean(dis)
    return dis

def track_run(path,start,end,init_center,h,bins,model,device,dim,latent=True):
    center = init_center
    center_list = []
    center_list.append(center)
    data = data_to_numpy(data_reader(path+"\{:03d}.vtu".format(start)))
    data = data[:,:dim]
    start_df = latent_df(data,3,center,h,bins,None,model,device,dim)
    # start_df = data_frame(data,3,center,h,bins,None)
    m = start_df.near_pc.copy()
    pc1 = m.copy()
    pc1 = mean_sub(pc1)
    # print(center)
    # scatter_3d(pc1,center=center)
    for i in range(start,end):
        data = data_to_numpy(data_reader(path+"\{:03d}.vtu".format(i)))
        data = data[:,:dim]
        scatter_3d(data,50,350,threshold=50,center=center)

        data_next = data_to_numpy(data_reader(path+"\{:03d}.vtu".format(i+1)))
        data_next = data_next[:,:dim]

        if latent:
            start_df = latent_df(data,3,center,h,bins,None,model,device,dim)
            target = latent_df(data_next,3,center,h,bins,None,model,device,dim)
        else:
            start_df = data_frame(data,3,center,h,bins,None)
            target = data_frame(data_next,3,center,h,bins,None)
        
        # m = start_df.near_pc
        # pc1 = m.copy()
        # pc1 = mean_sub(pc1)
        # scatter_3d(pc1)

        pc2 = target.near_pc.copy()
        pc2 = mean_sub(pc2)
        # scatter_3d(pc2)

        print(m.shape)

        ms = mean_shift(m,target,ite=20,dis=0.01)
        ms.shift()
        pc3 = target.near_pc.copy()
        pc3 = mean_sub(pc3)
        scatter_3d(pc3)

        center = target.center
        center_list.append(center)

        print("original distance:",nn_distance(pc1,pc2))
        print("after meanshift:",nn_distance(pc1,pc3))
    # print(center_list)


    

if __name__ == "__main__":
    data_dir = os.environ["data"]
    ############ work on artificial data ############

    data0 = np.random.rand(100000,3)
    data0[:,2] = np.sqrt((data0[:,0]-0.5) **2 + (data0[:,1]-0.5)**2)
    # print(data0[:,2])
    tar = data_frame(data0,2,(0.5,0.5),0.05,bins=20)
    pc1 = np.concatenate((tar.near_coord,tar.near_attr),axis = 1)
    plt.scatter(pc1[:,0],pc1[:,1],c=pc1[:,2])
    plt.show()
    
    data1 = np.random.rand(100000,3)
    data1[:,2] = np.sqrt((data1[:,0]-0.3) **2 + (data1[:,1]-0.3)**2)
    aim = data_frame(data1,2,(0.5,0.5),0.05,bins=20)
    pc2 = np.concatenate((aim.near_coord,aim.near_attr),axis = 1)
    plt.scatter(pc2[:,0],pc2[:,1],c=pc2[:,2])
    plt.show()

    ms = mean_shift(tar.near_pc,aim,ite=10)
    ms.shift()
    pc3 = np.concatenate((aim.near_coord,aim.near_attr),axis = 1)
    plt.scatter(pc3[:,0],pc3[:,1],c=pc3[:,2])
    plt.show()

    pc1[:,0] -= np.mean(pc1[:,0])
    pc1[:,1] -= np.mean(pc1[:,1])
    # pc1[:,2] -= np.mean(pc1[:,2])
    pc2[:,0] -= np.mean(pc2[:,0])
    pc2[:,1] -= np.mean(pc2[:,1])
    # pc2[:,2] -= np.mean(pc2[:,2])
    pc3[:,0] -= np.mean(pc3[:,0])
    pc3[:,1] -= np.mean(pc3[:,1])
    # pc3[:,2] -= np.mean(pc3[:,2])

    print("original distance:",nn_distance(pc1,pc2))
    print("after meanshift:",nn_distance(pc1,pc3))
    
    #################### latent

    # data = data_reader(data_dir+r"\2016_scivis_fpm\0.44\run41\024.vtu")
    # data = data_to_numpy(data)
    # data = data[:,:4]
    
    # with open("data/latent_024","rb") as file:
    #     d = pickle.load(file).cpu()
    
    # with open("data/latent_025","rb") as file:
    #     d2 = pickle.load(file).cpu()

    # pca = PCA(n_components=1)
    # dd = np.concatenate((d,d2),axis=0)
    # pca.fit(d)
    # d_embedded = pca.transform(d)
    # d_embedded2 = pca.transform(d2)

    # center = (1.5,-1,6.25)
    # new_data = np.concatenate((data,d_embedded),axis=1)
    # model = data_frame(new_data,3,center,1,bins=12)
    # pc1 = np.concatenate((model.near_coord,model.near_attr),axis = 1)
    # pc1[:,0] -= np.mean(pc1[:,0])
    # pc1[:,1] -= np.mean(pc1[:,1])
    # pc1[:,2] -= np.mean(pc1[:,2])
    # # scatter_3d(pc1)

    # data2 = data_reader(data_dir+r"\2016_scivis_fpm\0.44\run41\025.vtu")
    # data2 = data_to_numpy(data2)
    # data2 = data2[:,:4]

    # # center2 = (1.2,-1.3,5.95)
    # new_aim = np.concatenate((data2,d_embedded2),axis=1)
    # target = data_frame(new_aim,3,center,1,bins=12)
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

    # print(pc1.shape)
    # print("original distance:",nn_distance(pc1[:,:4],pc2[:,:4]))
    # print("after meanshift:",nn_distance(pc1[:,:4],pc3[:,:4]))

    ############
    # t1 = datetime.now()

    center = (1.5,-1,6.25)
    di1 = data_dir+"\\2016_scivis_fpm\\0.44\\run41\\024.vtu"
    # di2 = data_dir+"\\2016_scivis_fpm\\0.44\\run41\\025.vtu"

    data = data_reader(di1)
    data = data_to_numpy(data)
    data = data[:,:4]

    # data2 = data_reader(di2)
    # data2 = data_to_numpy(data2)
    # data2 = data2[:,:4]

    model = data_frame(data,3,center,0.7,bins=1000)
    m = model.near_pc
    pc1 = model.near_pc.copy()
    pc1[:,0] -= np.mean(pc1[:,0])
    pc1[:,1] -= np.mean(pc1[:,1])
    pc1[:,2] -= np.mean(pc1[:,2])
    # scatter_3d(pc1)

    center2 = (0,0.5,6)

    target = data_frame(data,3,center2,0.7,bins=1000)
    pc2 = target.near_pc.copy()
    pc2[:,0] -= np.mean(pc2[:,0])
    pc2[:,1] -= np.mean(pc2[:,1])
    pc2[:,2] -= np.mean(pc2[:,2])
    # scatter_3d(pc2)

    ms = mean_shift(m,target,ite=30)
    ms.shift()
    pc3 = np.concatenate((target.near_coord,target.near_attr),axis = 1)
    pc3[:,0] -= np.mean(pc3[:,0])
    pc3[:,1] -= np.mean(pc3[:,1])
    pc3[:,2] -= np.mean(pc3[:,2])
    # scatter_3d(pc3)

    center = target.center

    # print(pc1.shape)
    print("original distance:",nn_distance(pc1,pc2))
    print("after meanshift:",nn_distance(pc1,pc3))

    # t2 = datetime.now()
    # print(t2-t1)
    # ###############
