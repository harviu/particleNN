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
        data = normalize(data,3,10)
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
        self.latent_length = 6

        data = normalize(data,3,10)
        latent = np.zeros((len(data),self.latent_length))
        self.latent_mask = [False]* len(data)
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
        not_cal = []
        for n in self.near:
            if not self.latent_mask[n]:
                not_cal.append(n)
                self.latent_mask[n] = True
        coord = self.coord[not_cal]
        t = self.kd.query_ball_point(coord,0.6)
        if len(t) > 0:
            x = []
            for tt in t:
                xx = self.data[tt][:,:-self.latent_length]
                xx = mean_sub(xx,3)
                # print(xx)
                x.append(xx)
            x = to_tensor_list(x,device,dim)
            # print(x[0].shape)
            with torch.no_grad():
                y = model.encode(x)
            latent = y.detach().cpu().numpy()   
            # print(latent.shape) 
            self.data[not_cal,-self.latent_length:] = latent
            self.near_coord = self.coord[self.near]
            self.near_attr = self.attr[self.near]
            self.near_pc = self.data[self.near]



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

    def adaptive_range(self,data,target):
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

        # set new range
        data.set_range(new_ranges)
        data.update_hist()

        target_center = np.mean(target[:,:coord_dim],axis=0)
        target_hist = weighted_hist(target[:,:coord_dim],target[:,coord_dim:],target_center,data.h,data.bins,new_ranges)

        return target_hist, new_ranges

    def next_center(self):

        data = self.data
        target = self.target
        coord_dim = self.data.coord_dim
        
        target_hist, new_ranges = self.adaptive_range(data,target)

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
        
        # print(new_center)
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
            #calcualte initial similarity 
            target_hist, new_ranges = self.adaptive_range(self.data,self.target)
            init_similarity = hist_similarity(target_hist,self.data.hist)
            # print(init_similarity)
            
            center = self.data.center
            next_center = self.next_center()

            count = 0 
            while (True):
                #calculate new similarity
                self.data.set_center(next_center)
                self.data.update_hist()
                target_hist, new_ranges = self.adaptive_range(self.data,self.target)
                new_similarity = hist_similarity(target_hist,self.data.hist)
                count += 1
                if (new_similarity > init_similarity or count == 13):
                    break
                else:
                    next_center = (center + next_center)/2

            # t1 = datetime.now()
            # self.data.set_center((next_center+self.data.center)/2)
            # self.data.update_hist()

            i+=1
            if i == self.ite or np.sqrt(np.sum((center-next_center)**2))<self.dis:
                break
        print("Mean_shift_next_center",self.data.center)
        return self.data

def hist_similarity(h1,h2):
    h1 = h1.reshape(-1)
    h2 = h2.reshape(-1)
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

def track_run(path,start,end,step,init_center,h,bins,model,device,dim,latent=True):
    center = init_center
    center_list = []
    center_list.append(center)
    # data = data_to_numpy(data_reader(path+"\{:03d}.vtu".format(start)))
    # data = data[:,:dim]
    # start_df = latent_df(data,3,center,h,bins,None,model,device,dim)
    # start_df = data_frame(data,3,center,h,bins,None)
    # m = start_df.near_pc.copy()
    # pc1 = m.copy()
    # pc1 = mean_sub(pc1)
    # print(center)
    # scatter_3d(pc1,center=center)
    for i in range(start,end+step-1,step):
        data = data_to_numpy(data_reader(path+"\ds14_scivis_0128_e4_dt04_0.{:02d}00".format(i)))
        data = data[:,:dim]
        # scatter_3d(data,50,350,50,center,False)
        # scatter_3d(data,50,350,50,center,True,"{:03d}.png".format(i))

        data_next = data_to_numpy(data_reader(path+"\ds14_scivis_0128_e4_dt04_0.{:02d}00".format(i+step)))
        data_next = data_next[:,:dim]

        if latent:
            start_df = latent_df(data,3,center,h,bins,None,model,device,dim)
            target = latent_df(data_next,3,center,h,bins,None,model,device,dim)
        else:
            start_df = data_frame(data,3,center,h,bins,None)
            target = data_frame(data_next,3,center,h,bins,None)
        
        m = start_df.near_pc.copy()
        pc1 = m.copy()
        pc1 = mean_sub(pc1)
        # scatter_3d(pc1)

        pc2 = target.near_pc.copy()
        pc2 = mean_sub(pc2)
        # scatter_3d(pc2)

        ms = mean_shift(m,target,ite=20,dis=0.01)
        ms.shift()
        pc3 = target.near_pc.copy()
        pc3 = mean_sub(pc3)
        # scatter_3d(pc3)

        dis1 = nn_distance(pc1,pc2)
        dis2 = nn_distance(pc1,pc3)
        # if dis2< dis1:
        center = target.center
        center_list.append(tuple(center))

        # print("original distance:",dis1)
        # print("after meanshift:",dis2)
    print(center_list)
    return center_list

def get_benchmark(path, start,end,index):
    from thingking import loadtxt
    center_list = []
    for i in range(start,end+1):
        ID, DescID, Mvir, Vmax, Vrms, Rvir, Rs, Np, x, y, z, VX, VY, VZ, JX, JY, JZ, Spin, rs_klypin, Mvir_all, M200b, M200c, M500c, M2500c, Xoff, Voff, spin_bullock, b_to_a, c_to_a, A_x_, A_y_, A_z_, b_to_a_500c_, c_to_a_500c_, A_x__500c_, A_y__500c_, A_z__500c_, TU, M_pe_Behroozi, M_pe_Diemer = \
            loadtxt(path+"/rockstar/out_{:02d}.list".format(i-2), unpack=True)
        order = list(ID).index(index)
        index = DescID[order]
        this_center = (x[order],y[order],z[order])
        center_list.append(this_center)
    print(center_list)
    return center_list

def mean_error(track_list,truth_list):
    distance = np.array(track_list)-np.array(truth_list)
    distance = distance[1:]
    distance = distance **2
    distance = np.sum(distance,axis=1)
    distance = np.sqrt(distance)
    return np.mean(distance)

    

if __name__ == "__main__":
    data_dir = os.environ["data"]
    data_dir += "\\ds14_scivis_0128\\raw"
    track_list = track_run(data_dir,37,67,1,(24.94006, 30.34540, 13.88314),0.5,2,None,None,10,False)
    truth_list = get_benchmark(os.environ["data"]+"\\ds14_scivis_0128",37,67,9)
    print(mean_error(track_list,truth_list))

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

    # ms = mean_shift(tar.near_pc,aim,ite=10)
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

    ############
    # t1 = datetime.now()

    # center = (1.5,-1,6.25)
    # di1 = data_dir+"\\2016_scivis_fpm\\0.44\\run41\\024.vtu"
    # # di2 = data_dir+"\\2016_scivis_fpm\\0.44\\run41\\025.vtu"

    # data = data_reader(di1)
    # data = data_to_numpy(data)
    # data = data[:,:4]

    # # data2 = data_reader(di2)
    # # data2 = data_to_numpy(data2)
    # # data2 = data2[:,:4]

    # model = data_frame(data,3,center,0.7,bins=1000)
    # m = model.near_pc
    # pc1 = model.near_pc.copy()
    # pc1[:,0] -= np.mean(pc1[:,0])
    # pc1[:,1] -= np.mean(pc1[:,1])
    # pc1[:,2] -= np.mean(pc1[:,2])
    # # scatter_3d(pc1)

    # center2 = (1.2,-0.5,6)

    # target = data_frame(data,3,center2,0.7,bins=1000)
    # pc2 = target.near_pc.copy()
    # pc2[:,0] -= np.mean(pc2[:,0])
    # pc2[:,1] -= np.mean(pc2[:,1])
    # pc2[:,2] -= np.mean(pc2[:,2])
    # # scatter_3d(pc2)

    # ms = mean_shift(m,target,ite=30,dis=0.001)
    # ms.shift()
    # pc3 = np.concatenate((target.near_coord,target.near_attr),axis = 1)
    # pc3[:,0] -= np.mean(pc3[:,0])
    # pc3[:,1] -= np.mean(pc3[:,1])
    # pc3[:,2] -= np.mean(pc3[:,2])
    # # scatter_3d(pc3)

    # center = target.center

    # # print(pc1.shape)
    # print("original distance:",nn_distance(pc1,pc2))
    # print("after meanshift:",nn_distance(pc1,pc3))

    # t2 = datetime.now()
    # print(t2-t1)
    # ###############
