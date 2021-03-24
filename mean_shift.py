import os
import time
import pickle
import numpy as np
from matplotlib import pyplot as plt

from scipy.spatial import KDTree
from scipy.spatial.ckdtree import cKDTree
from sklearn.decomposition import PCA
from model.pointnet import PointNet as AE
from sklearn.cluster import KMeans

from process_data import *
from simple import show

class LatentRetriever():
    def __init__(self,data,model,args,latent_dimension,pca):
        source = args.source
        if source == "fpm":
            mean = [0, 0, 5, 23.9, 0, 0, 0.034]
            std = [2.68, 2.68, 3.09, 55.08, 0.3246, 0.3233, 0.6973]
        elif source == "cos":
            mean = [30.4, 32.8, 32.58, 0, 0, 0, 0, 0, 0, -732720]
            std = [18.767, 16.76, 17.62, 197.9, 247.2, 193.54, 420.92, 429, 422.3, 888474]
        data = normalize(data,mean,std)
        coord = data[:,:3]
        self.kd = cKDTree(coord,leafsize=30)
        self.mask = np.full((len(data),),False)
        self.args = args
        self.model = model
        self.data = data
        self.latent = np.zeros((len(data),latent_dimension))
        self.pca = pca
    def retriev(self,idx):
        args = self.args
        data = self.data
        kd = self.kd
        model = self.model
        not_infered = np.logical_not(self.mask)
        inference_idx = np.logical_and(not_infered,idx)
        #set infered
        self.mask = np.logical_or(self.mask,idx)
        if np.sum(inference_idx)>0:
            center = data[inference_idx,:3]
            _, nn = kd.query(center,args.k,n_jobs=-1)
            batch = data[nn][:,:,:args.dim]
            batch[:,:,:3] -= center[:,None,:]
            # batch -= data[inference_idx][:,None,:]
            batch =torch.FloatTensor(batch).cuda()
            model.eval()
            with torch.no_grad():
                latent = model.encode(batch).cpu().detach()
            self.latent[inference_idx] = self.pca.transform(latent)
        return self.latent[idx]

def filter(data,c1,c2,multiple = 1):
    # enlarge the area by multiple
    c1 = np.array(c1)
    c2 = np.array(c2)
    c0 = (c1+c2)/2
    c1 = c0 + multiple * (c1-c0)
    c2 = c0 + multiple * (c2-c0)
    x1,y1,z1 = c1
    x2,y2,z2 = c2
    condx = np.logical_and(data[:,0]>x1,data[:,0]<x2)
    condy = np.logical_and(data[:,1]>y1,data[:,1]<y2)
    condz = np.logical_and(data[:,2]>z1,data[:,2]<z2)
    cond = np.logical_and(condx,condy)
    cond = np.logical_and(cond,condz)
    return cond

def mean_shift_track(
    data1,data2,c1,c2, latent=False, 
    h=1,bins=10,eps=1e-4,ite=100, 
    model=None,args=None,latent_dimension=None,pca=None):

    # crop an approximate area
    t1 = data1[filter(data1,c1,c2,4)]
    d1_idx = filter(t1,c1,c2)
    d1 = t1[d1_idx]
    #show initial 
    # x1,y1,z1 = c1
    # x2,y2,z2 = c2
    # center = ((x1+x2)/2,(y1+y2)/2,(z1+z2)/2)
    # scatter_3d(t1,threshold=10,center=center)

    t2 = data2[filter(data2,c1,c2,4)]
    d2_idx = filter(t2,c1,c2)
    d2 = t2[d2_idx]
    center = np.mean(d1[:,:3],axis=0)
    if not latent:
        d1_attr = d1[:,3:]
    else:
        d1_lr = LatentRetriever(t1,model,args,latent_dimension,pca)
        d1_attr = d1_lr.retriev(d1_idx)
    w = 1 - np.sum(((d1[:,:3]-center)/h)**2,axis=-1)
    w[w<0] = 1e-5
    hist1,boundary = np.histogramdd(
        d1_attr,
        bins=bins,
        density=True,
        weights=w,
    )
    ranges = []
    for b in boundary:
        ranges.append((b[0],b[-1]))
    if not latent:
        d2_attr = d2[:,3:]
    else:
        d2_lr = LatentRetriever(t2,model,args,latent_dimension,pca)
        d2_attr = d2_lr.retriev(filter(t2,c1,c2))
    w = 1 - np.sum(((d2[:,:3]-center)/h)**2,axis=-1)
    w[w<0] = 1e-5
    hist2,_ = np.histogramdd(
        d2_attr,
        range=ranges,
        bins=bins,
        density=True,
        weights=w,
    )
    current_ite = 0
    reach_eps = False
    while(True):
        #calcualte initial similarity 
        init_similarity = hist_similarity(hist1,hist2)
        weights = np.sqrt(hist1/(hist2+1e-10))
        near_bins = []
        for i in range(d2_attr.shape[1]):
            bin_number = np.digitize(d2_attr[:,i],boundary[i],right=False)
            bin_number -= 1
            bin_number[bin_number==bins] = bins-1
            near_bins.append(bin_number)
        near_bins = tuple(near_bins)
        new_center = np.average(d2[:,:3],axis=0,weights=weights[near_bins])
        
        while (True):
            shift_vector = new_center - center
            # if the shift length is smaller than eps directly end 
            if np.sum(shift_vector **2) < eps ** 2:
                reach_eps = True
                break
            # update selection
            c1 += shift_vector
            c2 += shift_vector
            d2_idx = filter(t2,c1,c2)
            d2 = t2[d2_idx]
            if not latent:
                d2_attr = d2[:,3:]
            else:
                d2_attr = d2_lr.retriev(d2_idx)
            w = 1 - np.sum(((d2[:,:3]-new_center)/h)**2,axis=-1)
            w[w<0] = 1e-5
            hist2,_ = np.histogramdd(
                d2_attr,
                range=ranges,
                bins=bins,
                density=True,
                weights=w,
            )
            new_similarity = hist_similarity(hist1,hist2)
            # fine tune shift length
            if (new_similarity > init_similarity):
                break
            else:
                new_center = (center + new_center)/2
        center = new_center
        #check for ending conditions
        if reach_eps:
            break
        current_ite+=1
        if current_ite == ite:
            break
    # show final
    # x1,y1,z1 = c1
    # x2,y2,z2 = c2
    # center = ((x1+x2)/2,(y1+y2)/2,(z1+z2)/2)
    # scatter_3d(t2,threshold=10,center=center)

    return c1,c2, current_ite
    
def hist_similarity(h1,h2):
    h1 = h1.reshape(-1)
    h2 = h2.reshape(-1)
    return np.sum(np.sqrt(h1*h2))

def weighted_hist(near_coord,near_attr, center, h,bins,ranges):
    weights = 1 - np.sum(((near_coord-center)/h)**2,axis=-1)
    hist = np.histogramdd(
        near_attr,
        bins = bins,
        range = ranges,
        weights = weights,
        density = True)
    print(hist[0])
    hist[0][hist[0]<0] = 0
    return hist[0]


    

if __name__ == "__main__":
    
    try:
        data_path = os.environ['data']
    except KeyError:
        data_path = './data/'
        
    state_dict = torch.load("states_saved/fpm_knn128_dim7_vec64_CP5.pth")
    state = state_dict['state']
    args = state_dict['config']
    print(args)
    # halo_info = halo_reader(data_path+"/ds14_scivis_0128/rockstar/out_47.list")
    model = AE(args).float().cuda()
    model.load_state_dict(state)
    cluster_centers = np.load("cluster_center.npy")[3]
    
    # run41_25
    # c1 = (3.2,0.1,5.2)
    # c2 = (4.4,1.3,6.4)
    # c1 = (-2.7,-2.7,4.9)
    # c2 = (-1.3,-1.3,6.1)
    # c1 = (0.25,-2,5.3)
    # c2 = (1.75,-0,6.3)
    # c1 = (-2.5,-1,5.8)
    # c2 = (-1.5,0.2,6.7)
    c1 = (0,3.5,6.5)
    c2 = (1,4.5,7.5)
    # run41_35
    # c1 = (0.5,-1.8,2.5)
    # c2 = (2,-0.2,3.5)
    # run09_25
    # c1 = (1.2,-1.8,5)
    # c2 = (2.4,-0.8,6)

    with open("pca","rb") as f:
        pca = pickle.load(f)
    print(c1,c2)

    average_time_list = []
    length_list = []
    for ww in range(50):
        print(ww)
        w = 0.3 + 0.1 * math.sqrt(ww)
        c1 = (0.5-w,4-w,7-w)
        c2 = (0.5+w,4+w,7+w)
        data_directory = data_path+"/2016_scivis_fpm/0.44/run41/025.vtu"
        data = vtk_reader(data_directory)

        # latent = torch.load("run41_latent/{:03d}".format(25))
        # km = KMeans(5,n_init=3,n_jobs=-1)
        # embedding = km.fit_predict(latent)
        # new_cluster = km.cluster_centers_
        # distance = np.sum((new_cluster - cluster_centers) ** 2,-1)
        # interested_cluster = np.argmin(distance)
        # cluster_id = embedding
        # data = data[cluster_id==interested_cluster]
        # array_dict = {
        #     "concentration": data[:,3],
        #     "velocity": data[:,4:]
        # }
        # vtk_data = numpy_to_vtk(data[:,:3],array_dict)
        # show(vtk_data,c1=c1,c2=c2,outfile="tracking1_b5/25",show=False,time="25")
        # scatter_3d(data[::3],threshold=10)

        iteration_list = []
        time_list = []

        d1_idx = filter(data,c1,c2)
        d1_length = np.sum(d1_idx)
        # print(d1_length)
        length_list.append(d1_length)

        for i in range(25,45):
            data_directory = data_path+"/2016_scivis_fpm/0.44/run41/{:03d}.vtu".format(i)
            data1 = vtk_reader(data_directory)
            data1 = data1[:,:args.dim]

            data_directory = data_path+"/2016_scivis_fpm/0.44/run41/{:03d}.vtu".format(i+1)
            data2 = vtk_reader(data_directory)
            data2 = data2[:,:args.dim]

            t1 = time.time()
            c1,c2,iteration_number = mean_shift_track(data1,data2,c1,c2,True,h=1,bins=2,model=model,args=args,latent_dimension=4,pca=pca)
            t2 = time.time()
            time_list.append(t2-t1)

            iteration_list.append(iteration_number)

            # latent = torch.load("run41_latent/{:03d}".format(i+1))
            # km = KMeans(5,n_init=3,n_jobs=-1)
            # embedding = km.fit_predict(latent)
            # new_cluster = km.cluster_centers_
            # distance = np.sum((new_cluster - cluster_centers) ** 2,-1)
            # interested_cluster = np.argmin(distance)
            # cluster_id = embedding
            # data2 = data2[cluster_id==interested_cluster]
            # array_dict = {
            #     "concentration": data2[:,3],
            #     "velocity": data2[:,4:]
            # }
            # vtk_data = numpy_to_vtk(data2[:,:3],array_dict)
            # show(vtk_data,c1=c1,c2=c2,outfile="tracking1_b5/{}".format(i+1),show=False,time=str(i+1))
        # print(iteration_list)
        average_time = sum(time_list[1:])/(len(time_list)-1)
        average_time_list.append(average_time)

    np.save("length_list",length_list)
    np.save("average_time_list",average_time_list)

