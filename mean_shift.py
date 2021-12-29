import os
import time
import numpy as np
import torch

from sklearn.decomposition import PCA
from thingking import loadtxt
from torch.utils.data import DataLoader

from model.GeoConvNet import GeoConvNet
from utils.simple import show
from utils.process_data import data_reader, collect_file,PointData, numpy_to_vtp, collate_ball, halo_reader
from train import inference_latent

try:
    data_path = os.environ['data']
except KeyError:
    data_path = './data/'

class LatentRetriever():
    def __init__(self,data,model,lat_dim,k,r,ball,device:torch.device):
        self.pd = PointData(data,k,r,ball,np.arange(len(data)))
        self.mask = np.full((len(data),),False)
        self.model = model
        self.data = data
        self.lat_dim = lat_dim
        self.latent = np.zeros((len(data),lat_dim))
        self.device = device
        self.ball = ball
    def retriev(self,idx):
        data = self.data
        model = self.model
        not_infered = np.logical_not(self.mask)
        inference_idx = not_infered & idx
        self.mask = self.mask | idx #update mask
        if np.sum(inference_idx)>0: #get latent
            kwargs = {'pin_memory': True} if self.device.type=='cuda' else {}
            loader = DataLoader(self.pd, batch_sampler=[np.arange(len(data))[inference_idx]], collate_fn=collate_ball if args.ball else None, **kwargs)
            self.latent[inference_idx] = inference_latent(model, loader, self.lat_dim, self.ball, self.device,show=False)
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
    cond = condx & condy & condz
    return cond

def mean_shift_track(
    data1, data2, c1, c2, latent=False, 
    h=1, bins=10, eps=1e-4, ite=100, 
    d1_lr=None, d2_lr=None, pca = None):
    '''
    c1, c2: chosen bounding box
    latent: use latent or original attributes
    h: bandwidth
    bins: bins per dimension
    pca: saved pca for dimension reduction
    '''

    # crop an approximate area
    d1_idx = filter(data1,c1,c2)
    d1 = data1[d1_idx]

    # initial center
    x1,y1,z1 = c1
    x2,y2,z2 = c2
    center = np.array([(x1+x2)/2,(y1+y2)/2,(z1+z2)/2])
    # center = np.mean(d1[:,:3],axis=0)
    # scatter_3d(t1,threshold=10,center=center)

    d2_idx = filter(data2,c1,c2)
    d2 = data2[d2_idx]
    if not latent:
        d1_attr = d1[:,3:]
    else:
        d1_attr = d1_lr.retriev(d1_idx)
        d1_attr = pca.transform(d1_attr)
    w = 1 - np.sum(((d1[:,:3]-center)/h)**2,axis=-1)
    w[w<0] = 1e-10
    hist1,boundary = np.histogramdd(d1_attr, bins=bins, weights=w)
    hist1 /= hist1.sum()
    ranges = []
    for b in boundary:
        ranges.append((b[0],b[-1]))
    if not latent:
        d2_attr = d2[:,3:]
    else:
        d2_attr = d2_lr.retriev(d2_idx)
        d2_attr = pca.transform(d2_attr)
    w = 1 - np.sum(((d2[:,:3]-center)/h)**2,axis=-1)
    w[w<0] = 1e-10
    hist2,_ = np.histogramdd(d2_attr, range=ranges, bins=bins, weights=w)
    hist2 /= hist2.sum()
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
            new_c1 = c1+ shift_vector
            new_c2 = c2+ shift_vector
            d2_idx = filter(data2,new_c1,new_c2)
            d2 = data2[d2_idx]
            if not latent:
                d2_attr = d2[:,3:]
            else:
                d2_attr = d2_lr.retriev(d2_idx)
                d2_attr = pca.transform(d2_attr)
            w = 1 - np.sum(((d2[:,:3]-new_center)/h)**2,axis=-1)
            w[w<0] = 1e-10
            hist2,_ = np.histogramdd(d2_attr,range=ranges,bins=bins, weights=w)
            hist2 /= hist2.sum()
            new_similarity = hist_similarity(hist1,hist2)
            # fine tune shift length
            if (new_similarity > init_similarity):
                c1 = new_c1
                c2 = new_c2
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

    return c1,c2, current_ite
    
def hist_similarity(h1,h2):
    h1 = h1.reshape(-1)
    h2 = h2.reshape(-1)
    return np.sum(np.sqrt(h1*h2))


def get_benchmark(path, start,end,index):
    center_list = []
    r = []
    for i in range(start,end):
        ID, DescID, Mvir, Vmax, Vrms, Rvir, Rs, Np, x, y, z, VX, VY, VZ, JX, JY, JZ, Spin, rs_klypin, Mvir_all, M200b, M200c, M500c, M2500c, Xoff, Voff, spin_bullock, b_to_a, c_to_a, A_x_, A_y_, A_z_, b_to_a_500c_, c_to_a_500c_, A_x__500c_, A_y__500c_, A_z__500c_, TU, M_pe_Behroozi, M_pe_Diemer = \
            loadtxt(path+"/ds14_scivis_0128/rockstar/out_{:02d}.list".format(i), unpack=True)
        order = list(ID).index(index) # get order of the halo idx
        this_center = (x[order],y[order],z[order]) # get this center
        this_r = Rvir[order] # get this radius
        center_list.append(this_center)
        r.append(this_r/1000)
        index = DescID[order] # get next index
        if index == -1:
            print("halo disappear")
            break
    # print(center_list)
    return center_list, r
    
def normalize_fpm(c):
    return  (np.array(c)/10 + np.array([0.5,0.5,0])).tolist()

def normalize_cos(c):
    return  (np.array(c)/62.5).tolist()

if __name__ == "__main__":
    load_filename = './example/final_model.pth'
    use_cuda = torch.cuda.is_available()
    state_dict = torch.load(load_filename,map_location='cuda' if use_cuda else 'cpu')
    state = state_dict['state']
    args = state_dict['config']
    device = torch.device('cuda') if use_cuda else torch.device('cpu')
    data_type = args.source
    print(args)
    input_dim = 7 if args.source == "fpm" else 10
    model = GeoConvNet(args.lat_dim, input_dim, args.ball, args.enc_out, args.r).float().to(device)
    model.load_state_dict(state)
    model.eval()
    torch.set_grad_enabled(False)

    if args.source == "fpm":
        # run41_25
        file_list = collect_file(os.path.join(data_path, "2016_scivis_fpm/0.44/run41"),args.source,shuffle=False)
        start_ts = 24
        # c1 = (3.2,0.1,5.2)
        # c2 = (4.4,1.3,6.4)
        c1 = (-2.7,-2.7,4.9)
        c2 = (-1.3,-1.3,6.1)
        # c1 = (0.25,-2,5.3)
        # c2 = (1.75,-0,6.3)
        # c1 = (-2.5,-1,5.8)
        # c2 = (-1.5,0.2,6.7)
        # c1 = (0,3.5,6.5)
        # c2 = (1,4.5,7.5)
        # run41_35
        # start_ts = 34
        # c1 = (0.5,-1.8,2.5)
        # c2 = (2,-0.2,3.5)
        # run09_25
        # start_ts = 24
        # file_list = collect_file(os.path.join(data_path, "2016_scivis_fpm/0.44/run09"),args.source,shuffle=False)
        # c1 = (1.2,-1.8,5)
        # c2 = (2.4,-0.8,6)
        c1 = normalize_fpm(c1)
        c2 = normalize_fpm(c2)
    elif args.source == "cos":
        file_list = collect_file(os.path.join(data_path,"ds14_scivis_0128/raw"),args.source,shuffle=False)
        start_ts = 47
        # center, r = halo_reader(data_path+"/ds14_scivis_0128/rockstar/out_%d.list" % start_ts)
        halo_id = 3633
        gt_c,gt_r = get_benchmark(data_path,47,58,3633)
        gt_c = normalize_cos(gt_c)
        gt_r = normalize_cos(gt_r)
        xyz = gt_c[0]
        margin = 0.05 / 62.5
        width = gt_r[0] + margin
        c1 = (xyz[0]-width,xyz[1]-width,xyz[2]-width)
        c2 = (xyz[0]+width,xyz[1]+width,xyz[2]+width)



    pca = PCA(4)
    latent = np.load(os.path.join(args.result_dir,'latent.npy'))
    pca.fit(latent)

    iteration_list = []
    time_list = []
    c_list = [(np.array(c1)+np.array(c2))/2]
    show_fig = False
    use_latent = True

    for i in range(start_ts,start_ts+10):
        data1 = data_reader(file_list[i],data_type)
        ds1 = data1[filter(data1,c1,c2,5)]

        if i == start_ts:
            # visualize init position
            coord = ds1[:,:3]
            if args.source == 'fpm':
                array_dict = {
                    "concentration": ds1[:,3],
                    "velocity": ds1[:,4:]
                }
            else:
                array_dict = {
                    "phi":ds1[:,-1],
                    "velocity":ds1[:,3:6],
                    "acceleration":ds1[:,6:9],
                }
            vtk_data = numpy_to_vtp(coord,array_dict)
            show(vtk_data,time=str(i),c1=c1,c2=c2,outfile="result_tracking/{}".format(i),show=show_fig,data_type=args.source,thres=True)

        data2 = data_reader(file_list[i+1],data_type)
        ds2 = data2[filter(data2,c1,c2,5)]

        t1 = time.time()
        d1_lr = LatentRetriever(ds1,model,args.lat_dim,args.k,args.r,args.ball,device)
        d2_lr = LatentRetriever(ds2,model,args.lat_dim,args.k,args.r,args.ball,device)
        c1,c2,iteration_number = mean_shift_track(ds1,ds2,c1,c2,use_latent,d1_lr=d1_lr,d2_lr=d2_lr,pca=pca)
        t2 = time.time()
        time_list.append(t2-t1)

        c = (np.array(c1)+np.array(c2))/2
        c_list.append(c)

        iteration_list.append(iteration_number)

        coord = ds2[:,:3]
        if args.source == 'fpm':
            array_dict = {
                "concentration": ds2[:,3],
                "velocity": ds2[:,4:]
            }
        else:
            array_dict = {
                "phi":ds2[:,-1],
                "velocity":ds2[:,3:6],
                "acceleration":ds2[:,6:9],
            }
        vtk_data = numpy_to_vtp(coord,array_dict)
        show(vtk_data,time=str(i+1),c1=c1,c2=c2,outfile="result_tracking/{}".format(i+1),show=show_fig,data_type=args.source,thres=True)
    print("tracking time:", time_list)
    print("tracking iteration:", iteration_list)
    print("tracked result:", np.array(c_list))
    if args.source == 'cos':
        dist = np.sqrt(np.sum((np.array(gt_c) - np.array(c_list)) ** 2,axis=-1))
        print("tracking_distance:", dist/np.array(gt_r))


