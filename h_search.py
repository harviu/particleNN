from utils.process_data import all_file_loader, PointData, collate_ball, collect_file
from torch.utils.data import DataLoader
import torch
import os, math, time

try:
    data_path = os.environ['data']
except KeyError:
    data_path = './data/'



def get_error (r,data):
    eps = 1e-15
    sample_size = 10000
    batch_size = sample_size
    ri = r/2
    ro = r
    # choice = np.random.choice(len(data),sample_size)
    pd = PointData(data ,256 ,r , False, sample_size)
    summed_error = 0
    loader = DataLoader(pd, batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=collate_ball)
    for d in loader:
        data, mask = d
        sq_dist = torch.sum(data[...,:3]**2,axis=-1,keepdim=True)
        weight = 1 - (sq_dist - ri **2) / (ro**2 -ri ** 2)
        weight[weight < 0] = 1e-10 # avoid devided by zero
        weight[sq_dist< eps] = eps # remove center
        weight[torch.logical_not(mask)] = eps
        weight /= torch.sum(weight,axis=1,keepdim=True)
        est = (data[...,3:] * weight).sum(1)
        error = ((est - data[:,0,3:]) ** 2).mean(1).sum()
        summed_error += error.item()
    summed_error /= len(pd)
    return summed_error
    
def error_helper(args):
    d = args[0]
    r = args[1]
    return get_error(r,d)

def all_file_error(r,data):
    # pool = Pool(4)
    # summed = sum(pool.map(error_helper,zip(data,[r]*len(data))))
    summed = 0
    for d in data:
        summed += get_error(r,d)
    return summed / len(data)

def gss(data, a, b, tol=1e-5):
    """Golden-section search
    to find the minimum of f on [a,b]
    f: a strictly unimodal function on [a,b]
    """
    gr = (math.sqrt(5) + 1) / 2
    c = b - (b - a) / gr
    d = a + (b - a) / gr
    while abs(b - a) > tol:
        if all_file_error(c,data) < all_file_error(d,data):
            b = d
        else:
            a = c
        print("r =",(b + a) / 2)
        # We recompute both c and d here to avoid loss of precision which may lead to incorrect results or infinite loop
        c = b - (b - a) / gr
        d = a + (b - a) / gr

    return (b + a) / 2




if __name__ == "__main__":
    data_type = 'fpm_h'

    if data_type == 'jet3b':
        test_data = data_path + "/jet3b/run3g_50Am_jet3b_sph.3400"
        file_list = collect_file(os.path.join(data_path,"jet3b"), data_type, shuffle=False)
    elif data_type == 'fpm':
        test_data = data_path + "/2016_scivis_fpm/0.44/run03/025.vtu"
        file_list = collect_file(os.path.join(data_path,"2016_scivis_fpm/0.44/run41"), data_type, shuffle=False)
    elif data_type == 'cos':
        test_data = data_path + '/ds14_scivis_0128/raw/ds14_scivis_0128_e4_dt04_0.3500'
        file_list = collect_file(os.path.join(data_path,"ds14_scivis_0128/raw"), data_type, shuffle=False)
    elif data_type == 'fpm_h':
        test_data = data_path + '/2016_scivis_fpm/0.20/run03/025.vtu'
        data_type = 'fpm'
        file_list = collect_file(os.path.join(data_path,'2016_scivis_fpm/0.20/run03'), data_type, shuffle=False)

    data = all_file_loader(file_list,data_type)
    
    t1 = time.time()
    val = gss(data,0.01,0.05,tol=0.001)
    print("final r =", val)
    print("time used: ", time.time()-t1)
