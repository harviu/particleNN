import random
import os
import argparse
import pickle

import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision.utils import save_image

from matplotlib import pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import vtk
from vtk import *
from vtk.util import numpy_support

from model import VAE
from process_data import *
from latent_max import LatentMax
from mean_shift import *
from simple import show



def train(epoch):
    model.train()
    train_loss = 0
    for i, data in enumerate(loader.load_data(0,50000)):
        data = to_tensor_list(data,device,args.dim)
        optimizer.zero_grad()
        recon_batch = model(data)
        loss = loss_function(recon_batch, data)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if i % args.log_interval == 0:
            print('Train Epoch: {} [{:.0f}%]\tLoss: {:.6f}'.format(
                epoch, 
                100. * i / (50000//args.batch_size),
                loss.item(),
                ))
    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / (i+1)))
    torch.save(model.state_dict(),'result/CP{}.pth'.format(epoch))
    print('Checkpoint {} saved !'.format(epoch))

def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, data in enumerate(loader.load_data(50000)):
            data = to_tensor_list(data,device,args.dim)
            recon_batch = model(data)
            loss = loss_function(recon_batch, data)
            test_loss += loss.item()
            # scatter_3d(data[0].cpu())
            # scatter_3d(recon_batch[0].cpu())

    test_loss /= (i+1)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    

def random_check():
    start_frame = 71
    end_frame = 79
    from thingking import loadtxt
    path = os.environ["data"]+"\\ds14_scivis_0128"
    ID, DescID, Mvir, Vmax, Vrms, Rvir, Rs, Np, x, y, z, VX, VY, VZ, JX, JY, JZ, Spin, rs_klypin, Mvir_all, M200b, M200c, M500c, M2500c, Xoff, Voff, spin_bullock, b_to_a, c_to_a, A_x_, A_y_, A_z_, b_to_a_500c_, c_to_a_500c_, A_x__500c_, A_y__500c_, A_z__500c_, TU, M_pe_Behroozi, M_pe_Diemer = \
        loadtxt(path+"/rockstar/out_{:02d}.list".format(start_frame-2), unpack=True)

    center = np.array((x,y,z)).T
    me = []
    for i,c in enumerate(center):
        if Rvir[i]>300 and Rvir[i]<400 and i == 63:
            print("halo id: ",i)
            track_list = track_run(os.environ["data"]+"\\ds14_scivis_0128\\raw",start_frame,end_frame,1,c,0.4,2,model,device,args.dim,True)
            truth_list = get_benchmark(os.environ["data"]+"\\ds14_scivis_0128",start_frame,end_frame,ID[i])
            mme = mean_error(track_list,truth_list)
            me.append(mme)
            print(mme)
            break
        if len(me)==10:
            break
    me = np.array(me)
    print("all tested mean: ", np.mean(me))
    # np.save("result_saved/w_3",me)

def track_more():
    from thingking import loadtxt
    path = os.environ["data"]+"\\ds14_scivis_0128"
    ID, DescID, Mvir, Vmax, Vrms, Rvir, Rs, Np, x, y, z, VX, VY, VZ, JX, JY, JZ, Spin, rs_klypin, Mvir_all, M200b, M200c, M500c, M2500c, Xoff, Voff, spin_bullock, b_to_a, c_to_a, A_x_, A_y_, A_z_, b_to_a_500c_, c_to_a_500c_, A_x__500c_, A_y__500c_, A_z__500c_, TU, M_pe_Behroozi, M_pe_Diemer = \
        loadtxt(path+"/rockstar/out_{:d}.list".format(12-2), unpack=True)
    center = np.array((x,y,z)).T
    me = []
    id_list = []

    for i,c in enumerate(center):
        # if Rvir[i]>300 and Rvir[i]<400:
        print(i)
        r = max(1,Rvir[i]/1000)
        if i in [1,10,12]:
            continue
        try:
            truth_list = get_benchmark(os.environ["data"]+"\\ds14_scivis_0128",12,99,ID[i])
            # track_list = track_run(os.environ["data"]+"\\ds14_scivis_0128\\raw",12,99,1,c,r,2,model,device,args.dim,False)
            me.append(truth_list)
            id_list.append(i)
            if len(me)==30:
                break
        except ValueError:
            pass
        # mme = mean_error(track_list,truth_list)
    me = np.array(me)
    print(id_list)
    np.save("result_saved/truth_list",me)

if __name__ == "__main__":
    # input parsing
    parser = argparse.ArgumentParser(description='VAE MNIST Example')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('-p', '--phase', type=int,default=0,dest="phase",
                        help='phase')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=7, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('-w', '--load', dest='load', type=str,
                        default=False, help='load file model')
    parser.add_argument('-v', '--vector', dest='vector_length', type=int,
                        default=1024, help='vector length')
    parser.add_argument('-d', '--dim', dest='dim', type=int,
                        default=4, help='number of point dimensions')
    # parser.add_argument('-b', '--ball', dest='ball', action='store_true',
    #                     default=False, help='train with ball surrounding')
    parser.add_argument('--lr', dest='lr', type=float, default=0.001, help='learning-rate')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    try:
        data_path = os.environ['data'] + "/2016_scivis_fpm/0.44/"
    except KeyError:
        data_path = './data/'

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    model = VAE(args.vector_length,args.dim,256).float().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_function = model.loss

    # if args.ball:
    #     train_file = data_path+ "/ball_4_60000_normalized.npy"
    #     test_file = data_path+"/ball_4_10000_normalized.npy"
    # else:
    #     train_file = data_path+"/knn_128_4_60000.npy"
    #     test_file = data_path+"/knn_128_4_10000.npy"

    data_file = "./data/new.npy"

    if args.load:
        model.load_state_dict(torch.load(args.load, map_location=device))
        print('Model loaded from {}'.format(args.load))

    if args.phase == 1:
        loader = Loader(data_file,args.batch_size)
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(loader.load_data(0,50000)):
                data = to_tensor_list(data,device,args.dim)
                recon_batch = model(data)
                pc1 = data[11].cpu()
                pc2 = recon_batch[11].cpu()
                # pca = PCA(n_components=2)
                # pca.fit(np.concatenate((pc1,pc2),axis=0))
                # pc1_embedded = pca.transform(pc1)
                # pc2_embedded = pca.transform(pc2)
                # plt.scatter(pc1_embedded[:,0],pc1_embedded[:,1])
                # plt.show()
                # plt.scatter(pc2_embedded[:,0],pc2_embedded[:,1])
                # plt.show()
                scatter_3d(pc1[:,:4])
                scatter_3d(pc2[:,:4])
    elif args.phase == 0:
        loader = Loader(data_file,args.batch_size)
        for epoch in range(1, args.epochs + 1):
            train(epoch)
            test(epoch)
    elif args.phase == 2:
        random_check()
        # track_more()
        # track_list = track_run(os.environ["data"]+"\\ds14_scivis_0128\\raw",64,65,1,(2.76107, 30.67916, 11.95021),0.5,5,model,device,args.dim,False)
        # # track_list = [(24.94006, 30.3454, 13.88314), (25.020839443134015, 30.265330533416304, 13.717739060525183), (25.022390640517735, 30.265052441598918, 13.71440159593395), (25.026354747960873, 30.260516090309903, 13.63621451340121), (25.02644167937652, 30.2610223807962, 13.634984933676332), (25.03030100406902, 30.268520801604573, 13.620667915454497), (25.0556379532077, 30.310062760831826, 13.536663225457454), (25.060218022990064, 30.312160101878955, 13.525382675703927), (25.060220428835144, 30.312174206796307, 13.52535706736345), (25.12134407349925, 30.2943903012317, 13.36246014555585), (25.121342538179952, 30.294376871573192, 13.362453425065418), (25.10144800798555, 30.219118889829492, 13.279267399330013), (25.21986066725828, 30.559612319941063, 13.26933020260508), (25.21983822740463, 30.559549548288537, 13.269315357252161), (25.2188909625197, 30.557357829522665, 13.26921878389376), (25.187036795727117, 30.494596069305025, 13.280045349301004), (25.166096234330617, 30.473872885853204, 13.305058123954588), (25.13165940204533, 30.420586732130907, 13.27423491818125), (25.131020835890503, 30.42039148250388, 13.273549407670878), (25.124720162301152, 30.430625734286, 13.244873975535338), (25.122955023610714, 30.428104332993165, 13.242714872078515), (25.11877896480072, 30.421023730754523, 13.245281541378775), (25.119419011491345, 30.424337931281066, 13.24509254420824), (25.131902877611072, 30.46306752886657, 13.238944771128335), (25.22045358042592, 30.726903530549894, 13.199939478631105), (25.220422796950537, 30.726890722514412, 13.200150691126327), (25.142609620895733, 30.770683056833732, 13.594719766085412), (25.14005913170371, 30.76818225192866, 13.55889248286667), (25.13960150515846, 30.768577037948774, 13.541051134762661), (25.122865142290422, 30.802028307852346, 13.328673005257588), (25.137963580509805, 30.80194672075249, 13.313973103451524)]
        # # track_list = track_list[:6]
        # truth_list = get_benchmark(os.environ["data"]+"\\ds14_scivis_0128",64,65,5)
        # print(track_list,truth_list)
        # print(mean_error(track_list,truth_list))
