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

    data_file = "./data/new_sample.npy"

    if args.load:
        model.load_state_dict(torch.load(args.load, map_location=device))
        print('Model loaded from {}'.format(args.load))

    if args.phase == 1:
        loader = Loader(data_file,args.batch_size)
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(loader.load_data(50000)):
                data = to_tensor_list(data,device,args.dim)
                recon_batch = model(data)
                pc1 = data[11].cpu()
                pc2 = recon_batch[11].cpu()
                pca = PCA(n_components=2)
                pca.fit(np.concatenate((pc1,pc2),axis=0))
                pc1_embedded = pca.transform(pc1)
                pc2_embedded = pca.transform(pc2)
                print(pc1_embedded.shape)
                print(pc2_embedded.shape)
                # plt.scatter(pc1_embedded[:,0],pc1_embedded[:,1])
                # plt.show()
                scatter_3d(pc1)
                # plt.scatter(pc2_embedded[:,0],pc2_embedded[:,1])
                # plt.show()
                scatter_3d(pc2)
    elif args.phase == 0:
        loader = Loader(data_file,args.batch_size)
        for epoch in range(1, args.epochs + 1):
            train(epoch)
            test(epoch)
    elif args.phase == 2:
        track_run(data_path+"/run41/",30,60,1,(1,-1,3.75),0.8,1000,model,device,args.dim,False)
        ############# latent shift
        
        # with open("data/latent_024","rb") as file:
        #     l1 = pickle.load(file)

        # with open("data/latent_025","rb") as file:
        #     l2 = pickle.load(file)

        # pca = PCA()
        # pca.fit(np.concatenate((l1,l2),axis = 0))
        
        # center = (1.5,-1,6.25)
        # di1 = data_path+"\\run41\\024.vtu"
        # di2 = data_path+"\\run41\\011.vtu"

        # data = data_reader(di1)
        # data = data_to_numpy(data)
        # data = data[:,:4]
        # scatter_3d(data,50,350,threshold=50)

        # data2 = data_reader(di2)
        # data2 = data_to_numpy(data2)
        # data2 = data2[:,:4]

        # start_df = latent_df(data,3,center,0.7,30,None,model,device,args.dim)
        # m = start_df.near_pc
        # pc1 = m.copy()
        # pc1 = mean_sub(pc1)
        # scatter_3d(pc1,None,None)

        # center2 = (0,0,7)

        # target = latent_df(data,3,center2,0.7,30,None,model,device,args.dim)
        # pc2 = target.near_pc.copy()
        # pc2 = mean_sub(pc2)
        # scatter_3d(pc2)

        # ms = mean_shift(m,target,ite=30)
        # ms.shift()
        # pc3 = target.near_pc.copy()
        # pc3 = mean_sub(pc3)
        # scatter_3d(pc3)

        # # center = target.center

        # print("original distance:",nn_distance(pc1,pc2))
        # print("after meanshift:",nn_distance(pc1,pc3))

        ############# guided shift
        # first test this on the same frame
        # center = (1.5,-1,6.25)
        # data = data_reader(data_path+r"\run41\024.vtu")
        # data = data_to_numpy(data)
        # data = data[:,:args.dim]
        # df = data_frame(data,3,center,0.7,bins=1000)
        # target_pc = df.near_pc.copy()

        # # # set the start center
        # df.center = (2,-0.5,6.55)
        # df.update()
        # guide = LatentMax(model,target_pc,device,args.dim)
        # gs = guided_shift(target_pc,df,guide)
        # gs.shift()


        ############################## convert one file to new features
        # filename = data_path + "/run41/024.vtu"
        # data = data_reader(filename)
        # data = data_to_numpy(data)
        # coord = data[:,:3]
        # attr = data[:,3:]
        # mean=[2.39460057e+01, -4.29336209e-03, 9.68809421e-04, 3.44706680e-02]
        # std=[55.08245731,  0.32457581,  0.32332313,  0.6972805]
        # data[:,3:] = (data[:,3:] - mean)/std
        # coord_kd = KDTree(coord)
        # i = 0
        # dd = []
        # for point in coord:
        #     ball = coord_kd.query_ball_point(point,r=0.7)
        #     print("{}/{}".format(i+1,len(data)),end='\r')
        #     dd.append(data[ball])
        #     i+=1
        # with open("run41_025","wb") as file:
        #     pickle.dump(dd,file)

        ################## encode to latent ##############
        # with open("run41_024","rb") as file:
        #     data = pickle.load(file)
        #     data = to_tensor_list(data,device,args.dim)

        # model.eval()
        # latent = torch.zeros((len(data),args.vector_length))
        # with torch.no_grad():
        #     for i in range(0,len(data),args.batch_size):
        #         batch = data[i:i+args.batch_size]
        #         latent[i:i+args.batch_size] = model.encode(batch)
        #         print("{}/{}".format(i+1,len(data)),end='\r')

        # with open("latent","wb") as file:
        #     pickle.dump(latent,file)
        # print(latent.shape)

        # with open("data/latent_024","rb") as file:
        #     d = pickle.load(file).cpu()

        # pca = PCA(n_components=5)
        # d_embedded = pca.fit_transform(d)
        # print(pca.explained_variance_ratio_)
        # pc = np.concatenate((coord,d_embedded[:,3:]),axis=1)
        # print(pc.shape)
        # scatter_3d(pc[::100],None,None)
        # scatter_3d(data[::100],None,None)

        ######################################

        # target = p2[118901][None,:]
        # distance = p2-target
        # distance = torch.norm(distance, dim = -1)
        # print(distance.shape)

        # torch.save(distance,"to_self")

        # p = torch.load("to_self",map_location=device)

        # p = (p-min(p))/(max(p)-min(p)) * 357.19
        # data = data_reader(os.environ['data']+"/2016_scivis_fpm/0.44/run01/013.vtu")
        # new_concen = numpy_support.numpy_to_vtk(p)
        # data.GetPointData().GetArray(0).SetArray(new_concen,new_concen.GetSize(),1)
        # data.GetPointData().GetArray(0)
        # # plt.hist(p,bins=100)
        # # plt.show()
        # # print(len(p))
        # # print(data)

        # # print(min(p),max(p))
        # writer = vtkXMLUnstructuredGridWriter()
        # writer.SetFileName("to_self.vtu")
        # writer.SetInputData(data)
        # writer.Write()

        # sort = np.argsort(p)
        # print(sort)

        # data = data_reader(data_path + "/0.44/run01/013.vtu")
        # coord = numpy_support.vtk_to_numpy(data.GetPoints().GetData())
        # print(coord[118901])
        # data = data_reader(data_path + "/0.44/run01/010.vtu")
        # coord = data.GetPoints().GetData()
        # xyz = coord.GetTuple(118901)
        # print(xyz)

        # data = data_reader("new.vtu")
        # concen = numpy_support.vtk_to_numpy(data.GetPointData().GetArray(0))
        # print(np.where(concen>350))
        # show("to_self.vtu")
        # show(data_path + "/0.44/run01/010.vtu")
