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
from mean_shift_copy import *
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
        track_run(data_path+"/run41/",15,16,1,(2,-1,8.2),0.8,100,model,device,args.dim,False)
