"""
Training helper functions
"""

import random
import os
import argparse

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

from process_data import *
def inference(loader,model,args):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        # latent_all = torch.zeros((len(loader.dataset),args.vector_length),dtype=torch.float32,device="cpu")
        predict_all = torch.zeros((len(loader.dataset),args.dim-3),dtype=torch.float32,device="cpu")
        for i, d in enumerate(loader):
            if isinstance(d,list):
                data = d[0][:,:,:args.dim].float().cuda()
                if args.mode=="ball" :
                    mask = d[1].cuda()
            else:
                data = d[:,:,:args.dim].float().cuda()

            # latent = model.encode(data) 
            # latent_all[i*args.batch_size:(i+1)*args.batch_size] = latent.detach().cpu()
            dist = torch.sum(data[:,:,:3] ** 2,axis=-1)
            idx = torch.argsort(dist,axis=-1)[:,1]
            pred = model(data)
            pred = pred[torch.arange(pred.shape[0]),idx]
            predict_all[i*args.batch_size:(i+1)*args.batch_size] = pred.detach().cpu()
            # print("processed",i+1,"/",len(loader),end="\r")
        # print()
    return predict_all


def train(model,loader:DataLoader,optimizer,args,epoch):
        model.train()
        device = args.device
        train_loss = 0
        print("number of samples: ",len(loader.dataset))
        for i, d in enumerate(loader):
            if isinstance(d,list):
                data = d[0][:,:,:args.dim].float().to(device)
                if args.mode=="ball" :
                    mask = d[1].to(device)
            else:
                data = d[:,:,:args.dim].float().to(device)
            optimizer.zero_grad()
            recon_batch = model(data) 

            loss = model.MSE_loss(recon_batch, data,mask)
            loss.backward()
            train_loss += loss.item() * len(data)
            optimizer.step()

            if i % args.log_interval == 0:
                print('Train Epoch: {} [{:.0f}%]\tLoss: {:.6f}'.format(
                    epoch, 
                    100. * i / len(loader),
                    loss.item(),
                    ))
            if i == len(loader)-1:
                train_loss /= len(loader.dataset)
                print('====> Epoch: {} Average loss: {:.4f}'.format(
                    epoch, train_loss))
        if not os.path.isdir(args.result_dir):
            os.mkdir(args.result_dir)
        with open(args.result_dir+'/loss','a') as f:
            f.write("%f\n" % train_loss)
        if epoch == args.epochs:
            save_dict = {
                "state": model.state_dict(),
                "config":args,
            }
            torch.save(save_dict,args.result_dir+'/CP{}.pth'.format(epoch))
            print('Checkpoint {} saved !'.format(epoch))
