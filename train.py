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
def inference_latent(loader,model,args):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        latent_all = torch.zeros((len(loader.dataset),args.vector_length),dtype=torch.float32,device="cpu")
        for i, (d,index) in enumerate(loader):
            if isinstance(d,list):
                data = d[0][:,:,:args.dim].float().cuda()
                if args.mode=="ball" :
                    mask = d[1].cuda()
            else:
                data = d[:,:,:args.dim].float().cuda()

            latent = model.encode(data) 
            latent_all[i*args.batch_size:(i+1)*args.batch_size] = latent.detach().cpu()
            print("processed",i+1,"/",len(loader),end="\r")
        print()
    return latent_all


def train(model,loader:DataLoader,optimizer,args,epoch):
        log_interval = len(loader)//5
        model.train()
        device = args.device
        train_loss = 0
        print("number of samples: ",len(loader.dataset))
        for i, (d,index) in enumerate(loader):
            if isinstance(d,list):
                data = d[0][:,:,:args.dim].float().to(device)
                if args.mode=="ball" :
                    mask = d[1].to(device)
            else:
                data = d[:,:,:args.dim].float().to(device)
            optimizer.zero_grad()
            recon_batch = model(data) 

            # loss = model.MSE_loss(recon_batch, data,mask)
            loss = F.mse_loss(recon_batch,data[:,:,3:],reduction='mean')
            loss.backward()
            train_loss += loss.item() * len(data)
            optimizer.step()

            if i % log_interval == 0:
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
