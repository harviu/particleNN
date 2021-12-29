"""
Training helper functions
"""

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from utils.process_data import masked_mse_loss
def inference_latent(model,loader:DataLoader,lat_dim,ball,device,show=True):
    model.eval()
    cur_idx = 0
    if ball:
        n = sum([len(d[0]) for d in loader])
    else:
        n = sum([len(d) for d in loader])
    if show:
        print("===> number of samples: ",n)
    with torch.no_grad():
        latent_all = torch.zeros((n,lat_dim),dtype=torch.float32,device="cpu")
        for d in loader:
            if ball:
                data, mask = d
                mask = mask.to(device)
            else:
                data = d
            data = data.to(device)
            latent = model.encode(data,mask) if ball else model.encode(data)
            latent_all[cur_idx:cur_idx + len(data)] = latent.detach().cpu()
            cur_idx += len(data)
    return latent_all

def reconstruction(model,loader:DataLoader,out_dim,ball,device,show=True):
    model.eval()
    if ball:
        n = sum([len(d[0]) for d in loader])
    else:
        n = sum([len(d) for d in loader])
    if show:
        print("===> number of samples: ",n)
    output = torch.zeros((n,out_dim-3),dtype=torch.float32,device="cpu")
    cur_idx = 0
    with torch.no_grad():
        for d in loader:
            if ball:
                data, mask = d
                mask = mask.to(device)
            else:
                data = d
            data = data.to(device)
            recon_batch = model(data, mask) if ball else model(data)
            output[cur_idx:cur_idx + len(data)] = recon_batch[:,0,:].detach().cpu()
            cur_idx += len(data)
    return output


def train(model, loader:DataLoader, optimizer, ball, device):
    model.train()
    train_loss = 0
    count = 0
    print("===> number of samples: ",len(loader.dataset))
    for i, d in enumerate(loader):
        if ball:
            data, mask = d
            mask = mask.to(device)
        else:
            data = d
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch = model(data, mask) if ball else model(data)
        gt = data[...,3:]

        if ball:
            loss = masked_mse_loss(recon_batch,gt,mask)
        else:
            loss = F.mse_loss(recon_batch,gt,reduction='mean')
        loss.backward()
        train_loss += loss.item() * len(data)
        count += len(data)
        optimizer.step()

    train_loss /= count
    print('===> File Average loss: {:.6f}'.format(train_loss))
    return train_loss
