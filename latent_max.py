import os
import numpy as np

import torch
from torch import optim
from process_data import scatter_3d,normalize


class LatentMax():
    """
        Optimize the input point cloud to the specific latent representation
    """
    def __init__(self, model, target, device, dim):
        self.model = model
        self.model.eval()
        self.device = device 
        self.dim = dim
        # change target to sensor list (target was ndarray)
        # scatter_3d(target)
        # self.target = target.copy()
        target = normalize([target],device,3,dim)
        encoder = self.model.encode
        self.target_latent = encoder(target)

    def take_one_step_to_target(self,data_input):
        # !!!! change data_input to value not tensor
        # scatter_3d(data_input)
        # print(data_input-self.target)
        data_input = normalize([data_input],self.device,3,self.dim)
        coord = data_input[0][:,:3]
        attr = data_input[0][:,3]
        attr = torch.autograd.Variable(attr, requires_grad=True)
        # Define optimizer for the image
        # how does optimizer influence?
        # optimizer = optim.SGD(data_input, lr=0.1)
        optimizer = optim.Adam([attr], lr=0.01)
        original_loss = None
        while True:
            optimizer.zero_grad()
            # Assign create image to a variable to move forward in the model
            x = [torch.cat((coord,attr[:,None]),axis = 1)]
            # print(x.shape)
            encoder = self.model.encode
            # latent is the latent representation for initial point cloud
            latent = encoder(x)
            # Loss function is latent presentation of init pc to target latent
            # optimize the average for now, still we can optimize every dimension?
            loss = torch.sqrt(torch.sum((self.target_latent - latent)**2))
            if original_loss is None:
                original_loss = loss.item()
            # Backward
            loss.backward(retain_graph=True)
            # Update image
            optimizer.step()
            new_pc = data_input[0].detach().numpy()
            # scatter_3d(new_pc)
            # print(loss.item())
            if loss.item()< 0.8 * original_loss:
                print(loss.item())
                break
        # return ndarray
        new_pc = data_input[0].detach().numpy()
        # scatter_3d(new_pc)
        return new_pc


if __name__ == '__main__':
    pass