import os
import numpy as np

import torch
from torch import optim
from process_data import to_tensor_list,scatter_3d


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
        target = to_tensor_list([target],device,dim)
        encoder = self.model.encode
        mu, logvar = encoder(target)
        self.target_latent = self.model.reparameterize(mu, logvar)

    def take_one_step_to_target(self,data_input):
        # !!!! change data_input to value not tensor
        # scatter_3d(data_input)
        data_input = to_tensor_list([data_input],self.device,self.dim)
        data_input[0] = torch.autograd.Variable(data_input[0], requires_grad=True)
        # Define optimizer for the image
        # how does optimizer influence?
        # optimizer = optim.SGD(data_input, lr=0.1)
        optimizer = optim.Adam(data_input, lr=0.1)
        optimizer.zero_grad()
        # Assign create image to a variable to move forward in the model
        x = data_input
        encoder = self.model.encode
        # latent is the latent representation for initial point cloud
        mu, logvar = encoder(x)
        latent = self.model.reparameterize(mu, logvar)
        # Loss function is latent presentation of init pc to target latent
        # optimize the average for now, still we can optimize every dimension?
        loss = torch.mean(self.target_latent - latent)
        # Backward
        loss.backward(retain_graph=True)
        # Update image
        optimizer.step()
        print(data_input)
        # return ndarray
        new_pc = data_input[0].detach().numpy()
        # scatter_3d(new_pc)
        return new_pc


if __name__ == '__main__':
    pass