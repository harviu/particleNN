import os
import numpy as np

import torch
from torch import optim


class LatentMax():
    """
        Optimize the input point cloud to the specific latent representation
    """
    def __init__(self, model, latent):
        self.model = model
        self.model.eval()
        self.target = latent

    def take_one_step_to_target(self,data_input):
        # !!!! change data_input to value not tensor
        data_input[0] = torch.autograd.Variable(data_input[0], requires_grad=True)
        # Define optimizer for the image
        # how does optimizer influence?
        # optimizer = optim.SGD(init_pc, lr=0.1)
        optimizer = optim.Adam(data_input, lr=0.1)
        optimizer.zero_grad()
        # Assign create image to a variable to move forward in the model
        x = data_input
        encoder = self.model.encode
        # latent is the latent representation for initial point cloud
        latent = encoder(x)
        # Loss function is latent presentation of init pc to target latent
        # optimize the average for now, still we can optimize every dimension?
        loss = torch.mean(self.target - latent)
        print(loss)
        # Backward
        loss.backward(retain_graph=True)
        # Update image
        optimizer.step()
        data_input[0] = data_input[0].detach()
        return data_input


if __name__ == '__main__':
    pass