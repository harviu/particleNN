import os
import numpy as np

import torch
from torch.optim import Adam


class LatentMax():
    """
        Optimize the input point cloud to the specific latent representation
    """
    def __init__(self, model, latent):
        self.model = model
        self.model.eval()
        self.target = latent
        self.conv_output = 0

    def take_one_step_to_target(self,init_pc):
        # Define optimizer for the image
        optimizer = Adam(init_pc, lr=0.1, weight_decay=1e-6)
        optimizer.zero_grad()
        # Assign create image to a variable to move forward in the model
        x = init_pc
        encoder = self.model.encode
        # latent is the latent representation for initial point cloud
        latent = encoder(x)
        # Loss function is latent presentation of init pc to target latent
        # optimize the average for now, still we can optimize every dimension?
        loss = torch.mean(self.target - latent)
        # Backward
        loss.backward()
        # Update image
        optimizer.step()
        return init_pc


if __name__ == '__main__':
    pass