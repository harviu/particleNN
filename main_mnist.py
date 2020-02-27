import random
import os
import argparse

import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from scipy.signal import medfilt

from model import VAE
from process_data import to_tensor_list,image_to_pointcloud


def train(epoch):
    model.train()
    train_loss = 0
    for i, (raw, label) in enumerate(train_loader):
        points = image_to_pointcloud(raw,256)
        data = to_tensor_list(points,device)

        optimizer.zero_grad()
        recon_batch = model(data)
        loss = loss_function(recon_batch, data)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if i % args.log_interval == 0:
            print('Train Epoch: {} [{:.0f}%]\tLoss: {:.6f}'.format(
                epoch, 
                100. * i / len(train_loader),
                loss.item(),
                ))
    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / (i+1)))
    torch.save(model.state_dict(),'result_mnist/CP{}.pth'.format(epoch))
    print('Checkpoint {} saved !'.format(epoch))

def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (raw, label) in enumerate(test_loader):
            points = image_to_pointcloud(raw,256)
            data = to_tensor_list(points,device)

            recon_batch = model(data)
            loss = loss_function(recon_batch, data)
            test_loss += loss.item()

            d = data[0].cpu()
            r = recon_batch[0].cpu()
            plt.scatter(d[:,1],d[:,0])
            plt.gca().invert_yaxis()
            plt.axis('equal')
            plt.show()
            plt.scatter(r[:,1],r[:,0])
            plt.gca().invert_yaxis()
            plt.axis('equal')
            plt.show()

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
    parser.add_argument('-t', '--test', action='store_true', default=False,
                        help='test phase')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=7, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('-c', '--load', dest='load', type=str,
                        default=False, help='load file model')
    parser.add_argument('-v', '--vector', dest='vector_length', type=int,
                        default=1024, help='vector length')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    model = VAE(args.vector_length,2,128).float().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    try:
        data_dir = os.environ['data']
    except KeyError:
        data_dir = "../data"

    loss_function = model.loss
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(data_dir, train=True, download=True, transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(data_dir, train=False, download=True, transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    if args.load:
        model.load_state_dict(torch.load(args.load, map_location=device))
        print('Model loaded from {}'.format(args.load))

    if args.test:
        test(0)
    else:
        for epoch in range(1, args.epochs + 1):
            train(epoch)
            # test(epoch)