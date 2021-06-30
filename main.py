"""
Main function for network training using GeoConv
"""
import torch
from torch import optim
from torch.utils.data import DataLoader
import os, argparse
import numpy as np

from model.pointnet import PointNet
from process_data import collect_file,data_reader,PointData
from train import train


def parse_arguments():
    parser = argparse.ArgumentParser(description='PointNet')
    parser.add_argument('-b','--batch-size', type=int, default=512, help='input batch size for training')
    parser.add_argument('--sample-size', type=int, default=1000, help='sample size per file ')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('-p', '--patch-number', type=int,default=125,dest='p', help='how many patches in which one file is seperated')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=7, help='how many batches to wait before logging training status')
    parser.add_argument('-l', '--load', dest='load', type=str, default=False, help='load file model')
    parser.add_argument('-v', '--vector', dest='vector_length', type=int, default=16, help='vector length')
    parser.add_argument('--ball', dest='ball', action='store_true', default=False, help='train with ball surrounding')
    parser.add_argument('--learning-rate', dest='lr', type=float, default=0.001, help='learning-rate')
    parser.add_argument('-d','--data-source', dest='source', type=str, default="fpm", help='data source')
    parser.add_argument('-k', dest='k', type=int, default=256, help='k in knn')
    parser.add_argument('-r', dest='r', type=float, default=0.04, help='r in ball query')
    parser.add_argument('--enc-out', dest='enc_out', type=int, default=256, help='encoder output channel in geoGonv')
    parser.add_argument("--result-dir", dest="result_dir", type=str, default="states", help='the directory to save the result')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_arguments()
    load = args.load
    if load:
        state_dict = torch.load(load)
        state = state_dict['state']
        # load model related arguments
        config = state_dict['config']
        args = config
        args.start_epoch = config.epochs + 1 # change the start epoch for continous training
    else:
        args.start_epoch = 1
        args.cuda = not args.no_cuda and torch.cuda.is_available()
        args.device = torch.device("cuda" if args.cuda else "cpu")
        args.mode = "ball" if args.ball else 'knn'

    if args.source == "fpm":
        args.dim = 7
    elif args.source == "cos":
        args.dim = 10
    elif args.source == "eth":
        args.dim = 5
    
    try:
        data_path = os.environ['data']
    except KeyError:
        data_path = './data/'
    torch.manual_seed(args.seed)
    model = PointNet(args).float().to(args.device)
    if load:
        model.load_state_dict(state)
        print('Model loaded from {}'.format(load))
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    kwargs = {'pin_memory': True} if args.cuda else {}
    print(args)

    # prepare data
    if args.source == "eth":
        file_name = data_path + "/ethanediol.vti"
        data_source = data_reader(file_name, args.source)
        for epoch in range(args.start_epoch, args.epochs + 1):
            # choice = np.random.choice(len(data_source),args.sample_size)
            pd = PointData(data_source,args)
            loader = DataLoader(pd, batch_size=args.batch_size, shuffle=True, drop_last=False, **kwargs)
            train(model,loader,optimizer,args,epoch)
            
    else:
        if args.source == "fpm":
            file_list = collect_file(data_path+"/2016_scivis_fpm/0.44/run03",args.source,shuffle=True)
        elif args.source == "cos":
            file_list = collect_file(data_path+"/ds14_scivis_0128/raw",args.source,shuffle=True)
        for epoch in range(args.start_epoch, args.epochs + 1):
            for i,f in enumerate(file_list):
                print("file in process: ",f)
                data_source = data_reader(f, args.source)
                pd = PointData(data_source,args)
                loader = DataLoader(pd, batch_size=args.batch_size, shuffle=True, drop_last=False, **kwargs)
                train(epoch,args,loader)
                print("file processed {}/{}".format(i+1,len(file_list)))
            

