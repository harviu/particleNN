"""
Main function for network training using GeoConv
"""
import torch
from torch import optim
from torch.utils.data import DataLoader
import os, argparse
import numpy as np

from model.GeoConvNet import GeoConvNet
from utils.process_data import collate_ball, collect_file, data_reader, PointData
from train import train


def parse_arguments():
    parser = argparse.ArgumentParser(description='Training Script for Particle Latent Representation')
    # training parameters
    parser.add_argument('-b','--batch-size', type=int, default=128, help='input batch size for training')
    parser.add_argument('--sample-size', type=int, default=2000, help='sample size per file ')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--learning-rate', dest='lr', type=float, default=0.001, help='learning-rate')
    parser.add_argument('--sample-type', type=str, default='even',choices=['even','random'], help='Method used to sample')

    #model parameters (saved for inference)
    parser.add_argument('-v', '--lat-dim', type=int, default=16, help='Letent vector length')
    parser.add_argument('--ball', action='store_true', default=False, help='Train with ball or knn neighbor')
    parser.add_argument('-d','--data-source', dest='source', type=str, default="fpm", help='Data source type', choices=['fpm','cos','jet3b'])
    parser.add_argument('-k', dest='k', type=int, default=256, help='k in knn')
    parser.add_argument('-r', dest='r', type=float, default=0.03, help='r in ball query')
    parser.add_argument('--enc-out', type=int, default=256, help='Encoder output channel in geoGonv')

    #control parameters
    parser.add_argument('-l', '--load', dest='load', type=str,  help='load file model')
    parser.add_argument("--result-dir", dest="result_dir", type=str, default="states", help='the directory to save the result')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_arguments()
    load = args.load

    #Continue training
    if load:
        state_dict = torch.load(load)
        state = state_dict['state']
        config = state_dict['config']
        start_epoch = state_dict['end_epoch']
        args_dict = vars(args)
        args_dict.update(vars(config))
        args = argparse.Namespace(**args_dict) #Update with new parameters
    else:
        start_epoch = 1

    if not os.path.isdir(args.result_dir):
        os.mkdir(args.result_dir)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Total dimensionality of the dataset
    if args.source == "fpm":
        input_dim = 7
    elif args.source == "cos":
        input_dim = 10
    elif args.source == 'jet3b':
        input_dim = 5
    
    #Data directory
    try:
        data_path = os.environ['data']
    except KeyError:
        data_path = './data/'

    torch.manual_seed(args.seed)

    model = GeoConvNet(args.lat_dim, input_dim, args.ball, args.enc_out, args.r).float().to(device)
    if load:
        model.load_state_dict(state)
        print('Model loaded from {}'.format(load))
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    kwargs = {'pin_memory': True} if use_cuda else {}
    print(args)

    # prepare data
    if args.source == "fpm":
        file_list = collect_file(os.path.join(data_path,"2016_scivis_fpm/0.44/run41"),args.source,shuffle=True)
    elif args.source == "cos":
        file_list = collect_file(os.path.join(data_path,"ds14_scivis_0128/raw"),args.source,shuffle=True)
    elif args.source == "jet3b":
        file_list = collect_file(os.path.join(data_path,"jet3b"),args.source,shuffle=True)
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_loss = 0
        for i,f in enumerate(file_list):
            print("===> File in process: ",f)
            data_source = data_reader(f, args.source)
            if args.sample_type == 'random':
                choice = np.random.choice(len(data_source),args.sample_size)
                pd = PointData(data_source, args.k, args.r, args.ball, choice)
            elif args.sample_type == 'even':
                pd = PointData(data_source, args.k, args.r, args.ball, args.sample_size)
            loader = DataLoader(pd, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=collate_ball if args.ball else None, **kwargs,num_workers=0)
            train_loss = train(model,loader,optimizer,args.ball,device)
            epoch_loss += train_loss
            save_dict = {
                "state": model.state_dict(),
                "config":args,
                "end_epoch": epoch,
            }
            torch.save(save_dict,os.path.join(args.result_dir,'current_model.pth'))
            print("===> File processed: {}/{}".format(i+1,len(file_list)))
        epoch_loss /= len(file_list)
        print('==> Epoch average loss: {:.6f}'.format(epoch_loss))
        with open(os.path.join(args.result_dir,'epoch_loss_log.txt'),'a') as f:
            f.write("%f\n" % epoch_loss)
    save_dict = {
        "state": model.state_dict(),
        "config":args,
        "end_epoch": epoch,
    }
    torch.save(save_dict,os.path.join(args.result_dir,'final_model.pth'))
    print('Training complete. Final model saved!')