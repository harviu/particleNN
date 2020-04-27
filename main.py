import random
import os
import argparse
import pickle

import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision.utils import save_image

from matplotlib import pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import vtk
from vtk import *
from vtk.util import numpy_support

from model import VAE
from process_data import *
from latent_max import LatentMax
from mean_shift import *
from simple import show



def train(epoch):
    model.train()
    train_loss = 0
    for i, data in enumerate(loader.load_data(0,50000)):
        data = to_tensor_list(data,device,args.dim)
        optimizer.zero_grad()
        recon_batch = model(data)
        loss = loss_function(recon_batch, data)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if i % args.log_interval == 0:
            print('Train Epoch: {} [{:.0f}%]\tLoss: {:.6f}'.format(
                epoch, 
                100. * i / (50000//args.batch_size),
                loss.item(),
                ))
    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / (i+1)))
    torch.save(model.state_dict(),'result/CP{}.pth'.format(epoch))
    print('Checkpoint {} saved !'.format(epoch))

def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, data in enumerate(loader.load_data(50000)):
            data = to_tensor_list(data,device,args.dim)
            recon_batch = model(data)
            loss = loss_function(recon_batch, data)
            test_loss += loss.item()
            # scatter_3d(data[0].cpu())
            # scatter_3d(recon_batch[0].cpu())

    test_loss /= (i+1)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    

def random_check():
    from thingking import loadtxt
    path = os.environ["data"]+"\\ds14_scivis_0128"
    ID, DescID, Mvir, Vmax, Vrms, Rvir, Rs, Np, x, y, z, VX, VY, VZ, JX, JY, JZ, Spin, rs_klypin, Mvir_all, M200b, M200c, M500c, M2500c, Xoff, Voff, spin_bullock, b_to_a, c_to_a, A_x_, A_y_, A_z_, b_to_a_500c_, c_to_a_500c_, A_x__500c_, A_y__500c_, A_z__500c_, TU, M_pe_Behroozi, M_pe_Diemer = \
        loadtxt(path+"/rockstar/out_{:02d}.list".format(91-2), unpack=True)

    center = np.array((x,y,z)).T
    me = []
    for i,c in enumerate(center):
        if Rvir[i]>300 and Rvir[i]<400:
            print(i)
            track_list = track_run(os.environ["data"]+"\\ds14_scivis_0128\\raw",91,99,1,c,0.4,2,model,device,args.dim,True)
            truth_list = get_benchmark(os.environ["data"]+"\\ds14_scivis_0128",91,99,ID[i])
            mme = mean_error(track_list,truth_list)
            print(mme)
            me.append(mme)
        if len(me)==3:
            break
    me = np.array(me)
    np.save("result_saved/w_3",me)

def track_more():
    from thingking import loadtxt
    path = os.environ["data"]+"\\ds14_scivis_0128"
    ID, DescID, Mvir, Vmax, Vrms, Rvir, Rs, Np, x, y, z, VX, VY, VZ, JX, JY, JZ, Spin, rs_klypin, Mvir_all, M200b, M200c, M500c, M2500c, Xoff, Voff, spin_bullock, b_to_a, c_to_a, A_x_, A_y_, A_z_, b_to_a_500c_, c_to_a_500c_, A_x__500c_, A_y__500c_, A_z__500c_, TU, M_pe_Behroozi, M_pe_Diemer = \
        loadtxt(path+"/rockstar/out_{:d}.list".format(12-2), unpack=True)
    center = np.array((x,y,z)).T
    me = []

    for i,c in enumerate(center):
        # if Rvir[i]>300 and Rvir[i]<400:
        print(i)
        r = max(1,Rvir[i]/1000)
        if i == 12:
            continue
        # try:
        # truth_list = get_benchmark(os.environ["data"]+"\\ds14_scivis_0128",12,99,ID[i])
        track_list = track_run(os.environ["data"]+"\\ds14_scivis_0128\\raw",12,99,1,c,r,2,model,device,args.dim,False)
        me.append(track_list)
        if len(me)==30:
            break
        # except ValueError:
        #     pass
        # mme = mean_error(track_list,truth_list)
    me = np.array(me)
    np.save("result_saved/track_list",me)

if __name__ == "__main__":
    # input parsing
    parser = argparse.ArgumentParser(description='VAE MNIST Example')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('-p', '--phase', type=int,default=0,dest="phase",
                        help='phase')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=7, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('-w', '--load', dest='load', type=str,
                        default=False, help='load file model')
    parser.add_argument('-v', '--vector', dest='vector_length', type=int,
                        default=1024, help='vector length')
    parser.add_argument('-d', '--dim', dest='dim', type=int,
                        default=4, help='number of point dimensions')
    # parser.add_argument('-b', '--ball', dest='ball', action='store_true',
    #                     default=False, help='train with ball surrounding')
    parser.add_argument('--lr', dest='lr', type=float, default=0.001, help='learning-rate')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    try:
        data_path = os.environ['data'] + "/2016_scivis_fpm/0.44/"
    except KeyError:
        data_path = './data/'

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    model = VAE(args.vector_length,args.dim,256).float().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_function = model.loss

    # if args.ball:
    #     train_file = data_path+ "/ball_4_60000_normalized.npy"
    #     test_file = data_path+"/ball_4_10000_normalized.npy"
    # else:
    #     train_file = data_path+"/knn_128_4_60000.npy"
    #     test_file = data_path+"/knn_128_4_10000.npy"

    data_file = "./data/new.npy"

    if args.load:
        model.load_state_dict(torch.load(args.load, map_location=device))
        print('Model loaded from {}'.format(args.load))

    if args.phase == 1:
        loader = Loader(data_file,args.batch_size)
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(loader.load_data(0,50000)):
                data = to_tensor_list(data,device,args.dim)
                recon_batch = model(data)
                pc1 = data[11].cpu()
                pc2 = recon_batch[11].cpu()
                # pca = PCA(n_components=2)
                # pca.fit(np.concatenate((pc1,pc2),axis=0))
                # pc1_embedded = pca.transform(pc1)
                # pc2_embedded = pca.transform(pc2)
                # plt.scatter(pc1_embedded[:,0],pc1_embedded[:,1])
                # plt.show()
                # plt.scatter(pc2_embedded[:,0],pc2_embedded[:,1])
                # plt.show()
                scatter_3d(pc1[:,:4])
                scatter_3d(pc2[:,:4])
    elif args.phase == 0:
        loader = Loader(data_file,args.batch_size)
        for epoch in range(1, args.epochs + 1):
            train(epoch)
            test(epoch)
    elif args.phase == 2:
        # random_check()
        track_more()
        # track_list = track_run(os.environ["data"]+"\\ds14_scivis_0128\\raw",64,65,1,(2.76107, 30.67916, 11.95021),0.5,2,model,device,args.dim,True)
        # # track_list = [(24.94006, 30.3454, 13.88314), (25.020839443134015, 30.265330533416304, 13.717739060525183), (25.022390640517735, 30.265052441598918, 13.71440159593395), (25.026354747960873, 30.260516090309903, 13.63621451340121), (25.02644167937652, 30.2610223807962, 13.634984933676332), (25.03030100406902, 30.268520801604573, 13.620667915454497), (25.0556379532077, 30.310062760831826, 13.536663225457454), (25.060218022990064, 30.312160101878955, 13.525382675703927), (25.060220428835144, 30.312174206796307, 13.52535706736345), (25.12134407349925, 30.2943903012317, 13.36246014555585), (25.121342538179952, 30.294376871573192, 13.362453425065418), (25.10144800798555, 30.219118889829492, 13.279267399330013), (25.21986066725828, 30.559612319941063, 13.26933020260508), (25.21983822740463, 30.559549548288537, 13.269315357252161), (25.2188909625197, 30.557357829522665, 13.26921878389376), (25.187036795727117, 30.494596069305025, 13.280045349301004), (25.166096234330617, 30.473872885853204, 13.305058123954588), (25.13165940204533, 30.420586732130907, 13.27423491818125), (25.131020835890503, 30.42039148250388, 13.273549407670878), (25.124720162301152, 30.430625734286, 13.244873975535338), (25.122955023610714, 30.428104332993165, 13.242714872078515), (25.11877896480072, 30.421023730754523, 13.245281541378775), (25.119419011491345, 30.424337931281066, 13.24509254420824), (25.131902877611072, 30.46306752886657, 13.238944771128335), (25.22045358042592, 30.726903530549894, 13.199939478631105), (25.220422796950537, 30.726890722514412, 13.200150691126327), (25.142609620895733, 30.770683056833732, 13.594719766085412), (25.14005913170371, 30.76818225192866, 13.55889248286667), (25.13960150515846, 30.768577037948774, 13.541051134762661), (25.122865142290422, 30.802028307852346, 13.328673005257588), (25.137963580509805, 30.80194672075249, 13.313973103451524)]
        # # track_list = track_list[:6]
        # truth_list = get_benchmark(os.environ["data"]+"\\ds14_scivis_0128",64,67,5)
        # print(mean_error(track_list,truth_list))

        # track_run(data_path+"/run41/",15,20,1,(2, -1.2, 8.2),0.5,1000,model,device,args.dim,False)
        # track_run(data_path+"/run41/",15,20,1,(3.7, 0.7, 7.9),0.5,10,model,device,args.dim,True)
        #best case
        # print("with")
        # track_run(data_path+"/run41/",10,19,1,(3.7, 0.7, 9.1),0.5,10,model,device,args.dim,True)
        # print("without")
        # track_run(data_path+"/run41/",10,19,1,(3.7, 0.7, 9.1),0.5,1000,model,device,args.dim,False)
        # track_run(data_path+"/run40/",10,20,1,(3.7, 0.7, 9.1),0.5,10,model,device,args.dim,True)
        # track_run(data_path+"/run40/",10,20,1,(1, -3.4, 9.2),0.5,10,model,device,args.dim,True)
        # track_run(data_path+"/run32/",20,30,1,(3.7, 0.5, 6.5),0.5,10,model,device,args.dim,True)
        # track_run(data_path+"/run05/",20,30,1,(-0.5, -3.7, 6.5),0.5,1000,model,device,args.dim,False)
        # track_run(data_path+"/run05/",20,30,1,(4.2, -2.1, 7),0.5,10,model,device,args.dim,True)
        # track_run(data_path+"/run05/",15,30,1,(4.2, -2.1, 8.2),0.5,1000,model,device,args.dim,False)

        # vtk_data = data_reader(data_path+"/run01/020.vtu")
        # # print(vtk_data)
        # numpy_data = data_to_numpy(vtk_data)
        # print(numpy_data.shape)
        # kdtree = KDTree(numpy_data[:,:3],1000)
        # x = np.arange(-5,5,0.5)
        # y = np.arange(-5,5,0.5)
        # z = np.arange(0,10,0.5)
        # index = []
        # for xx in x:
        #     for yy in y:
        #         for zz in z:
        #             if (xx**2 + yy**2 <25):
        #                 index.append((xx,yy,zz))

        
        # points = kdtree.query_ball_point((-0.9,-1.2,6.4),0.8)
        # # print(points)
        # numpy_data = numpy_data[:,:4]
        # index = numpy_data[:,:3][points]
        # pc = numpy_data[points]  
        # points = kdtree.query_ball_point(index,0.7)
        # data = []
        # for p in points:
        #     d = numpy_data[p]
        #     d = mean_sub(d)
        #     d[:,3] = (d[:,3] - 23.946)/55.08
        #     data.append(d)
        # data = to_tensor_list(data,device)
        # with torch.no_grad():
        #     latent = model.encode(data)
        #     latent = latent.cpu().numpy()

        # sc0 = np.concatenate((index,latent[:,0][:,None]),axis=1)
        # sc1 = np.concatenate((index,latent[:,1][:,None]),axis=1)
        # # print(sc.shape)
        # scatter_3d(pc)
        # scatter_3d(sc0)
        # scatter_3d(sc1)
        
            
        ############# latent shift
        
        # with open("data/latent_024","rb") as file:
        #     l1 = pickle.load(file)

        # with open("data/latent_025","rb") as file:
        #     l2 = pickle.load(file)

        # pca = PCA()
        # pca.fit(np.concatenate((l1,l2),axis = 0))
        
        # center = (1.5,-1,6.25)
        # di1 = data_path+"\\run41\\024.vtu"
        # di2 = data_path+"\\run41\\011.vtu"

        # data = data_reader(di1)
        # data = data_to_numpy(data)
        # data = data[:,:4]
        # scatter_3d(data,50,350,threshold=50)

        # data2 = data_reader(di2)
        # data2 = data_to_numpy(data2)
        # data2 = data2[:,:4]

        # start_df = latent_df(data,3,center,0.7,30,None,model,device,args.dim)
        # m = start_df.near_pc
        # pc1 = m.copy()
        # pc1 = mean_sub(pc1)
        # scatter_3d(pc1,None,None)

        # center2 = (0,0,7)

        # target = latent_df(data,3,center2,0.7,30,None,model,device,args.dim)
        # pc2 = target.near_pc.copy()
        # pc2 = mean_sub(pc2)
        # scatter_3d(pc2)

        # ms = mean_shift(m,target,ite=30)
        # ms.shift()
        # pc3 = target.near_pc.copy()
        # pc3 = mean_sub(pc3)
        # scatter_3d(pc3)

        # # center = target.center

        # print("original distance:",nn_distance(pc1,pc2))
        # print("after meanshift:",nn_distance(pc1,pc3))

        ############# guided shift
        # first test this on the same frame
        # center = (1.5,-1,6.25)
        # data = data_reader(data_path+r"\run41\024.vtu")
        # data = data_to_numpy(data)
        # data = data[:,:args.dim]
        # df = data_frame(data,3,center,0.7,bins=1000)
        # target_pc = df.near_pc.copy()

        # # # set the start center
        # df.center = (2,-0.5,6.55)
        # df.update()
        # guide = LatentMax(model,target_pc,device,args.dim)
        # gs = guided_shift(target_pc,df,guide)
        # gs.shift()


        ############################## convert one file to new features
        # filename = data_path + "/run41/024.vtu"
        # data = data_reader(filename)
        # data = data_to_numpy(data)
        # coord = data[:,:3]
        # attr = data[:,3:]
        # mean=[2.39460057e+01, -4.29336209e-03, 9.68809421e-04, 3.44706680e-02]
        # std=[55.08245731,  0.32457581,  0.32332313,  0.6972805]
        # data[:,3:] = (data[:,3:] - mean)/std
        # coord_kd = KDTree(coord)
        # i = 0
        # dd = []
        # for point in coord:
        #     ball = coord_kd.query_ball_point(point,r=0.7)
        #     print("{}/{}".format(i+1,len(data)),end='\r')
        #     dd.append(data[ball])
        #     i+=1
        # with open("run41_025","wb") as file:
        #     pickle.dump(dd,file)

        ################## encode to latent ##############
        # with open("run41_024","rb") as file:
        #     data = pickle.load(file)
        #     data = to_tensor_list(data,device,args.dim)

        # model.eval()
        # latent = torch.zeros((len(data),args.vector_length))
        # with torch.no_grad():
        #     for i in range(0,len(data),args.batch_size):
        #         batch = data[i:i+args.batch_size]
        #         latent[i:i+args.batch_size] = model.encode(batch)
        #         print("{}/{}".format(i+1,len(data)),end='\r')

        # with open("latent","wb") as file:
        #     pickle.dump(latent,file)
        # print(latent.shape)

        # with open("data/latent_024","rb") as file:
        #     d = pickle.load(file).cpu()

        # pca = PCA(n_components=5)
        # d_embedded = pca.fit_transform(d)
        # print(pca.explained_variance_ratio_)
        # pc = np.concatenate((coord,d_embedded[:,3:]),axis=1)
        # print(pc.shape)
        # scatter_3d(pc[::100],None,None)
        # scatter_3d(data[::100],None,None)

        ######################################

        # target = p2[118901][None,:]
        # distance = p2-target
        # distance = torch.norm(distance, dim = -1)
        # print(distance.shape)

        # torch.save(distance,"to_self")

        # p = torch.load("to_self",map_location=device)

        # p = (p-min(p))/(max(p)-min(p)) * 357.19
        # data = data_reader(os.environ['data']+"/2016_scivis_fpm/0.44/run01/013.vtu")
        # new_concen = numpy_support.numpy_to_vtk(p)
        # data.GetPointData().GetArray(0).SetArray(new_concen,new_concen.GetSize(),1)
        # data.GetPointData().GetArray(0)
        # # plt.hist(p,bins=100)
        # # plt.show()
        # # print(len(p))
        # # print(data)

        # # print(min(p),max(p))
        # writer = vtkXMLUnstructuredGridWriter()
        # writer.SetFileName("to_self.vtu")
        # writer.SetInputData(data)
        # writer.Write()

        # sort = np.argsort(p)
        # print(sort)

        # data = data_reader(data_path + "/0.44/run01/013.vtu")
        # coord = numpy_support.vtk_to_numpy(data.GetPoints().GetData())
        # print(coord[118901])
        # data = data_reader(data_path + "/0.44/run01/010.vtu")
        # coord = data.GetPoints().GetData()
        # xyz = coord.GetTuple(118901)
        # print(xyz)

        # data = data_reader("new.vtu")
        # concen = numpy_support.vtk_to_numpy(data.GetPointData().GetArray(0))
        # print(np.where(concen>350))
        # show("to_self.vtu")
        # show(data_path + "/0.44/run01/010.vtu")
