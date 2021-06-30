from process_data import *

import random
import os
import argparse
import pickle
import time

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

from model.pointnet import PointNet as AE
from process_data import *
from train import inference

from scipy.spatial.ckdtree import cKDTree
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans,DBSCAN
import pandas 
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from simple import show
from mean_shift import LatentRetriever


class Node():
    def __init__(self,value):
        self.left = None
        self.right = None
        self.value = value

def traverse(node:Node):
    idx = node.value
    res[idx] = node.idx
    if node.left is not None:
        traverse(node.left)
    if node.right is not None:
        traverse(node.right)
    
def hierk(node:Node,level=0):
    idx = node.value
    node.level=level
    if len(idx)<20000:
        return node
    else:
        km = KMeans(2,n_init=10,n_jobs=-1)
        res = km.fit_predict(latent[idx])
        n1 = Node(idx[res==0])
        n2 = Node(idx[res==1])
        node.left = hierk(n1,level+1)
        node.right = hierk(n2,level+1)
        return node

def vis_latent(latent,model,filename,r=0.05,dim=50):
    # show the latent block
    x = np.linspace(-0.05, 0.05, dim)
    y = np.linspace(-0.05, 0.05, dim)
    z = np.linspace(-0.05, 0.05, dim)
    xyz = np.meshgrid(x,y,z)
    xyz = np.stack([xyz[0].flatten(),xyz[1].flatten(),xyz[2].flatten()],axis=-1)
    xyz = xyz[None,:,:]
    # xyz = np.random.rand(400,3)
    # xyz = (xyz-0.5) * (args.r/0.5)
    # condition = np.sum((xyz * xyz),axis=-1) < args.r * args.r
    # xyz = xyz[condition][None,:,:]
    xyz = torch.from_numpy(xyz).cuda().float()

    with torch.no_grad():
        model.eval()
        if not torch.is_tensor(latent):
            latent = torch.from_numpy(latent[None,:]).cuda().float()
        signal = model.decode(latent,xyz)
        pc = torch.cat([xyz,signal],dim=-1)
        pc = pc.cpu().numpy()[0]
    vtk_write_image(dim,dim,dim,signal.cpu().numpy()[0],filename)

def tsne_project(embedding, tsne_projected):
    fig, ax = plt.subplots()
    for e in range(np.max(embedding)+1):
        ax.scatter(tsne_projected[embedding==e][:,0],tsne_projected[embedding==e][:,1],label=e)
    ax.legend()
    plt.show()

def reconstruction(pd,model):
    loader = DataLoader(pd, batch_size=32, shuffle=False, drop_last=False)
    r = pd.r
    model.eval()
    with torch.no_grad():
        for i, d in enumerate(loader):
            data = d[0][:,:,:args.dim].float().cuda()
            mask = d[1].cuda()

            latent = model.encode(data) 
            xyz = data[:,:,:3]

            recon = model.decode(latent,xyz)
            
            torch.set_printoptions(precision=4,sci_mode =False)
            for i in range(len(data)):
                pc = data[i,:mask[i]].cpu().detach()
                pc2 = recon[i,:mask[i]].cpu().detach()
                xyz = xyz[i,:mask[i]].cpu().detach()

                vis_latent(latent[i][None,:],model,"lr41_25_volume.vti",r)
                array_dict = {
                        "concentration": pc[:,3],
                    }
                vtk_data = numpy_to_vtk(xyz,array_dict)
                vtk_write(vtk_data,"lr41_25_gt.vtu")
                
                array_dict = {
                        "concentration": pc2[:,0],
                    }
                vtk_data = numpy_to_vtk(xyz,array_dict)
                vtk_write(vtk_data,"lr41_25_recon.vtu")
                
                fig = plt.figure()
                ax = fig.add_subplot(121, projection='3d')
                ax.title.set_text("original")
                ax2 = fig.add_subplot(122, projection='3d')
                ax2.title.set_text("recon")
                vmax = max(np.max(pc[:,3].numpy()),np.max(pc2[:,0].numpy()))
                print(np.max(pc[:,3].numpy()),np.max(pc2[:,0].numpy()))
                ax.scatter(pc[:,0],pc[:,1],pc[:,2],c=pc[:,3],vmin=0,vmax=vmax)
                ax2.scatter(pc[:,0],pc[:,1],pc[:,2],c=pc2[:,0],vmin=0,vmax=vmax)
                plt.show()



if __name__ == "__main__":
    res = np.load("eth_predict.npy")
    vtk_write_image(115,116,134,res[:,1],"predict.vti")
    print(res.shape)
    # rho = res[:,0]
    # s = res[:,1]
    # plt.imshow(s.reshape(134,116,115)[:,:,57])
    # plt.show()
    # data_path = os.environ['data']
    # mode = "cos"
    # # IoU_list = []
    # # loss_list = []
    # # for i in range(2,100,1):
    # #     print(i)
    # i = 49
    # if mode == "cos":
    #     data_directory = data_path + "/2016_scivis_fpm/0.44/run41/025.vtu"
    #     # state_dict_directory = "states_saved/fpm_knn128_dim7_vec64_CP35.pth"
    #     state_dict_directory = "states/CP10.pth"
    #     data = vtk_reader(data_directory)
    # else:
    #     if i == 100:
    #         data_directory = data_path + '/ds14_scivis_0128/raw/ds14_scivis_0128_e4_dt04_0.{:02d}00'.format(i)
    #     else:
    #         data_directory = data_path + '/ds14_scivis_0128/raw/ds14_scivis_0128_e4_dt04_0.{:02d}00'.format(i)
    #     state_dict_directory = "states_saved/cos_k64_v256/CP29.pth"
    #     # state_dict_directory = "states_saved/cos_k128_v256/CP16.pth"
    #     # state_dict_directory = "states_saved/cos_k128_v512/CP35.pth"
    #     # state_dict_directory = "states_saved/cos_k128_v768/CP13.pth"
    #     # state_dict_directory = "states_saved/cos_k256_v512/CP20.pth"
    #     # state_dict_directory = "states_saved/cos_k64_v512/CP35.pth"
    #     # state_dict_directory = "states_saved/cos_k32_v256/CP35.pth"
    #     halo_directory = data_path + '/ds14_scivis_0128/rockstar/out_{}.list'.format(i-2)
    #     data = sdf_reader(data_directory)
    
    # state_dict = torch.load(state_dict_directory)
    # state = state_dict['state']
    # args = state_dict['config']
    # print(args)
    # model = AE(args,256).float().cuda()
    # model.load_state_dict(state)
    # if args.have_label:
    #     hp = halo_reader(halo_directory)
    #     pd = PointData(data,args,np.arange(len(data)),hp)
    #     latent,predict,loss = inference(pd,model,1000,args)
    #     label = pd.label
    #     # torch.save(latent,"cos_latent_middle49")
    #     # torch.save(predict,"predict")
    # else:
    #     dim = 30
    #     x = np.linspace(-5, 5, dim)
    #     y = np.linspace(-5, 5, dim)
    #     z = np.linspace(0, 10, dim)
    #     xyz = np.meshgrid(x,y,z)
    #     xyz = np.stack([xyz[0].flatten(),xyz[1].flatten(),xyz[2].flatten()],axis=-1)
    #     cond = xyz[:,0]**2+xyz[:,1]**2 < 25
    #     xyz = xyz[cond]
    #     pd = PointData(data,args,xyz)
    #     # pd = PointData(data,args,np.arange(len(data)))
    #     loader = DataLoader(pd, batch_size=1000, shuffle=False, drop_last=False)
        # # major_axis = np.zeros((len(data),4))
        # # for i,(d,m) in enumerate(loader):
        # #     d = d[0,:m,:4]
        # #     pca = PCA()
        # #     pca.fit(d)
        # #     major_axis[i] = pca.components_[0]
        # #     print(i,end="\r")
        # # print(major_axis.shape)
        # # np.save("major_axis",major_axis)

        # # full = np.zeros((len(pd),256*3))
        # # for i, d in enumerate(loader):
        # #     if i == len(loader) -1:
        # #         full[i*32:] = d[0][:,:,:3].reshape(-1,768)
        # #     else:
        # #         full[i*32:(i+1)*32] = d[0][:,:,:3].reshape(32,-1)
        # # print(full.shape)
        # latent = inference(pd,model,1500,args)
        # print(latent.shape)
        # torch.save(latent,"fpm_latent_41_25_geoconv")

    

    ################# analysis ##################
    # data_directory = 'D:\\OneDrive - The Ohio State University\\data/ds14_scivis_0128/raw/ds14_scivis_0128_e4_dt04_0.4900'
    # halo_directory = 'D:\\OneDrive - The Ohio State University\\data/ds14_scivis_0128/rockstar/out_47.list'
    # hp = halo_reader(halo_directory)
    # halo_position, halo_radius = hp
    # data = sdf_reader(data_directory)
    # coord = data[:,:3]
    # kd = cKDTree(coord,leafsize=100)
    # label = np.zeros(len(data))
    # for i in range(len(halo_position)):
    #     positive = kd.query_ball_point(halo_position[i],halo_radius[i])
    #     label[positive] = 1
    # emb = np.load("./results/cosmology/kmeans_49.npy")
    # cond = np.array(emb==3) 
    # print(np.sum(label[cond])/np.sum(label))

    # latent = torch.load("./results/cosmology/latent_all_49")
    # print(latent.shape)
    # km = KMeans(4)
    # emb = km.fit_predict(latent)
    # np.save("./results/cosmology/kmeans_49",emb)
    # array_dict = {
    #     "predict": emb,
    #     "label": label,
    #     "phi":data[:,-1],
    #     "velocity":data[:,3:6],
    #     "acceleration":data[:,6:9],
    # }
    # vtk_data = numpy_to_vtk(data[:,:3],array_dict)
    # vtk_write(vtk_data,"test_cos.vtu")

    ################# recon_all ################
    # model.eval()
    # num = len(pd) * 256
    # coord_list = np.zeros((num,3))
    # attr_list = np.zeros((num,))
    # count = 0
    # batch_count = 0
    # mse = 0
    # with torch.no_grad():
    #     for i, d in enumerate(loader):
    #         data = d[0][:,:,:args.dim].float().cuda()
    #         mask = d[1].cuda()
    #         recon_batch = model(data)

    #         centers = pd.center[batch_count:batch_count+len(mask)]
    #         batch_count += len(mask)
    #         for b in range(len(mask)):
    #             m = mask[b]
    #             coord = data[b,:m,:3]
    #             attr = recon_batch[b,:m,0]
    #             mse += torch.sum((attr - data[b,:m,3]) ** 2)
    #             center = centers[b][None,:]
    #             coord_list[count:count+m] = coord.cpu().detach().numpy() + center
    #             attr_list[count:count+m] = attr.cpu().detach().numpy()
    #             count += m
    #         print(count)
    #     mse /= count
    #     print(mse)
    #     coord_list = coord_list[:count]
    #     attr_list = attr_list[:count]
    #     coord_list *= 10
    #     coord_list[:,0] -= 5
    #     coord_list[:,1] -= 5
    #     attr_list *= 357.19
    #     np.save("coord",coord_list)
    #     np.save("attr",attr_list)

    # array_dict = {
    #         "concentration": attr_list,
    #     }
    # vtk_data = numpy_to_vtk(coord_list,array_dict)
    # vtk_write(vtk_data,"recon.vtu")

    # coord = np.load("coord.npy")
    # attr = np.load("attr.npy")
    # coord = torch.from_numpy(coord).cuda()
    # attr = torch.from_numpy(attr).cuda()

    # r_coord = data[:,:3]
    # r_coord = torch.from_numpy(r_coord).cuda()
    # r_attr = data[:,3]
    # r_attr = torch.from_numpy(r_attr).cuda()
    # print(torch.max(r_attr),torch.max(attr))

    # new_attr = torch.zeros_like(r_attr)

    # for i, c in enumerate(r_coord):
    #     co = coord - c[None,:]
    #     distance = torch.sum(torch.abs(co), dim=-1)
    #     near_coord = torch.where(distance < 0.001)
    #     new_attr[i] = torch.mean(attr[near_coord])
    #     # print(new_attr[i],r_attr[i])
    #     print(i)
    # mse = torch.mean( ((new_attr - r_attr) ** 2) )
    # print(mse)

    # array_dict = {
    #         "concentration": new_attr.cpu().numpy(),
    #     }
    # vtk_data = numpy_to_vtk(data[:,:3],array_dict)
    # vtk_write(vtk_data,"recon.vtu")



    # predict = np.argmax(predict,1)
    # IoU_value = IoU(predict,label)
    # sub = predict + label
    # IoU_list.append(IoU_value)
    # loss_list.append(loss)
    # np.save("loss_list",loss_list)
    # np.save("iou_list",IoU_list)


    # for i in range(5):
    #     c0 = c
    #     c0[0] += i/10
    #     print(c0)
    #     with torch.no_grad():
    #         model.eval()
    #         points = torch.from_numpy(c[None,:]).cuda().float()
    #         signal = model.decode(points,xyz)
    #         pc = torch.cat([xyz,signal],dim=-1)
    #         pc = pc.cpu().numpy()[0]
    #     vtk_write_image(dim,dim,dim,signal.cpu().numpy()[0],"test%d.vti" % i)
    


    # fig = plt.figure()
    # # ax = fig.add_subplot(121, projection='3d')
    # # ax.title.set_text("original")
    # ax2 = fig.add_subplot(111, projection='3d')
    # ax2.title.set_text("recon")
    # # ax.scatter(pc[:,0],pc[:,1],pc[:,2],c=pc[:,3])
    # ax2.scatter(pc[:,0],pc[:,1],pc[:,2],c=pc[:,3])
    # plt.show()

    ############ PCA and histogram #############
    # pd = PointData(data,args,np.arange(len(data)))
    # pca_latent = np.zeros((len(data),4,4))
    # for i,d in enumerate(pd):
    #     print(i)
    #     mask = d[1]
    #     pc = d[0]
    #     pc = pc[:mask,:4]
    #     pca = PCA()
    #     pca.fit(pc)
    #     pca_lat = pca.components_
    #     pca_latent[i] = pca_lat
    #     print(pca.explained_variance_)
    # np.save("pca_latent",pca_latent)
    # pca_latent = np.load("pca_latent.npy")
    # # pca_latent = pca_latent[:,0,:]
    # pca_latent = pca_latent.reshape(-1,16)
    # kmeans = KMeans(6)
    # pca_id = kmeans.fit_predict(pca_latent)
    # latent = torch.load("fpm_latent_41_25_geoconv")
    # km2 = KMeans(6)
    # embedding = km2.fit_predict(latent)
    # print(embedding.shape,embedding.shape)
    # array_dict = {
    #         "pca": pca_id,
    #         # "mean": mean_neighbor,
    #         "embedding": embedding,
    #         "concentration": data[:,3],
    #         "velocity": data[:,4:]
    #     }
    # vtk_data = numpy_to_vtk(data[:,:3],array_dict)
    # vtk_write(vtk_data,"test_pca.vtu")
    ############ grid reconstruction ##############
    # xyz = [[-2.31,1.71,5.7]]
    # pd = PointData(data,args,xyz)
    # reconstruction(pd,model)
    # exit()


    ############# calculate k-means and projection ############
    # latent = torch.load("block_latent_25_geoconv")
    # print(latent.shape)
    # km = KMeans(6,n_init=10,n_jobs=-1)
    # embedding = km.fit_predict(latent)
    # centers = km.cluster_centers_
    # np.save("centers",centers)
    # np.save("emb_grid",embedding)
    # tsne = TSNE(n_jobs=-1)
    # new = tsne.fit_transform(latent)
    # np.save("tsne_projected",new)

    ########## show the grid projection   #####
    # tsne_projected = np.load("tsne_projected.npy")
    # embedding = np.load("emb_grid.npy")
    # print(tsne_projected.shape,embedding.shape)
    # tsne_project(embedding,tsne_projected)

    ########## vis latent centeres ##########
    # centers = np.load("centers.npy")
    # print(centers.shape)
    # for i in range(len(centers)):
    #     c = centers[i]
    #     vis_latent(c,model,"cluster_%d.vti" % i)

    ###### output actual grid class ###########
    # dim = 30
    # x = np.linspace(1/dim, 1, dim)
    # y = np.linspace(1/dim, 1, dim)
    # z = np.linspace(1/dim, 1, dim)
    # xyz = np.meshgrid(x,y,z)
    # xyz = np.stack([xyz[0].flatten(),xyz[1].flatten(),xyz[2].flatten()],axis=-1)
    # cond = (xyz[:,0]-0.5)**2+(xyz[:,1]-0.5)**2 < 0.25
    # xyz = xyz[cond]
    # embedding = np.load("emb_grid.npy")
    # xyz *= 10
    # xyz[:,0]-=5
    # xyz[:,1]-=5
    # print(xyz.shape,embedding.shape)
    # print(embedding.shape)
    # if mode=="cos":
    #     array_dict = {
    #         "predict": predict,
    #         "label": label,
    #         "sub": sub,
    #         "phi":data[:,-1],
    #         "velocity":data[:,3:6],
    #         "acceleration":data[:,6:9],
    #     }
    # else:
    #     array_dict = {
    #         # "pca": pca_out,
    #         # "mean": mean_neighbor,
    #         "embedding": embedding,
    #         # "concentration": data[:,3],
    #         # "velocity": data[:,4:]
    #     }
    # # vtk_data = numpy_to_vtk(data[:,:3],array_dict)
    # vtk_data = numpy_to_vtk(xyz,array_dict)
    # vtk_write(vtk_data,"test_grid.vtu")

    # hp = halo_reader(halo_directory)
    # print(hp)
    # halo_writer(hp[0],hp[1],"halo49.vtu")

    # cluster_centers = np.load("cluster_center.npy")[3]
    # distance = np.sum((new_cluster - cluster_centers) ** 2,-1)
    # interested_cluster = np.argmin(distance)
    # print(distance,interested_cluster)

    ############### DBSCAN #############
    # cluster_id = embedding
    # data = data[cluster_id==interested_cluster]
    # # print(data.shape)
    # db = DBSCAN(0.44,30)
    # res2 = db.fit_predict(data[:,:3])
    # res2 = res2.astype(np.int)
    # print(np.max(res2))

    # array_dict = {
    #     # "pca": pca_output,
    #     # "mean": mean_neighbor,
    #     "embedding": res2,
    #     "concentration": data[:,3],
    #     "velocity": data[:,4:]
    # }
    # vtk_data = numpy_to_vtk(data[:,:3],array_dict)
    # vtk_write(vtk_data,"result_overview.vtu")


    ############### parallel coordinates #############
    # lat = np.concatenate((latent,embedding[:,None]),1)
    # df = pandas.DataFrame(data=lat[::100])
    # plt.figure(figsize=(9,4))
    # plt.xlabel('Embedding Dimension')
    # plt.ylabel('Value')
    # pandas.plotting.parallel_coordinates(df,class_column=args.vector_length,color=('#3985ad'))
    # legend = plt.legend()
    # legend.remove()
    # plt.show()

    ################ tsne #################
    # tsn = TSNE(2)
    # d_latent = tsn.fit_transform(latent)
    # np.save("tsne_latent",d_latent)
    # d_latent = np.load("tsne_latent.npy")
    # plt.scatter(d_latent[:,0],d_latent[:,1],c = res,marker='.')
    # plt.show()

    #################### Hierarchical ####################
    # root = Node(np.arange(len(latent)))
    # hierk(root)
    # node_list = [root]
    # save_idx = []
    # idx = 0
    # while(len(node_list)>0):
    #     node = node_list.pop(0)
    #     if node is not None:
    #         print(idx,node.level)
    #         if node.level==5 and (idx==16 or idx==18):
    #             save_idx+=list(node.value)
    #         # sub_data = data[node.value]
    #         # array_dict = {
    #         #     "concentration":sub_data[:,3],
    #         #     "velocity":sub_data[:,4:],
    #         # }
    #         # vtk_data = numpy_to_vtk(sub_data[:,:3],array_dict)
    #         # show(vtk_data,outfile="{}_{}".format(node.level,idx))
    #         node.idx = idx
    #         node_list.append(node.left)
    #         node_list.append(node.right)
    #     idx += 1
    # # traverse(root)

    ########## convert to density #############
    # coord = data[:,:3]
    # kd = cKDTree(coord,leafsize=100)
    # x = np.linspace(-5,5,64)
    # y = np.linspace(-5,5,64)
    # z = np.linspace(0,10,64)
    # xv,yv,zv = np.meshgrid(x,y,z)
    # xv = xv.reshape(-1)
    # yv = yv.reshape(-1)
    # zv = zv.reshape(-1)
    # idx = np.array([xv,yv,zv]).T
    # nn = kd.query_ball_point(idx,r=0.4,n_jobs=-1)
    # density = np.zeros((len(idx)))
    # print(density.shape)
    # for i,n in enumerate(nn):
    #     if len(n) > 1 :
    #         center = idx[i]
    #         n = data[n]
    #         dis = np.sqrt(np.sum((n[:,:3]-center[:3])**2,-1))
    #         weight = 1- 1/(dis + 1e-8)
    #         density[i] = np.average(n,0,weight)[3]
    #         print(i)
    # vtk_data = vtk.vtkImageData()
    # vtk_data.SetDimensions(64,64,64)
    # scaler = numpy_support.numpy_to_vtk(density)
    # vtk_data.GetPointData().AddArray(scaler)
    # writer = vtk.vtkXMLImageDataWriter()
    # writer.SetFileName("test.vti")
    # writer.SetInputData(vtk_data)
    # writer.Write()