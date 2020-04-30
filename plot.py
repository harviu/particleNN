from matplotlib import pyplot as plt 
import numpy as np 
from vtk import *
from vtk.util import numpy_support

# l = np.load("result_saved/w_3.npy")
# a = np.load("result_saved/wo_3.npy")


# plt.grid(True)
# plt.plot(l[0],c="c",marker="o",label='Halo 1, latent')
# plt.plot(a[0],c="c",marker="^",label='Halo 1, no latent')
# plt.plot(l[1],c="m",marker="o",label='Halo 2, latent')
# plt.plot(a[1],c="m",marker="^",label='Halo 2, no latent')
# plt.plot(l[2],c="green",marker="o",label='Halo 3, latent')
# plt.plot(a[2],c="green",marker="^",label='Halo 3, no latent')
# plt.xticks(ticks=range(8),labels=range(91,99))
# plt.xlabel("Time")
# plt.ylabel("Distance")
# plt.legend()
# plt.show()

center = np.load("result_saved/truth_list.npy")
# print(center.shape)
center = center.reshape(-1,3)
coords = numpy_support.numpy_to_vtk(center)
points = vtkPoints()
points.SetData(coords)
# print(coords)

vtk_data = vtkUnstructuredGrid()
vtk_data.SetPoints(points)

time = np.arange(0.12,1,0.01)
time = np.tile(time,30)
time = numpy_support.numpy_to_vtk(time)
# print(len(time))
vtk_data.GetPointData().AddArray(time)
vtk_data.GetPointData().GetArray(0).SetName("time")
print(vtk_data.GetPointData())
print(vtk_data.GetPoints())


writer = vtkXMLUnstructuredGridWriter()
writer.SetFileName("truth.vtu")
writer.SetInputData(vtk_data)
writer.Write()
    
def data_to_numpy(vtk_data):
    coord = numpy_support.vtk_to_numpy(vtk_data.GetPoints().GetData())
    concen = numpy_support.vtk_to_numpy(vtk_data.GetPointData().GetArray(0))[:,None]
    velocity = numpy_support.vtk_to_numpy(vtk_data.GetPointData().GetArray(1))
    point_data = np.concatenate((coord,concen,velocity),axis=-1)
    return point_data

# reader = vtkXMLUnstructuredGridReader()
# reader.SetFileName(r"D:\OneDrive - The Ohio State University\data\2016_scivis_fpm\0.44\run03\002.vtu")
# reader.Update()

# data = reader.GetOutput()
# print(data.GetPoints().GetData())