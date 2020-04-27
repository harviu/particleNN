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
for k in center:
    print(k[0])

vtk_data = vtkUnstructuredGrid()
vtk_data.SetPoint