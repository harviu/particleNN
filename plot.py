from matplotlib import pyplot as plt 
import numpy as np 

l = np.load("with_latent_distance.npy")
a = np.load("without_latent_distance.npy")


plt.grid(True)
plt.plot(l[0],c="blue",marker="o")
plt.plot(a[0],c="blue",marker="^")
plt.plot(l[1],c="red",marker="o")
plt.plot(a[1],c="red",marker="^")
plt.plot(l[2],c="green",marker="o")
plt.plot(a[2],c="green",marker="^")
plt.xticks(ticks=range(30))
plt.xlabel("Time")
plt.ylabel("Distance")
plt.show()