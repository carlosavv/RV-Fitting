import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from slice import slice
import sys
np.set_printoptions(threshold=sys.maxsize)
from slice import segment


points = np.loadtxt("N2_RV_P0.txt")
# points = preProcess(points)
x = points[:, 0]
y = points[:, 1]
z = points[:, 2]


N = 6
slice(N, points)
slices = []
temp = []

layers = []
bins = slice.bins
for j in range(0,len(slice.slices)):
	layers.append([slice.slices[j][:,0],slice.slices[j][:,1],slice.slices[j][:,2]])


layers = np.array(layers)
# for i in range(0,len(temp)):
# 	for j in range(0,len(temp[i][0])):
# 		layers.append((temp[:,0][i][j],temp[:,1][i][j],temp[:,2][i][j]))

# print(layers)
fig = plt.figure()
ax = plt.axes(projection="3d")
# ax.scatter(layers[-1][0],layers[-1][1],layers[-1][2])

plt.show()

