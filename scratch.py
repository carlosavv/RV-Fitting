import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from slice import slice

points = np.loadtxt("N2_RV_P0.txt")
# points = preProcess(points)
x = points[:, 0]
y = points[:, 1]
z = points[:, 2]


N = 6
slice(N, points)
slices = []
temp = []

# store radii at each slice
for j in range(0,len(slice.slices)):
    temp.append(cylinder(slice.slices[j][:,0],slice.slices[j][:,1],slice.slices[j][:,2]))    