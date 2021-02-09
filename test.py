from tools import preProcess
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np

path = "d:/Workspace/RV-Mapping/RVshape/"
rm_file = "RVendo_0"
std_file = "standardSample_RVendo_0"
remapped_RV_file = path + rm_file
standard_RV_file = path + std_file
# std_xyz = preProcess(xyz)
# std_rv_data = np.loadtxt(standard_RV_file + ".csv", delimiter=",")
rm_rv_data = np.loadtxt(remapped_RV_file + ".txt")

path = "d:/Workspace/RV-Mapping/RVshape/"
rm_file = "RVendo_20"
std_file = "standardSample_RVendo_0"
remapped_RV_file = path + rm_file
standard_RV_file = path + std_file
# std_xyz = preProcess(xyz)
# std_rv_data = np.loadtxt(standard_RV_file + ".csv", delimiter=",")
rm_rv_data1 = np.loadtxt(remapped_RV_file + ".txt")

X = preProcess(rm_rv_data)
Y = preProcess(rm_rv_data1)

fig = plt.figure()
ax = plt.axes(projection = '3d')

ax.scatter(X[:,0],X[:,1],X[:,2], color ='b')
ax.scatter(Y[:,0],Y[:,1],Y[:,2],color ='r')
plt.show()