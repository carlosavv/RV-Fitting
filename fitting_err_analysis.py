import numpy as np 
import matplotlib.pyplot as plt 
from geomdl.visualization import VisMPL as vis
import scipy.spatial.distance as dist
plt.style.use('seaborn-bright')

og_rv = np.loadtxt("d:/Workspace/RV-Mapping.py/transformed_N2_RV_P0.csv",delimiter = ',')

test = np.loadtxt("N2_RV_P0.dat")


fig = plt.figure()
ax = plt.axes(projection = '3d')

ax.scatter(test[:,0],test[:,1],test[:,2])
ax.scatter(og_rv[:,0],og_rv[:,1],og_rv[:,2])
plt.show()
# nurbs_rv = np.loadtxt("")