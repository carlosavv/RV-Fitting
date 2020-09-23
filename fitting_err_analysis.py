import numpy as np 
import matplotlib.pyplot as plt 
from geomdl.visualization import VisMPL as vis
import scipy.spatial.distance as dist
from scipy.spatial.distance import cdist
plt.style.use('seaborn')

og_rv = np.loadtxt("c:/Workspace/RV-Mapping.py/transformed_N2_RV_P0.csv",delimiter = ',')

# test = np.loadtxt("N2_RV_P0.dat")


nurbs_rv = np.loadtxt("N2_RV_P0_NURBS_surf_pts.csv",delimiter = ',')
fig = plt.figure()
ax = plt.axes(projection = '3d')

ax.scatter(nurbs_rv[:,0],nurbs_rv[:,1],nurbs_rv[:,2])
ax.scatter(og_rv[:,0],og_rv[:,1],og_rv[:,2])
plt.show()

print(len(nurbs_rv))
print(len(og_rv))

# for each RV og point, find the minimum distance from the NURBS surface
min_distances = cdist(nurbs_rv,og_rv,'euclidean').min(axis=1)

err = np.sqrt(np.sum(min_distances**2)/len(min_distances))
print(err)

fig = plt.figure()
plt.plot(min_distances)
plt.show()

# distances = [] 
# for i in range(0,len(og_rv)):
	# for j in range(0,len(nurbs_rv)):
		# distances.append(nurbs_rv[j] - og_rv[i])
# print(distances)