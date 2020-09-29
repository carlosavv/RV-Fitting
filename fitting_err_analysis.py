import numpy as np 
import matplotlib.pyplot as plt 
from geomdl.visualization import VisMPL as vis
import scipy.spatial.distance as dist
from scipy.spatial.distance import cdist
plt.style.use('seaborn')

og_rv_ed = np.loadtxt("d:/Workspace/RV-Mapping.py/transformed_N2_RV_P0.csv",delimiter = ',')
nurbs_rv_ed = np.loadtxt("N2_RV_P0_NURBS_surf_pts.csv",delimiter = ',')

og_rv_es = np.loadtxt("d:/Workspace/RV-Mapping.py/transformed_N2_RV_P4.csv",delimiter = ',')
nurbs_rv_es = np.loadtxt("N2_RV_P4_NURBS_surf_pts.csv",delimiter = ',')
# for each RV og point, find the minimum distance from the NURBS surface
min_distances_ed = cdist(nurbs_rv_ed,og_rv_ed,'euclidean').min(axis=1)
min_distances_es = cdist(nurbs_rv_es,og_rv_es,'euclidean').min(axis=1)

rmse_ed = np.sqrt(np.sum(min_distances_ed**2)/len(min_distances_ed))
rmse_es = np.sqrt(np.sum(min_distances_es**2)/len(min_distances_es))

print("rmse_ed = ", rmse_ed)
print("rmse_es = ", rmse_es)

# fig = plt.figure()
# plt.plot(min_distances)
# plt.show()

fig = plt.figure()
ax = plt.axes(projection = '3d')

ax.scatter(nurbs_rv_ed[:,0],nurbs_rv_ed[:,1],nurbs_rv_ed[:,2],label = 'NURBS Surface Points')
ax.scatter(og_rv_ed[:,0],og_rv_ed[:,1],og_rv_ed[:,2], label = 'Standard RV Surface Points')
ax.legend()
plt.show()

fig = plt.figure()
ax = plt.axes(projection = '3d')

ax.scatter(nurbs_rv_es[:,0],nurbs_rv_es[:,1],nurbs_rv_es[:,2],label = 'NURBS Surface Points')
ax.scatter(og_rv_es[:,0],og_rv_es[:,1],og_rv_es[:,2], label = 'standard RV Surface Points')
ax.legend()
plt.show()
# distances = [] 
# for i in range(0,len(og_rv)):
	# for j in range(0,len(nurbs_rv)):
		# distances.append(nurbs_rv[j] - og_rv[i])
# print(distances)