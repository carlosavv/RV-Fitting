import numpy as np 
import matplotlib.pyplot as plt 
from geomdl.visualization import VisPlotly as vis
from geomdl import construct
from geomdl import exchange
from test_fitRemappedRV import fit_Remapped_RV
from fitStandardRV import fit_Standard_RV
plt.style.use('seaborn')
from scipy.spatial import KDTree

# input data
rvED = np.loadtxt("d:/Workspace/RV-Mapping/RVshape/transformed_RVendo_0.txt")
sampled_rvED = np.loadtxt("d:/Workspace/RV-Mapping/RVshape/standardSample_15x25_RVendo_0.csv",delimiter=',')

# the sampled data is a M X N 2D matrix
N = 25
M = 15

# fit surface 
surfED = fit_Standard_RV(N,M,sampled_rvED)
surfED.delta = 0.02
# assign a tree and get the distances and indices of closest neighbors 
tree = KDTree(surfED.evalpts, leafsize = rvED.shape[0] + 1)
neighbor_distances, neighbor_idx = tree.query(rvED)

# sort the result in a way where it has (minimum distance, index)
sorted_neighbors_dist_idx = sorted(zip(neighbor_distances,neighbor_idx))
surf_curves = construct.extract_curves(surfED)
plot_extras = [
    dict(points=surf_curves["u"][0].evalpts, name="u", color="red", size=5),
    dict(points=surf_curves["v"][0].evalpts, name="v", color="black", size=5),
]
surfED.vis = vis.VisSurface()
# surfED.render(extras = plot_extras)

temp = np.array(surfED.evalpts[sorted_neighbors_dist_idx[0][1]])
surface_pts = np.array(surfED.evalpts)
fig = plt.figure()
ax = plt.axes(projection = '3d')
ax.scatter3D(surface_pts[:,0],surface_pts[:,1],surface_pts[:,2])
ax.scatter(temp[0],temp[1],temp[2])
plt.show()