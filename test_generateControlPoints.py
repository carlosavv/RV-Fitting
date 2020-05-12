from geomdl.shapes import surface
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
from scipy.spatial.distance import cdist
import sys
from geomdl import utilities as utils
from geomdl import NURBS
from geomdl import exchange
from geomdl import construct
from geomdl import fitting
from geomdl import control_points as cp
from geomdl.visualization import VisMPL as vis
from geomdl import compatibility as compat
from geomdl import operations
np.set_printoptions(threshold=sys.maxsize)
from slice import slice

def remove_3dpoint(points,loc):
	x = points[:, 0]
	x = np.delete(x, loc)
	y = points[:, 1]
	y = np.delete(y, loc)
	z = points[:, 2]
	z = np.delete(z, loc)
	points = np.array([x,y,z]).T
	return points

def map2cylinder(points):

	return u,v

def main():
	points = np.loadtxt("N2_RV_P0.txt")
	zmin_loc_temp = np.where(points[:, 1] == 0)[0]
	zmin_loc = zmin_loc_temp
	points = remove_3dpoint(points,zmin_loc)

	# compute radius and height of the remapped RV	
	radius = (points[:,0].max() - points[:,0].min())
	height = points[:,2].max()

	# generate the cylindrical surface
	cylinder = surface.cylinder(radius,height)

	# store the evaluated surface points in an array
	cyl_pts = np.array(cylinder.evalpts)

	# store the control points used to generate the cylindrical surface
	cyl_cpts = np.array(cylinder.ctrlpts)	
	print(height)
	print(radius)

	# plot the cylindrical surface and control points for visualization 
	fig = plt.figure()
	ax = plt.axes(projection="3d")
	ax.scatter3D(cyl_pts[:,0],cyl_pts[:,1],cyl_pts[:,2])
	ax.scatter3D(cyl_cpts[:,0],cyl_cpts[:,1],cyl_cpts[:,2])
	ax.scatter3D(points[:,0],points[:,1],points[:,2])

	a = cdist(cyl_cpts,points).min(axis=1)
	test = np.zeros((len(cyl_cpts),3))
	for i in range(0,len(cyl_cpts)):
		test[i] = (cyl_cpts[i]/np.linalg.norm(cyl_pts[i]))*a[i]
	ax.scatter3D(test[:,0],test[:,1],test[:,2])




	# test = np.insert(test,[len(test),len(test),len(test)],[test[0],test[1],test[2]],axis = 0)
	# print(test)
	# np.savetxt("cpts_test.csv", test, delimiter=",")

	# plt.show()
	# ctrlpts = exchange.import_csv("cpts_test.csv")
	# print(operations.refine_knotvector(surf, [2, 0]))
	# print(len(ctrlpts))
	# print(ctrlpts)
	# size_u = 5
	# size_v = 4
	# degree_u = 3
	# degree_v = 2

	# # Do global surface approximation
	# surf = fitting.approximate_surface(ctrlpts, size_u, size_v, degree_u, degree_v)
	# surf.delta = 0.05
	# surf.vis = vis.VisSurface()
	# surf.render()	
main()