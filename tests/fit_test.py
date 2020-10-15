from geomdl.shapes import surface
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from geomdl.visualization import VisMPL
import numpy as np
from scipy.spatial.distance import cdist
import sys
from geomdl import utilities as utils
import sys
from geomdl import NURBS
from geomdl import exchange
from geomdl import construct
from geomdl import fitting
from geomdl import control_points as cp
from geomdl.visualization import VisMPL as vis
from geomdl import compatibility as compat
np.set_printoptions(threshold=sys.maxsize)

def slice(n, points):
	x = points[:, 0]
	y = points[:, 1]
	z = points[:, 2]
	fig = plt.figure()
	ax = plt.axes(projection="3d")
	ax.scatter3D(x, y, z) 	

	bottom_bound = points[:, 2].min()
	top_bound = points[:, 2].max()
	slice.bins = np.linspace(bottom_bound, top_bound, n + 1)
	bins = slice.bins
	slice.slices = []
	slices = slice.slices

	for ii in range(len(bins) - 1):
		slices.append(points[(points[:, 2] > bins[ii]) & (points[:, 2] < bins[ii + 1])])
		if ii == range(len(bins)):
			slices.append(points[(points[:, 2] == bins[ii+1])])

	fig4 = plt.figure()
	ax3 = plt.axes(projection="3d")
	for i in range(len(slices)):
		ax3.scatter(slices[i][:, 0], slices[i][:, 1], slices[i][:, 2])

def remove_3dpoint(points,loc):
	x = points[:, 0]
	x = np.delete(x, loc)
	y = points[:, 1]
	y = np.delete(y, loc)
	z = points[:, 2]
	z = np.delete(z, loc)
	points = np.array([x,y,z]).T
	return points

def main():
	points = np.loadtxt("N2_RV_P0.txt")

	zmin_loc_temp = np.where(points[:, 1] == 0)[0]
	zmin_loc = zmin_loc_temp

	# points = remove_3dpoint(points,zmin_loc)
	N = 5
	slice(N, points)

	slice0 = slice.slices[0]
	x0 = slice0[:,0]
	y0 = slice0[:,1]
	z0 = slice0[:,2]
	slice1 = slice.slices[1]
	x1 = slice1[:,0]
	y1 = slice1[:,1]
	z1 = slice1[:,2]
	slice2 = slice.slices[2]
	x2 = slice2[:,0]
	y2 = slice2[:,1]
	z2 = slice2[:,2]
	slice3 = slice.slices[3]
	x3 = slice3[:,0]
	y3 = slice3[:,1]
	z3 = slice3[:,2]
	slice4 = slice.slices[4]
	x4 = slice4[:,0]
	y4 = slice4[:,1]
	z4 = slice4[:,2]
	print(np.shape(points))
	ranges = slice.bins

	diameter = x0.max()-x0.min()
	radius = diameter
	height = z0.max()

	diameter1 = x1.max()-x1.min()
	radius1 = diameter1
	height1 = z1.max()

	diameter2 = x2.max()-x2.min()
	radius2 = diameter2
	height2 = z2.max()

	diameter3 = x3.max()-x3.min()
	radius3 = diameter3
	height3 = z3.max()
	

	diameter4 = x4.max()-x4.min()
	radius4 = diameter4/2
	height4 = x4.max()
	
	# Generate a cylindrical surface using the shapes component	
	cylinder = surface.cylinder(radius, height )
	# vis_config = VisMPL.VisConfig(ctrlpts=True)
	# vis_comp = VisMPL.VisSurface(config=vis_config)
	# cylinder.vis = vis_comp
	# cylinder.render()
	cyl_points = cylinder.evalpts
	print("*****************")
	# print(len(cyl_points))
	print("*****************")

	cylinder1 = surface.cylinder(radius1, height1)
	cyl_points1 = cylinder1.evalpts
	# vis_config = VisMPL.VisConfig(ctrlpts=True)
	# vis_comp = VisMPL.VisSurface(config=vis_config)
	# cylinder1.vis = vis_comp
	# cylinder1.render()

	cylinder2 = surface.cylinder(radius2, height2)
	cyl_points2 = cylinder2.evalpts
	# vis_config = VisMPL.VisConfig(ctrlpts=True)
	# vis_comp = VisMPL.VisSurface(config=vis_config)
	# cylinder2.vis = vis_comp
	# cylinder2.render()

	cylinder3 = surface.cylinder(radius3, height3 )
	cyl_points3 = cylinder3.evalpts
	# vis_config = VisMPL.VisConfig(ctrlpts=True)
	# vis_comp = VisMPL.VisSurface(config=vis_config)
	# cylinder2.vis = vis_comp
	# cylinder2.render()

	cylinder4 = surface.cylinder(radius4, height4)
	cyl_points4 = cylinder4.evalpts
	# vis_config = VisMPL.VisConfig(ctrlpts=True)
	# vis_comp = VisMPL.VisSurface(config=vis_config)
	# cylinder4.vis = vis_comp
	# cylinder4.render()


	#try using points from the surface not the control points
	cylinder_cpts = np.array(cylinder.ctrlpts)
	cylinder_cpts1 = np.array(cylinder1.ctrlpts)
	cylinder_cpts2 = np.array(cylinder2.ctrlpts)
	cylinder_cpts3 = np.array(cylinder3.ctrlpts)
	cylinder_cpts4 = np.array(cylinder4.ctrlpts)

	a = np.array(cdist(np.array(cyl_points),slice0)).min(axis=0) 	
	b = np.array(cdist(np.array(cyl_points1),slice1)).min(axis=0)
	c = np.array(cdist(np.array(cyl_points2),slice2)).min(axis=0)
	d = np.array(cdist(np.array(cyl_points3),slice3)).min(axis=0)
	e = np.array(cdist(np.array(cyl_points4),slice4)).min(axis=0)

	test = np.zeros((len(cylinder_cpts),3))
	test1 = np.zeros((len(cylinder_cpts1),3))
	test2 = np.zeros((len(cylinder_cpts2),3))
	test3 = np.zeros((len(cylinder_cpts3),3))
	test4 = np.zeros((len(cylinder_cpts4),3))

	for i in range(0,len(cylinder_cpts)):
		test[i] = (cylinder_cpts[i]/np.linalg.norm(cylinder_cpts[i]))*a[i] 
		test1[i] = (cylinder_cpts1[i]/np.linalg.norm(cylinder_cpts1[i]))*b[i]
		test2[i] = (cylinder_cpts2[i]/np.linalg.norm(cylinder_cpts2[i]))*c[i]
		test3[i] = (cylinder_cpts3[i]/np.linalg.norm(cylinder_cpts3[i]))*d[i]
		test4[i] = (cylinder_cpts4[i]/np.linalg.norm(cylinder_cpts4[i]))*e[i]

	test_zeros_loc = np.where(test[:, 2] == 0)[0]
	test1_zeros_loc = np.where(test1[:, 2] == 0)[0]
	test2_zeros_loc = np.where(test2[:, 2] == 0)[0]
	test3_zeros_loc = np.where(test3[:, 2] == 0)[0]
	test4_zeros_loc = np.where(test4[:, 2] == 0)[0]

	test = remove_3dpoint(test,test_zeros_loc)
	test1 = remove_3dpoint(test1,test1_zeros_loc)
	test2 = remove_3dpoint(test2,test2_zeros_loc)
	test3 = remove_3dpoint(test3,test3_zeros_loc)
	test4 = remove_3dpoint(test4,test4_zeros_loc)

	test = np.array([test[:,0],test[:,1],np.ones(len(test[:,2]))*ranges[0]]).T
	test1 = np.array([test1[:,0],test1[:,1],np.ones(len(test1[:,2]))*ranges[1]]).T
	test2 = np.array([test2[:,0],test2[:,1],np.ones(len(test2[:,2]))*ranges[2]]).T
	test3 = np.array([test3[:,0],test3[:,1],np.ones(len(test3[:,2]))*ranges[3]]).T
	test4 = np.array([test4[:,0],test4[:,1],np.ones(len(test4[:,2]))*ranges[4]]).T

	# for i in range(0,len(test1)):
		# test1[i] = test1[i] + [0,0,height]
		# test2[i] = test2[i]+[0,0,height1]
		# test3[i] = test3[i]+[0,0,height2]

	test = remove_3dpoint(test,len(test)-1)	
	test1 = remove_3dpoint(test1,len(test1)-1)
	test2 = remove_3dpoint(test2,len(test2)-1)
	test3 = remove_3dpoint(test3,len(test3)-1)
	test4 = remove_3dpoint(test4,len(test4)-1)

	# test = np.insert(test,[0,len(test)],[[0,0,-5],[0,0,ranges[0]]],axis=0)
	# test1 = np.insert(test1,[0,len(test1)],[0,0,ranges[1]],axis=0)
	# test2 = np.insert(test2,[0,len(test2)],[0,0,ranges[2]],axis=0)
	test = np.insert(test,[len(test)-2,len(test)-1,len(test)],[test[0],test[1],test[2]],axis=0)
	test1 = np.insert(test1,[len(test1)-2,len(test1)-1,len(test1)],[test1[0],test1[1],test1[2]],axis=0)
	test2 = np.insert(test2,[len(test2)-2,len(test2)-1,len(test2)],[test2[0],test2[1],test2[2]],axis=0)
	test3 = np.insert(test3,[len(test3)-2,len(test3)-1,len(test3)],[test3[0],test3[1],test3[2]],axis=0)
	test = np.insert(test4,[len(test4)-2,len(test4)-1,len(test4)],[test4[0],test4[1],test4[2]],axis=0)


	# print(test)
	# print(test1) 
	# print(test2)
	# print(test3)

	X = np.row_stack((test,test1,test2,test3,test4))
	print(X)
	# np.random.shuffle(X)

	np.savetxt("cpts_test.csv", X, delimiter=",")
	# np.savetxt("cpts_test1.csv", test1, delimiter=",")
	fig = plt.figure()
	ax = plt.axes(projection="3d")
	ax.scatter3D(test[:,0],test[:,1],np.ones(len(test[:,2]))*ranges[0],color = 'red')
	ax.scatter3D(test1[:,0],test1[:,1],test1[:,2],color = 'blue')
	ax.scatter3D(test2[:,0],test2[:,1],test2[:,2],color = 'green')
	ax.scatter3D(test3[:,0],test3[:,1],test3[:,2],color = 'yellow')
	ax.scatter3D(test4[:,0],test4[:,1],test4[:,2],color = 'purple')
	# ax.scatter3D(cylinder_cpts[:,0],cylinder_cpts[:,1],cylinder_cpts[:,2])
	# ax.scatter3D(cylinder_cpts1[:,0],cylinder_cpts1[:,1],cylinder_cpts1[:,2])

	# try fitting a NURBS Surface
	surf = NURBS.Surface()
	surf.delta = 0.03
	p_ctrlpts = exchange.import_csv("cpts_test.csv")
	print(len(p_ctrlpts))
	p_weights = np.ones((len(p_ctrlpts)))
	p_size_u = 9
	p_size_v = 5
	p_degree_u = 3
	p_degree_v = 3
	t_ctrlptsw = compat.combine_ctrlpts_weights(p_ctrlpts, p_weights)
	n_ctrlptsw = compat.flip_ctrlpts_u(t_ctrlptsw, p_size_u, p_size_v)
	n_knotvector_u = utils.generate_knot_vector(p_degree_u, p_size_u)
	n_knotvector_v = utils.generate_knot_vector(p_degree_v, p_size_v)

	surf.degree_u = p_degree_u
	surf.degree_v = p_degree_v
	surf.set_ctrlpts(n_ctrlptsw, p_size_u, p_size_v)
	surf.knotvector_u = n_knotvector_u
	surf.knotvector_v = n_knotvector_v
	vis_config = vis.VisConfig(ctrlpts=True, axes=True, legend=True)
	surf.vis = vis.VisSurface(vis_config)
	surf.evaluate()
	surf.render()
	
	#try fitting a surface
	# size_u = 7
	# size_v = 5
	# degree_u = 3
	# degree_v = 3

	# # Do global surface interpolation
	# surf = fitting.approximate_surface(points, size_u, size_v, degree_u, degree_v)

	# # Plot the interpolated surface
	# surf.delta = 0.05
	# surf.vis = vis.VisSurface()
	# surf.render()
main()
