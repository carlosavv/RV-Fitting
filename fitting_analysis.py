import numpy as np 
import matplotlib.pyplot as plt 
from geomdl.visualization import VisPlotly as vis
from geomdl import construct
from geomdl import exchange
from test_fitRemappedRV import fit_Remapped_RV
from fitStandardRV import fit_Standard_RV
plt.style.use('seaborn')
from scipy.spatial import KDTree
from scipy.optimize import minimize
from geomdl import operations

def get_min_uv():


	# input data
	rvED = np.loadtxt("transformed_RVendo_0.txt")
	sampled_rvED = np.loadtxt("standardSample_15x25_RVendo_0.csv",delimiter=',')

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



	def calc_diff(uv,x,y,z):

		u_temp = uv[0] - int(uv[0])
		v_temp = uv[1] - int(uv[1])

		if u_temp > 0:
			uv[0] = u_temp
		if u_temp < 0:
			uv[0] = 1 + u_temp
		if v_temp > 0:
			uv[1] = v_temp
		if v_temp < 0:
			uv[1] = 1 + v_temp

		return np.linalg.norm( [x - surfED.evaluate_single((uv[0],uv[1]))[0],
								y - surfED.evaluate_single((uv[0],uv[1]))[1],
								z - surfED.evaluate_single((uv[0],uv[1]))[2]] )

	uv_pts = []
	for i in range(0,len(rvED)):
		
		res = minimize(calc_diff,[0.45,0.65], args = (rvED[i,0],rvED[i,1],rvED[i,2]), method= 'SLSQP')
		uv_pts.append([res.x[0],res.x[1]])

	uv_pts = np.array(uv_pts)
	print(uv_pts)

	surface_pts = []
	for i in range(0,len(uv_pts)):
		surface_pts.append(surfED.evaluate_single((uv_pts[i,0],uv_pts[i,1])))

	surface_pts = np.array(surface_pts)
	# np.savetxt("NURBS_nodes_rv_endo0.txt",surface_pts)
	# np.savetxt("NURBS_uv_rv_endo0.txt",uv_pts)
	# print(surface_pts == rvED)

	# surfED.render(extras = plot_extras)

	# temp = np.array(surfED.evalpts[sorted_neighbors_dist_idx[0][1]])

	# surface_pts = []
	# test = np.linspace(0,1,51)
	# print(np.array(surfED.evaluate_single((0.00001,0.00001))))
	# print(temp)

	# for i in range(0,len(test)):
	#     for j in range(0,len(test)):
	#         surface_pts.append(surfED.evaluate_single((test[i],test[j])))
	#         if np.array(surfED.evaluate_single((test[i],test[j])))[0] == temp[0] and np.array(surfED.evaluate_single((test[i],test[j])))[1] == temp[1] and np.array(surfED.evaluate_single((test[i],test[j])))[2] == temp[2]:
	#             print(test[i],test[j])
	# surface_pts = np.array(surface_pts)
	# fig = plt.figure()
	# ax = plt.axes(projection = '3d')
	# ax.scatter3D(surface_pts[0,0],surface_pts[0,1],surface_pts[0,2])
	# ax.scatter(rvED[0,0],rvED[0,1],rvED[0,2])

	# plot_extras = [
	# 	dict(points=surface_pts, name="surf_pt", color="red", size=5),
	# 	dict(points=rvED, name="rv node", color="black", size=5),
	# 	dict(points=[surfED.evaluate_single((0,1)),surfED.evaluate_single((1,0)),surfED.evaluate_single((0,0)),surfED.evaluate_single((1,1))], name="surf_pt", color="white", size=5),
	# ]
	# surfED.vis = vis.VisSurface()
	# surfED.render(extras = plot_extras)

def main():
	# get_min_uv()
	
	# input data
	rvED = np.loadtxt("transformed_RVendo_0.txt")
	rvES = np.loadtxt("transformed_RVendo_20.txt")
	sampled_rvED = np.loadtxt("standardSample_15x25_RVendo_0.csv",delimiter=',')
	sampled_rvES = np.loadtxt("standardSample_15x25_RVendo_20.csv",delimiter=',')

	# the sampled data is a M X N 2D matrix
	N = 25
	M = 15

	# fit surface 
	surfED = fit_Standard_RV(N,M,sampled_rvED)
	surfED.delta = 0.02
	testED = np.array(surfED.evalpts)
	
	surfES = fit_Standard_RV(N,M,sampled_rvES)
	surfES.delta = 0.02
	testES = np.array(surfES.evalpts)
		
	uv_pts = np.loadtxt("NURBS_uv_rv_endo0.txt")

	surface_pts_ed = []
	surface_pts_es = []

	for i in range(0,len(uv_pts)):
		surface_pts_ed.append(surfED.evaluate_single((uv_pts[i,0],uv_pts[i,1])))
		surface_pts_es.append(surfES.evaluate_single((uv_pts[i,0],uv_pts[i,1])))

	surface_pts_ed = np.array(surface_pts_ed)
	surface_pts_es = np.array(surface_pts_es)

	fig = plt.figure()
	ax = plt.axes(projection = '3d')
	ax.scatter(testES[:,0],testES[:,1],testES[:,2])
	ax.scatter(surface_pts_es[:,0],surface_pts_es[:,1],surface_pts_es[:,2])
	plt.show()

if __name__ == "__main__":
	main()
