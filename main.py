# imports

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from geomdl.visualization import VisMPL as vis
from slice import slice
from tools import preProcess
from geomdl import construct
import numpy as np
import sys 
from fitRemappedRV import fit_Remapped_RV
from fitStandardRV import fit_Standard_RV

sys.path.append('../')

'''
use this to fit a surface on to the remapped RV

'''

def main():

	path = "rv_data/"
	rm_file = "N2_RV_P0_rm"
	std_file = "sampled_N2_RV_P0"
	remapped_RV_file = path + rm_file
	standard_RV_file = path + std_file

	# std_xyz = preProcess(xyz)
	std_rv_data = np.loadtxt(standard_RV_file + ".csv",delimiter = ',')
	rm_rv_data = np.loadtxt(remapped_RV_file + ".csv",delimiter = '	')
	N = 15

	surf = fit_Remapped_RV(N, rm_rv_data)
	surf = fit_Standard_RV(N,std_rv_data)



	# Extract curves from the approximated surface
	surf_curves = construct.extract_curves(surf)
	plot_extras = [
		dict(
			points=surf_curves['u'][0].evalpts,
			name="u",
			color="red",
			size=10
		),
		dict(
			points=surf_curves['v'][0].evalpts,
			name="v",
			color="black",
			size=10
		)
	]
	surf.delta = 0.02
	surf.vis = vis.VisSurface()
	surf.render(extras=plot_extras)
	# # exchange.export_obj(surf, rm_file + "_fit.obj")
	# exchange.export_obj(surf, reg_file + "_fit.obj")
	# # visualize data samples, original RV data, and fitted surface
	# eval_surf = np.array(surf.evalpts)
	# np.savetxt(reg_file + "_NURBS_surf_pts.csv",eval_surf,delimiter = ',')
	# # eval_surf = preProcess(eval_surf)

	# fig = plt.figure()
	# ax = plt.axes(projection="3d")
	# ax.scatter(eval_surf[:,0],eval_surf[:,1],eval_surf[:,2], color = 'r')
	# # ax.scatter3D(points[:, 0],points[:, 1],points[:, 2])
	# ax.scatter3D(xyz[:, 0],xyz[:, 1],xyz[:, 2])
	# ax.scatter(X[:,0],X[:,1],X[:,2])

	# # ax.scatter(X[:,0],X[:,1],X[:,2])
	# cpts = np.array(surf.ctrlpts)
	# # np.savetxt('cpts_'+rm_file,cpts, delimiter = '	')
	# np.savetxt('cpts_'+ reg_file + ".csv",cpts, delimiter = ',')
	# fig = plt.figure()
	# ax = plt.axes(projection = "3d")
	# ax.scatter(X[:,0],X[:,1],X[:,2])
	# ax.scatter(cpts[:,0],cpts[:,1],cpts[:,2])
	# plt.show()


	
main()