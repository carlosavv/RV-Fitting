from surfaceFit import fitSurface
import glob,os
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import sys
from geomdl.visualization import VisMPL as vis
from slice import slice
from tools import preProcess
from geomdl import construct
import numpy as np


def main():
	# load data
	path = os.getcwd()
	os.chdir(path)
	rvFiles = []
	for file in glob.glob("*.csv"):
		rvFiles.append(file)

	print(rvFiles)
	rv_data = []
	for i in range(0,len(rvFiles)):
		rv_data.append(np.loadtxt((rvFiles[i]),delimiter = ','))
	N = 10
	print(rv_data[1])
	surfs = []

	for i in range(0,len(rv_data)):
		surfs.append(fitSurface(N,rv_data[i]))


	# Extract curves from the approximated surface
	surf_curves = construct.extract_curves(surfs[4])
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
	surfs[4].delta = 0.03
	surfs[4].vis = vis.VisSurface()
	surfs[4].render(extras=plot_extras)

	# visualize data samples, original RV data, and fitted surface
	eval_surf = np.array(surfs[4].evalpts)

	fig = plt.figure()
	ax = plt.axes(projection="3d")
	ax.scatter(eval_surf[:,0],eval_surf[:,1],eval_surf[:,2])
	ax.scatter3D(rv_data[4][:,0],rv_data[4][:,1],rv_data[4][:,2])
	ax.scatter(fitSurface.X[:,0],fitSurface.X[:,1],fitSurface.X[:,2])
	cpts = np.array(surfs[4].ctrlpts)
	fig = plt.figure()
	ax = plt.axes(projection = "3d")
	ax.scatter(fitSurface.X[:,0],fitSurface.X[:,1],fitSurface.X[:,2])
	ax.scatter(cpts[:,0],cpts[:,1],cpts[:,2])
	plt.show()

main()