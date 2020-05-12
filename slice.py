import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import sys
from scipy.spatial import ConvexHull, convex_hull_plot_2d
np.set_printoptions(threshold=sys.maxsize)

def slice(n, points):

	# function that computes and stores equally spaced cross section slices of 3D data

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

	# fig4 = plt.figure()
	# ax3 = plt.axes(projection="3d")
	# for i in range(len(slices)):
		# ax3.scatter(slices[i][:, 0], slices[i][:, 1], slices[i][:, 2])
	# plt.show()

