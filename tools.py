import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import sys
from scipy.spatial import ConvexHull, convex_hull_plot_2d
np.set_printoptions(threshold=sys.maxsize)
from slice import slice


def preProcess(X):

	xmean = np.mean(X[:,0])
	ymean = np.mean(X[:,1])
	zmean = np.mean(X[:,2])
	mean_data = np.array([xmean,ymean,zmean])
	Xnew = X - mean_data
	return Xnew

def vertices(N,points):
	# N = No. of slices - 1
	slice(N, points)
	slices = slice.slices
	ranges = slice.bins

	hull_array = []
	for i in range(0, len(slices)):
		hull_array.append(ConvexHull(slices[i][:, 0:2])) 

	# initialize simplices and vertices list arrays
	simplices = []
	vertices = []

	# get the simplex at each slice 
	for i in range(0, len(slices)):
		for simplex in hull_array[i].simplices:
			# plt.plot(slices[i][simplex, 0], slices[i][simplex, 1], "k-")
			simplices.append(slices[i][simplex[:]])

	# get the vertices at each slice
	for i in range(0, len(slices)):
		vertices.append(
		[
			slices[i][hull_array[i].vertices[:], 0],
			slices[i][hull_array[i].vertices[:], 1],
			np.ones((len(slices[i][hull_array[i].vertices[:], 0]))) * ranges[i],
		]
			)

	# store xyz points and set vertex layers
	vertex_layer1 = np.array([vertices[0][0],vertices[0][1],vertices[0][2]]).T
	vertex_layer2 = np.array([vertices[1][0],vertices[1][1],vertices[1][2]]).T
	vertex_layer3 = np.array([vertices[2][0],vertices[2][1],vertices[2][2]]).T
	vertex_layer4 = np.array([vertices[3][0],vertices[3][1],vertices[3][2]]).T
	vertex_layer5 = np.array([vertices[4][0],vertices[4][1],vertices[4][2]]).T
	vertex_layer6 = np.array([vertices[5][0],vertices[5][1],vertices[5][2]]).T

	# store vertex layers into array
	vertex_layers = np.array([vertex_layer1,vertex_layer2,vertex_layer3,vertex_layer4,vertex_layer5,vertex_layer6])

	#append all of the layers into one array from least to greatest
	test = []
	for i in range(0, len(vertex_layers)):
		np.random.shuffle(vertex_layers[i])
		test.append(vertex_layers[i][:])

	temp = []
	for i in range(0,len(test)):
		for j in range(0,len(test[i])):
			temp.append(test[i][j])
	
	convex_vertices = np.array(temp)
	print(len(convex_vertices))  
	# ax.scatter3D(convex_vertices[:,0],convex_vertices[:,1],convex_vertices[:,2])
	# plt.show()

	return convex_vertices