from slice import slice
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import sys
from scipy.spatial import ConvexHull, convex_hull_plot_2d
np.set_printoptions(threshold=sys.maxsize)

def main():
	
	# N = No. of slices - 1
	N = 6
	points = np.loadtxt("N2_RV_P0.txt")
	# test = np.loadtxt("N2ED_cpts.txt")
	slice(N, points)
	slices = slice.slices
	slice1 = slice.slices[0]

	ranges = slice.bins
	# np.savetxt("N2_RV_P0.csv", points, delimiter=",")

	fig = plt.figure()
	ax = plt.axes(projection="3d")
	
	# A = []	
	# for i in range(0,len(slices)):
	# 	A.append((slices[i][(slices[i][:,2] == slices[i][:,2].max())]))
	# 	ax.scatter(A[i][0][0],A[i][0][1],A[i][0][2])
	# np.savetxt("slice1.csv", slice1, delimiter=",")
	# np.savetxt("N2ED_cpts.csv", test, delimiter=",")

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
		# ax.scatter3D(
		# slices[i][hull_array[i].vertices[:], 0],
		# slices[i][hull_array[i].vertices[:], 1],
		# np.ones((len(slices[i][hull_array[i].vertices[:], 1]))) * ranges[i],
		# )
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
	ax.scatter3D(convex_vertices[:,0],convex_vertices[:,1],convex_vertices[:,2])
	plt.show()

main()
