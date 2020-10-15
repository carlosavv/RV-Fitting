import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import sys
from scipy.spatial import ConvexHull, convex_hull_plot_2d

np.set_printoptions(threshold=sys.maxsize)
from slice import slice
from conversions import cylinder2cart


def preProcess(X):

    xmean = np.mean(X[:, 0])
    ymean = np.mean(X[:, 1])
    zmean = np.mean(X[:, 2])
    mean_data = np.array([xmean, ymean, zmean])
    Xnew = X - mean_data
    return Xnew


def vertices(N, points):
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
    vertex_layers = []
    for i in range(0, len(vertices)):
        vertex_layers.append(
            np.array([vertices[i][0], vertices[i][1], vertices[i][2]]).T
        )

    # append all of the layers into one array from least to greatest
    test = []
    for i in range(0, len(vertex_layers)):
        np.random.shuffle(vertex_layers[i])
        test.append(vertex_layers[i][:])

    temp = []
    for i in range(0, len(test)):
        for j in range(0, len(test[i])):
            temp.append(test[i][j])

    convex_vertices = np.array(temp)
    print(len(convex_vertices))

    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.scatter3D(convex_vertices[:, 0], convex_vertices[:, 1], convex_vertices[:, 2])
    plt.show()

    return convex_vertices


def split_into_angles(M, layers):

    """
	function that splits data in angled segments

	arguments: M - number of segments
			   layers - array of layers to be segmented 

	returns: array with segments
	"""
    theta = np.linspace(layers[:, 1].min(), layers[:, 1].max(), M + 1)
    points = []

    for i in range(len(theta) - 1):
        points.append(layers[(layers[:, 1] > theta[i]) & (layers[:, 1] < theta[i + 1])])

    data = np.array(points)
    t = []

    for i in range(len(data)):
        t.append(cylinder2cart(data[i][:, 0], data[i][:, 1], data[i][:, 2]))

    data = np.array(t)

    return data
