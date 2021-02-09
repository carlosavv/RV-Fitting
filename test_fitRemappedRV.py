# imports
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import sys
from geomdl import fitting
from geomdl.visualization import VisMPL as vis
from slice import slice
from tools import preProcess, split_into_angles
from geomdl import construct
from conversions import cart2cylinder,cylinder2cart

'''

By default we use flag = False because this is the way to properly fit the remapped RV, 
however, if we want to just extract the data to be fitted for the pull-back RV
Fit, then use flag = True

'''
def fit_Remapped_RV(N,M, points, flag=False):

    slice(N, points)
    slices = []
    cyl_coord_pts = []
    layers = []
    bins = slice.bins

    for j in range(0, len(slice.slices)):
        cyl_coord_pts.append(
            cart2cylinder(
                slice.slices[j][:, 0],
                slice.slices[j][:, 1],
                bins[j] * np.ones(len(slice.slices[j][:, 2])),
            )
        )

    cyl_coord_pts = np.array(cyl_coord_pts)

    # store all slices into layers array
    for i in range(0, len(cyl_coord_pts)):
        for j in range(0, len(cyl_coord_pts[i][0])):
            layers.append(
                [
                    cyl_coord_pts[:, 0][i][j],
                    cyl_coord_pts[:, 1][i][j],
                    cyl_coord_pts[:, 2][i][j],
                ]
            )

    # segment the layers into angled segments
    layers = np.array(layers)

    segments = split_into_angles(M, layers)

    # find average points at each segment and slice

    chunks = []
    segment = []

    for i in range(0, len(segments)):
        segment.append(np.array([segments[i][0], segments[i][1], segments[i][2]]).T)
        for j in range(0, len(bins)):
            chunks.append(segment[i][segment[i][:, 2] == bins[j]])

    chunks = np.array(chunks)

    xbar = []
    ybar = []
    zbar = []

    cylData = []
    cartData = []

    if flag == True:
        for j in range(0, len(chunks)):
            if chunks[j].size == 0:
                print('')
            else:
                cylData.append(cart2cylinder(chunks[j][:, 0], chunks[j][:, 1], chunks[j][:, 2]))
        
        for i in range(0, len(cylData)): 

            cartData.append(
            cylinder2cart(cylData[i][0].max(), cylData[i][1].max(),
            cylData[i][2].max()) )

        for i in range(0, (N + 1)):
            cartData.append(
                cylinder2cart(cylData[i][0].max(), cylData[i][1].max(), cylData[i][2].max())
            )
        X = np.array(cartData)
        # print(X)
        # ax.scatter(X[:,0],X[:,1],X[:,2])
    else:
        fig = plt.figure()
        ax =plt.axes(projection = '3d')
        for j in range(0, len(chunks)):
            print(j)
            xbar.append(chunks[j][:, 0].mean())
            ybar.append(chunks[j][:, 1].mean())
            zbar.append(chunks[j][:, 2].max())
            print(len(chunks[j]))
            ax.scatter(chunks[j][:,0],chunks[j][:,1],chunks[j][:,2])
            plt.show()
        for i in range(0, (N + 1)):
            xbar.append(chunks[i][:, 0].mean())
            ybar.append(chunks[i][:, 1].mean())
            zbar.append(chunks[i][:, 2].max())

        X = np.array([xbar, ybar, zbar]).T
    # test = []

    # # this orders the points from least to greatest height (z values)
    # for i in range(0, len(bins)):
    #     test.append(X[X[:, 2] == bins[i]])
    # for j in range(0, len(test)):
    #     for ii in range(0, len(test[i])):
    #         data.append([test[j][ii][0], test[j][ii][1], test[j][ii][2]])

    # data = np.array(data)

    # set up the fitting parameters
    p_ctrlpts = X
    size_u = M + 1
    size_v = N + 1
    degree_u = 3
    degree_v = 3

    # fit a surface by applying global interpolation
    remapped_NURBS_RV_surf = fitting.interpolate_surface(
        p_ctrlpts, size_u, size_v, degree_u, degree_v
    )

    return remapped_NURBS_RV_surf,X
