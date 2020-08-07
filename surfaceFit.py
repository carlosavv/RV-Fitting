# imports
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import sys
from geomdl import fitting
from geomdl.visualization import VisMPL as vis
from slice import slice
from tools import preProcess
from geomdl import construct

def cylinder(x,y,z):
    r = np.sqrt(x**2+y**2)
    theta = np.arctan2(y,x)
    z = z
    return r,theta,z


def cart(r,theta,z):
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    z = z
    return x,y,z

def split_into_angles(M,layers):

	'''
	function that splits data in angled segments
	'''
	theta = np.linspace(layers[:,1].min(),layers[:,1].max(),M+1)

	points = []
	for i in range(len(theta)-1):
		points.append(layers[(layers[:, 1] > theta[i]) & (layers[:, 1] < theta[i + 1])])
	data = np.array(points)
	# fig = plt.figure()
	# ax = plt.axes(projection="3d")
	t = []
	for i in range(len(data)):
		t.append(cart(data[i][:, 0], data[i][:, 1], data[i][:, 2]))
		# ax.scatter(t[i][:, 0], t[i][:, 1], t[i][:, 2])
	data = np.array(t)


	# for i in range(len(data)):
	# 	ax.scatter(data[i][0], data[i][1], data[i][2])
	return data

def fitSurface(N, points):

	slice(N, points)
	slices = []
	temp = []
	layers = []
	bins = slice.bins

	for j in range(0,len(slice.slices)):
		temp.append(cylinder(slice.slices[j][:,0],slice.slices[j][:,1],bins[j]*np.ones(len(slice.slices[j][:,2]))))
	temp = np.array(temp)

	# store all slices into layers array
	for i in range(0,len(temp)):
		for j in range(0,len(temp[i][0])):
			layers.append([temp[:,0][i][j],temp[:,1][i][j],temp[:,2][i][j]])

	# segment the layers into angled segments
	layers = np.array(layers)
	M = N
	segments = split_into_angles(M,layers)

	# find average points at each segment and slice

	temp1 = []
	data = []
	fig = plt.figure()
	ax = plt.axes(projection= "3d")
	segment = []

	for i in range(0,len(segments)):
		segment.append(np.array([segments[i][0],segments[i][1],segments[i][2]]).T)
		for j in range(0,len(bins)):
			temp1.append(segment[i][segment[i][:,2] == bins[j]])

	chunks = np.array(temp1)
	xbar = []
	ybar = []
	zbar = []
	for j in range(0,len(chunks)):
		xbar.append(chunks[j][:,0].mean())
		ybar.append(chunks[j][:,1].mean())
		zbar.append(chunks[j][:,2].max())
	for i in range(0,(N+1)):
		xbar.append(chunks[i][:,0].mean())
		ybar.append(chunks[i][:,1].mean())
		zbar.append(chunks[i][:,2].max())
	test = []
	fitSurface.X = np.array([xbar,ybar,zbar]).T
	X = fitSurface.X
	# this orders the points from least to greatest height (z values)
	for i in range(0,len(bins)):
		test.append(X[X[:,2] == bins[i]])
	for j in range(0,len(test)):
		for ii in range(0,len(test[i])):
			data.append([test[j][ii][0],test[j][ii][1],test[j][ii][2]])

	data = np.array(data)
	ax.scatter(xbar,ybar,zbar)
	print(len(X))

	# set up the fitting parameters
	p_ctrlpts = X
	size_u = N+1
	size_v = M+1
	degree_u = 3
	degree_v = 3

	# Do global surface approximation
	surf = fitting.interpolate_surface(p_ctrlpts, size_u, size_v, degree_u, degree_v)

	return surf