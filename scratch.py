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
	x = []
	y = []
	z = []
	r0 = layers[0][0].max()
	print(r0)
	# theta0 = 0
	# z0 = layers[0][2].mean()
	k = np.linspace(0,M,M+1)
	theta = np.linspace(0,2*np.pi,M+1)
	# x0 = r0*np.cos(theta0)
	# y0 = r0*np.sin(theta0)

	xyz_points = []
	# xyz_points.append([x0,y0,z0])
	for i in range(0,len(layers)):
		# for j in range(0,len(layers[i][0])):
		for ii in range(0,len(theta)):
			xyz_points.append(
				[ 	layers[i][0].max()*np.cos(theta[ii]),
					layers[i][1].max()*np.sin(theta[ii]),
					layers[i][2].max()
				]
			)
	data = np.array(xyz_points)
	# print(data)
	print(len(data))
	fig = plt.figure()
	ax = plt.axes(projection="3d")
	ax.scatter(data[:,0],data[:,1],data[:,2])
	plt.show()
	return data


	# return mean_r,mean_theta,mean_z


points = np.loadtxt("N2_RV_P0.txt")
# points = preProcess(points)

N = 6
slice(N, points)
slices = []
temp = []
layers = []
bins = slice.bins

for j in range(0,len(slice.slices)):
	layers.append([slice.slices[j][:,0],slice.slices[j][:,1],bins[j]*np.ones(len(slice.slices[j][:,2]))])
# bins[j]*np.ones(len(slice.slices[j][:,2]))
layers = np.array(layers)

# for i in range(0,len(temp)):
# 	for j in range(0,len(temp[i][0])):
# 		layers.append((temp[:,0][i][j],temp[:,1][i][j],temp[:,2][i][j]))

# print(layers)
split_into_angles(N,layers)
X = split_into_angles(N,layers)

p_ctrlpts = X
size_u = N+1
size_v = N+1
degree_u = 3
degree_v = 3

# Do global surface approximation
surf = fitting.approximate_surface(p_ctrlpts, size_u, size_v, degree_u, degree_v,centripetal = True)

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
surf.delta = 0.03
surf.vis = vis.VisSurface()
surf.render(extras=plot_extras)


'''
Next steps: 
- get the evaluated points of generated surface
- optimize (minimize) distance from generated surface to remapped RV
- move control points based on this optimization
'''

eval_surf = np.array(surf.evalpts)

fig = plt.figure()
ax = plt.axes(projection="3d")
ax.scatter(eval_surf[:,0],eval_surf[:,1],eval_surf[:,2])
ax.scatter3D(points[:, 0],points[:, 1],points[:, 2])
ax.scatter(X[:,0],X[:,1],X[:,2])
cpts = np.array(surf.ctrlpts)
fig = plt.figure()
ax = plt.axes(projection = "3d")
ax.scatter(X[:,0],X[:,1],X[:,2])
ax.scatter(cpts[:,0],cpts[:,1],cpts[:,2])
plt.show()
