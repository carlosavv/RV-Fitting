# imports
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import sys
from geomdl import utilities as utils
from geomdl import NURBS
from geomdl import exchange
from geomdl import construct
from geomdl import fitting
from geomdl.visualization import VisMPL as vis
from geomdl import compatibility as compat
np.set_printoptions(threshold=sys.maxsize)
from slice import slice
from geomdl import operations
from scipy.spatial.distance import cdist
# import scipy.optimize.least_squares as ls

# function that parameterizes into cylindrical coordinates
def cylinder(x,y,z):
    r = np.sqrt(x**2+y**2)
    theta = np.arctan(y/x)
    z = z
    return r,theta,z

# function the parameterizes into cartesian coordinates
def cart(r,theta,z):
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    z = z
    return x,y,z


# def 
# load remapped RV data
points = np.loadtxt("N2_RV_P0.txt")
x = points[:, 0]
y = points[:, 1]
z = points[:, 2]

A = []

# parameterize into cylindrical coordinates
for i in range(0,len(z)):
    A.append(cylinder(x[i],y[i],z[i]))
A  = np.array(A)

# split RV into evenly spaced slices
N = 10
slice(N, points)
slices = []
temp = []

# store radii at each slice
for j in range(0,len(slice.slices)):
    temp.append(cylinder(slice.slices[j][:,0],slice.slices[j][:,1],slice.slices[j][:,2]))

temp = np.array(temp)
print(temp[0][0:3])


# radius = np.linspace(A[:,0].max(), A[:,0].min(), N)
radius = []
theta = []
z = []
for i in range(0,len(temp)):
    radius.append(temp[i][0].mean())
    theta.append(temp[i][1].mean())
    z.append(temp[i][2].mean())

print(radius)
# create evenly spaced heights
# z = np.linspace(A[:,2].min(), A[:,2].max(), N)
print(len(z))

# evenly spaced angles from 0 to 2pi
# theta = np.linspace(-np.pi,np.pi,N)
# print(len(theta))

# parametrize data back into cartesian coordinates
X = []
for j in range(0,len(theta)):
    for i in range(0,len(radius)):
        X.append(cart(radius[i],theta[j],z[i]))
# these are now candidate control points
X = np.array(X)

fig = plt.figure()
ax = plt.axes(projection="3d")
ax.scatter(X[:,0],X[:,1],X[:,2])

print(len(X))
np.savetxt("cpts_test.csv", X, delimiter=",")

# setup pre reqs for surface fitting
p_ctrlpts = X
size_u = N
size_v = N
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
# ax.scatter(eval_surf[:,0],eval_surf[:,1],eval_surf[:,2])
# ax.scatter3D(points[:, 0],points[:, 1],points[:, 2])
ax.scatter(X[:,0],X[:,1],X[:,2])
cpts = np.array(surf.ctrlpts)
ax.scatter(cpts[:,0],cpts[:,1],cpts[:,2])
plt.show()