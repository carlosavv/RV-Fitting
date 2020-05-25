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

# function that parameterizes into cylindrical coordinates
def cylinder(x,y,z):
    r = np.sqrt(x**2+y**2)
    theta = round(np.arctan(y/x)*(180/np.pi))
    z = z
    return r,theta,z

# function the parameterizes into cartesian coordinates
def cart(r,theta,z):
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    z = z
    return x,y,z

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
N = 6
slice(N, points)
slices = []
temp = []

# store radii at each slice
for j in range(0,len(slice.slices)):
    temp.append(np.sqrt(slice.slices[j][:,0]**2 + slice.slices[j][:,1]**2))
radius = []
for i in range(0,len(temp)):
    radius.append(temp[i].mean())

# create evenly spaced heights
z = np.linspace(A[:,2].min(), A[:,2].max(), N)

# evenly spaced angles from 0 to 2pi
theta = np.linspace(0,2*np.pi,N)

# for each theta find points within eps of theta and take avg. radius of those points 

# parametrize data back into cartesian coordinates
X = []
for i in range(0,len(theta)):
    for j in range(0,len(z)):
        X.append(cart(radius[j],theta[i],z[j]))
# these are now candidate control points
X = np.array(X)
print(X)
fig = plt.figure()
ax = plt.axes(projection="3d")
ax.scatter(X[:,0],X[:,1],X[:,2])
plt.show()
np.savetxt("cpts_test.csv", X, delimiter=",")

# setup pre reqs for surface fitting
p_ctrlpts = X
size_u = N
size_v = N
degree_u = 3
degree_v = 3

# Do global surface approximation
surf = fitting.approximate_surface(p_ctrlpts, size_u, size_v, degree_u, degree_v)

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
fig = plt.figure()
ax = plt.axes(projection="3d")
ax.scatter(X[:,0],X[:,1],X[:,2])
cpts = np.array(surf.ctrlpts)
ax.scatter(cpts[:,0],cpts[:,1],cpts[:,2])
plt.show()