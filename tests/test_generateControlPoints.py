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
# import scipy.optimize.least_squares as ls

# function that parameterizes into cylindrical coordinates
def cylinder(x,y,z):
    r = np.sqrt(x**2+y**2)
    theta = np.arctan2(y,x)
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
points = np.loadtxt('D:/Workspace/RV-Fitting/rv_data/N2_RV_P0_rm.csv',delimiter = '\t')
# points = preProcess(points)
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
bins = slice.bins

# store radii at each slice
for j in range(0,len(slice.slices)):
    temp.append(cylinder(slice.slices[j][:,0],slice.slices[j][:,1],bins[j]*np.ones(len(slice.slices[j][:,2]))))

temp = np.array(temp)
# radius = np.linspace(A[:,0].max(), A[:,0].min(), N)
radius = []
theta = []
z = []

for i in range(0,len(temp)):
    radius.append(temp[i][0].mean())
    theta.append(temp[i][1])
    z.append(temp[i][2].mean())
theta = np.array(theta)
# for each theta find points within eps of theta and take avg. radius of those points 

# create evenly spaced heights
# z = np.linspace(A[:,2].min(), A[:,2].max(), N)

# evenly spaced angles from 0 to 2pi
theta = np.linspace(A[:,1].min(), A[:,1].max(), 20)
# print(len(theta))

# parametrize data back into cartesian coordinates

X = []
for i in range(0,len(radius)):
    for j in range(0,len(theta)):
        X.append(cart(radius[i],theta[j],z[i]))
# these are now candidate data points
X = np.array(X)
print(X)

fig = plt.figure()
ax = plt.axes(projection="3d")
plt.title('Cartesian')
ax.scatter(X[:,0],X[:,1],X[:,2])

np.savetxt("cpts_test.csv", X, delimiter=",")

# setup pre reqs for surface fitting
p_ctrlpts = X
size_u = N+1
size_v = 20
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