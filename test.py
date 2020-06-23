import numpy as np 
import matplotlib.pyplot as plt 
from mpl_toolkits import mplot3d
from geomdl import fitting
from geomdl import convert
from geomdl import construct
from geomdl.visualization import VisMPL as vis
import sys
np.set_printoptions(threshold=sys.maxsize)
from tools import preProcess
from geomdl import exchange

# data = np.loadtxt("sdata.csv",delimiter = ',')


# data = preProcess(data)
# data = np.array([x,y,z]).T
# print(len(data))


# generate a helical axis first

t = np.linspace(0,2*np.pi/3,8)
test = []
for j in range(0,len(t)):
    test.append([50*np.cos(t[j]),50*np.sin(t[j]),25*t[j]])
test = np.array(test)
np.savetxt("cpts.dat",test,delimiter = ' ')

# create regularly spaced points about each helical axis point 
# to create a helical tube

theta = np.linspace(0,2*np.pi,8)
test1 = []
for j in range(0,len(test)):
    for i in range(0,len(theta)):
        test1.append([test[j,0] + 15*np.cos(theta[i]),test[j,1] + 15*np.sin(theta[i]),test[j,2]])

points = np.array(test1)

fig = plt.figure()
ax = plt.axes(projection = '3d')
ax.set_facecolor('xkcd:grey')
ax.scatter(points[:,0],points[:,1],points[:,2],color = "red")
plt.show()

# set up the fitting parameters
p_ctrlpts = points
size_u = 8
size_v = 8
degree_u = 3
degree_v = 3

# Do global surface approximation
surf = fitting.approximate_surface(p_ctrlpts, size_u, size_v, degree_u, degree_v)

surf = convert.bspline_to_nurbs(surf)
print("surf",type(surf))

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
tube_pcl = np.array(surf.evalpts)
tube_cpts = np.array(surf.ctrlpts)

np.savetxt("cpts_tube.dat",tube_cpts,delimiter = ' ')
from matplotlib import cm
surf.delta = 0.018
surf.vis = vis.VisSurface( ctrlpts = False)
surf.render(extras=plot_extras)
exchange.export_obj(surf, "fitted_helix.obj")
# visualize data samples, original RV data, and fitted surface

np.savetxt("RV_tube.dat",tube_pcl,delimiter = ' ')

fig = plt.figure()
ax = plt.axes(projection="3d")
ax.scatter(tube_pcl[:,0],tube_pcl[:,1],tube_pcl[:,2])
ax.scatter(points[:,0],points[:,1],points[:,2],color = "red")
plt.show()

