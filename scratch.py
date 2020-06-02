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

	theta = np.linspace(layers[:,1].min(),layers[:,1].max(),M)

	xyz_points = []
	# print(layers)
	for i in range(len(theta)-1):
		xyz_points.append(layers[(layers[:, 1] > theta[i]) & (layers[:, 1] < theta[i + 1])])
		if theta[i+1] == max(theta):
			xyz_points.append(layers[(layers[:, 1] > theta[i]+(theta[i+1]-theta[i])/2) & (layers[:,1] < max(theta))])

	data = np.array(xyz_points)
	fig = plt.figure()
	ax = plt.axes(projection="3d")
	t = []
	for i in range(len(data)):
		t.append(cart(data[i][:, 0], data[i][:, 1], data[i][:, 2]))
		# ax.scatter(t[i][:, 0], t[i][:, 1], t[i][:, 2])
	data = np.array(t)

	for i in range(len(data)):
		ax.scatter(data[i][0], data[i][1], data[i][2])
	plt.show()
	return data

points = np.loadtxt("N2_RV_P0.txt")
# points = preProcess(points)

N = 5
slice(N, points)
slices = []
temp = []
layers = []
bins = slice.bins

for j in range(0,len(slice.slices)):
	temp.append(cylinder(slice.slices[j][:,0],slice.slices[j][:,1],bins[j]*np.ones(len(slice.slices[j][:,2]))))
# bins[j]*np.ones(len(slice.slices[j][:,2]))
temp = np.array(temp)

for i in range(0,len(temp)):
	for j in range(0,len(temp[i][0])):
		layers.append([temp[:,0][i][j],temp[:,1][i][j],temp[:,2][i][j]])

# print(layers)

layers = np.array(layers)
segments = split_into_angles(N,layers)

# find average points at each segment and slice
x = []
y = []
z = []
temp1 = []
data = []
fig = plt.figure()
ax = plt.axes(projection= "3d")
segment = []
for i in range(0,len(segments)):
	segment.append(np.array([segments[i][0],segments[i][1],segments[i][2]]).T)

print(segment[0][segment[0][:,2] == bins[0]])




# print(np.array([segments[0][0],segments[0][1],segments[0][2]]).T)
# for j in range(0,len(segments)):
# 	# print(np.array([segments[i][0],segments[i][1],segments[i][2]]).T)
# 	slice(N,np.array([segments[j][0],segments[j][1],segments[j][2]]).T)
# # for j in range(0,len(slice.slices)):
# print(segment_array)
# 	temp1.append([slice.slices[j][:,0],slice.slices[j][:,1],slice.slices[j][:,2]])
# ax.scatter(temp1[j][0],temp1[j][1],temp1[j][2])

print(temp1)
print(x)

ax.scatter(x,y,z)
plt.show()
p_ctrlpts = data
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
