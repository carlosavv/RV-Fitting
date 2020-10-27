# imports
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from geomdl import fitting
from geomdl.visualization import VisPlotly as vis
from geomdl import construct
from geomdl import convert
from geomdl	import exchange
# plt.style.use("seaborn")
import sys
sys.path.append("../")


def cylinder2cart(R,h,theta,z):

	'''
	parameters: 

	R - radius
	theta - angle
	z - height

	returns:

	r - cylindrical position vector

	'''
	x = R*np.cos(2*np.pi*theta)
	y = R*np.sin(2*np.pi*theta)
	Z = h*z
	r = [x,y,Z]
	return r

def r_theta(R,h,theta,z):
	# partial derivative of r with respect to theta
	return [-R*2*np.pi*np.sin(2*np.pi*theta),R*2*np.pi*np.cos(2*np.pi*theta),h*0]

def r_z(R,h,theta,z):
	# partial derivative of r with respect to z
	return [0,0,h] 

def compute_metric_tensor(uv):
	'''
	args: uv - array: linspace from [0,1] to index the metric tensor
	returns: metric tensor array where each column corresponds to the respective index of uv 
	'''
	E = []
	F = []
	G = []
	for i in range(0,len(uv)):
		E.append(np.dot(surf.derivatives(uv[i],uv[i],1)[1][0],surf.derivatives(uv[i],uv[i],1)[1][0]))
		F.append(np.dot(surf.derivatives(uv[i],uv[i],1)[1][0],surf.derivatives(uv[i],uv[i],1)[0][1]))
		G.append(np.dot(surf.derivatives(uv[i],uv[i],1)[0][1],surf.derivatives(uv[i],uv[i],1)[0][1]))

	return np.array([E,F,F,G]).T
R = 4
theta = np.linspace(0,1,20)
h = 10
z = np.linspace(0,1,20)
r = []
drdt = []
drdz = []

for i in range(0,len(z)):
	for j in range(0,len(theta)):
		r.append(cylinder2cart(R,h,theta[j],z[i]))
		drdt.append(r_theta(R,h,theta[j],z[i]))
		drdz.append(r_z(R,h,theta[j],z[i]))

r = np.array(r)
print(len)
fig = plt.figure()
ax = plt.axes(projection = '3d')
ax.scatter(r[:,0],r[:,1],r[:,2],s = 50, color = 'r')

pts = r 

size_u = 20
size_v = 20
degree_u = 3
degree_v = 3

# Do global surface approximation
surf = fitting.approximate_surface(pts, size_u, size_v, degree_u, degree_v)
surf = convert.bspline_to_nurbs(surf)

surf.delta = 0.025
surf.vis = vis.VisSurface()
evalpts = np.array(surf.evalpts)
ax.scatter(evalpts[:,0],evalpts[:,1],evalpts[:,2], color = 'b')
# plt.show()
surf_curves = construct.extract_curves(surf)
plot_extras = [
	dict(
		points=surf_curves['u'][0].evalpts,
		name="u",
		color="red",
		size=5
	),
	dict(
		points=surf_curves['v'][0].evalpts,
		name="v",
		color="black",
		size=5
	)
]

# surf.render(extras = plot_extras)
exchange.export_obj(surf,"clyinder_ed.obj")
# compute analytical metric tensor
a_E = []
a_F = []
a_G = []
for i in range(0,len(drdt)):
	a_E.append(np.dot(drdz[i],drdz[i]))
	a_F.append(np.dot(drdz[i],drdt[i]))
	a_G.append(np.dot(drdt[i],drdt[i]))
a_metric_tensor = np.array([a_E,a_F,a_F,a_G]).T
print('analytical metric tensor = ')
print(a_metric_tensor)

# compute the metric tensor

uv = np.linspace(0,1,11)

metric_tensor = compute_metric_tensor(uv)
print('NURBS fit metric tensor = ')
print(metric_tensor[0])

