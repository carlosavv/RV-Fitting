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

class Cylinder():
	
	def __init__(self,R,h,theta,z):
		self.R = R 
		self.h = h
		self.theta = theta
		self.z = z
	
	def cylinder2cart(self):
		x = self.R*np.cos(2*np.pi*self.theta)
		y = self.R*np.sin(2*np.pi*self.theta)
		Z = self.h*self.z
		r = [x,y,Z]
		return r
	
	def partial_theta(self):
		# partial derivative of r with respect to theta
		return [-self.R*2*np.pi*np.sin(2*np.pi*self.theta),self.R*2*np.pi*np.cos(2*np.pi*self.theta),self.h*0]
	
	def partial_z(self):
		return [0,0,self.h]

def compute_metric_tensor(surf,uv):
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

def compute_strain(m1,m2):
	return 0.5*(m2-m1)

def compute_stretch(eps,m1,m2):
	lambda_u = 1/( np.sqrt(1 - 2*(eps[0]/m2[0]) ) )
	lambda_v = 1/( np.sqrt(1 - 2*(eps[1]/m2[3]) ) )
	return [lambda_u,lambda_v]

# Starting configuration
R_ed = 4
theta = np.linspace(0,1,20)
h_ed = 10
z = np.linspace(0,1,20)
r_ed = []
drdt_ed = []
drdz_ed = []

# Deformed Configuration
R_es = 3
h_es = 8
r_es = []
drdt_es = []
drdz_es = []

for i in range(0,len(z)):
	for j in range(0,len(theta)):
		cylinder_ed = Cylinder(R_ed,h_ed,theta[j],z[i])
		cylinder_es = Cylinder(R_es,h_es,theta[j],z[i])
		r_ed.append(cylinder_ed.cylinder2cart())
		r_es.append(cylinder_es.cylinder2cart())
		drdt_ed.append(cylinder_ed.partial_theta())
		drdt_es.append(cylinder_es.partial_theta())
		drdz_ed.append(cylinder_ed.partial_z())
		drdz_es.append(cylinder_es.partial_z())

print(len(drdt_ed))
pts_ed = r_ed
size_u = 20
size_v = 20
degree_u = 3
degree_v = 3

# Do global surface approximation
surf_ed = fitting.approximate_surface(pts_ed, size_u, size_v, degree_u, degree_v)
surf_ed = convert.bspline_to_nurbs(surf_ed)

surf_ed.delta = 0.025
surf_ed.vis = vis.VisSurface()
evalpts_ed = np.array(surf_ed.evalpts)

pts_es = r_es

# Do global surface approximation
surf_es = fitting.approximate_surface(pts_es, size_u, size_v, degree_u, degree_v)
surf_es = convert.bspline_to_nurbs(surf_es)

surf_es.delta = 0.025
surf_es.vis = vis.VisSurface()
evalpts_es = np.array(surf_es.evalpts)

exchange.export_obj(surf_ed,"clyinder_ed.obj")
exchange.export_obj(surf_es,"clyinder_es.obj")

# compute analytical metric tensor for ed phase
a_E_ed = []
a_F_ed = []
a_G_ed = []
for i in range(0,len(drdt_ed)):
	a_E_ed.append(np.dot(drdz_ed[i],drdz_ed[i]))
	a_F_ed.append(np.dot(drdz_ed[i],drdt_ed[i]))
	a_G_ed.append(np.dot(drdt_ed[i],drdt_ed[i]))

a_metric_tensor_ed = np.array([a_E_ed,a_F_ed,a_F_ed,a_G_ed]).T

# print('analytical metric tensor (ed phase) = ')
# print(a_metric_tensor_ed)

# compute analytical metric tensor for es phase
a_E_es = []
a_F_es = []
a_G_es = []
for i in range(0,len(drdt_es)):
	a_E_es.append(np.dot(drdz_es[i],drdz_es[i]))
	a_F_es.append(np.dot(drdz_es[i],drdt_es[i]))
	a_G_es.append(np.dot(drdt_es[i],drdt_es[i]))

a_metric_tensor_es = np.array([a_E_es,a_F_es,a_F_es,a_G_es]).T

# print('analytical metric tensor (es phase) = ')
# print(a_metric_tensor_es)

# compute the NURBS metric tensor (ed phase)
uv = np.linspace(0,1,11)
metric_tensor_ed = compute_metric_tensor(surf_ed,uv)

# print('NURBS (ed) fit metric tensor = ')
# print(metric_tensor_ed)

# compute the NURBS metric tensor (es phase)
metric_tensor_es = compute_metric_tensor(surf_es,uv)

# print('NURBS (es) fit metric tensor = ')
# print(metric_tensor_es)

strain = []
stretch = []
for i in range(0,len(metric_tensor_ed)):
	strain.append(compute_strain(metric_tensor_ed[i],metric_tensor_es[i]))
	stretch.append(compute_stretch(strain[i],metric_tensor_ed[i],metric_tensor_es[i]))

print(np.array(strain))
# u-stretch
print(np.array(stretch))
