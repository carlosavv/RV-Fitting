from cylinder import Cylinder 
from cylinder import compute_metric_tensor 
from cylinder import compute_strain, compute_stretch
import numpy as np 
from geomdl import fitting
from geomdl.visualization import VisPlotly as vis
from geomdl import construct
from geomdl import convert
from geomdl	import exchange
import random as rand

N = 8
z = np.linspace(0,1,N)
theta = np.linspace(0,1,N)

# Deform Configuration

rho = 32
h = 80
r = []
drdt = []
drdz = []
eps1 = 0.1
eps2 = 0.12
# 	[	rho * ( 1 + eps1*sin(2*pi*theta ) + eps2*sin( 2 * pi *z ) ) cos (2 * pi * theta ), 
# 		rho * ( 1 + eps1* sin(2*pi*theta ) + eps2*sin( 2 * pi *z ) ) sin(2 * pi * theta ), 
# 	 	h * z 	
# 	]

for i in range(0,len(z)):
	for j in range(0,len(theta)):
		# eps = rand.uniform(0,0.05)
		# print(eps)
		cylinder = Cylinder( rho *((1 + eps1*np.sin( 2 * np.pi *z[i] ) ) + eps2*np.sin( 2 * np.pi *theta[i] ))
			,h,theta[j],z[i])
		r.append(cylinder.cylinder2cart())
		drdt.append(cylinder.partial_theta())
		drdz.append(cylinder.partial_z())

pts = r
size_u = N
size_v = N
degree_u = 3
degree_v = 3

# Do global surface approximation
surf = fitting.approximate_surface(pts, size_u, size_v, degree_u, degree_v)
surf = convert.bspline_to_nurbs(surf)

surf.delta = 0.02
surf.vis = vis.VisSurface()
evalpts = np.array(surf.evalpts)
surf.render()
