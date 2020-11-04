from cylinder import Cylinder 
import numpy as np 
from geomdl import fitting
from geomdl import convert
from geomdl	import exchange

N = 11
M = 2
eps1 = 0.1
eps2 = 0.1
z = np.linspace(0,1,N)
theta = np.linspace(0,1,N)

# ED configuration
rho_ed = 30
h_ed = 80
r_ed = []
drdt_ed = []
drdz_ed = []

# Deformed Configuration
rho_es = 25
h_es = 60
r_es = []
drdt_es = []
drdz_es = []
# eps2*np.sin( 2 * np.pi * theta[j] ) )
for i in range(0,len(z)):
	for j in range(0,len(theta)):
		
		cylinder_ed = Cylinder(rho_ed *( (1 + eps1*np.sin( M*2 * np.pi *z[i] ) )) , h_ed,theta[j],z[i])
		r_ed.append(cylinder_ed.cylinder2cart())
		drdt_ed.append(cylinder_ed.partial_theta())
		drdz_ed.append(cylinder_ed.partial_z())

		cylinder_es = Cylinder(rho_es *((1 + eps1*np.sin( 2 * np.pi *z[i] ) ) + eps2*np.sin( 2 * np.pi * theta[i] ))
			,h_es,theta[j],z[i])
		r_es.append(cylinder_es.cylinder2cart())
		drdt_es.append(cylinder_es.partial_theta())
		drdz_es.append(cylinder_es.partial_z())

pts_ed = r_ed
pts_es = r_es
size_u = N
size_v = N
degree_u = 3
degree_v = 3

# Do global surface approximation
surf_ed = fitting.approximate_surface(pts_ed, size_u, size_v, degree_u, degree_v)
surf_ed = convert.bspline_to_nurbs(surf_ed)

# Do global surface approximation
surf_es = fitting.approximate_surface(pts_es, size_u, size_v, degree_u, degree_v)
surf_es = convert.bspline_to_nurbs(surf_es)


return surf_ed,surf_es