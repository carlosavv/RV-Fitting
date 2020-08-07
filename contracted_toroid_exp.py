import numpy as np 
import matplotlib.pyplot as plt
from geomdl import fitting
from geomdl.visualization import VisMPL as vis
from slice import slice
from tools import preProcess
from geomdl import construct
from geomdl import convert
from geomdl	import exchange

def compute_torus_xyz(theta,phi,R,r,a,b):
	radius = r - (a*np.sin(phi) + b*np.sin(phi))
	x = (R + radius*np.cos(theta))*np.cos(phi)
	y = (R + radius*np.cos(theta))*np.sin(phi)
	z = radius*np.sin(theta)
	return x,y,z

def compute_centralAxis_xyz(phi,R):
	return R*np.cos(phi),R*np.sin(phi),0

N = 10
theta = np.linspace(0,2*np.pi,N)
phi = np.linspace(np.pi/32,15*np.pi/32,N)
a = 15
b = 0
R = 90
r = 25
toroid = []
caxis = []
for i in range(0,len(theta)):
	for j in range(0,len(phi)):
		toroid.append(compute_torus_xyz(theta[i],phi[j],R,r,a,b))
		caxis.append(compute_centralAxis_xyz(phi[i],R))


toroid = np.array(toroid)
np.savetxt("con_tapered_toroid.csv",toroid, delimiter = ',')

caxis = np.array(caxis)
np.savetxt("con_toroid_caxis.csv",caxis, delimiter = ',')


fig = plt.figure(dpi = 175)
ax = plt.axes(projection = '3d')
plt.style.use('dark_background')
ax.scatter(toroid[:,0],toroid[:,1],toroid[:,2], color = 'green',marker = 'o')
ax.plot(caxis[:,0],caxis[:,1],caxis[:,2],color = 'red')
ax.axis("off")
p_ctrlpts = toroid
size_u = N
size_v = N
degree_u = 3
degree_v = 3

# Do global surface approximation
surf = fitting.approximate_surface(p_ctrlpts, size_u, size_v, degree_u, degree_v)

surf = convert.bspline_to_nurbs(surf)

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
toroid_pcl = np.array(surf.evalpts)
toroid_cpts = np.array(surf.ctrlpts)
print(len(toroid_cpts))

# np.savetxt("cpts_bezier.dat",r,delimiter = ',')
# from matplotlib import cm
surf.delta = 0.02
surf.vis = vis.VisSurface()
surf.render(extras=plot_extras)
exchange.export_obj(surf, "con_tapered_toroid.obj")
np.savetxt("tapered_toroid.dat",toroid_pcl,delimiter = ' ')
np.savetxt("con_tapered_toroid.dat",toroid_pcl,delimiter = ',')
# np.savetxt("tube_cpts.dat",tube_cpts,delimiter = ' ')
np.savetxt("con_toroid_cpts.csv",toroid_cpts,delimiter = ',')


plt.show()


