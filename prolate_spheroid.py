import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from geomdl import fitting
from geomdl import convert
from geomdl import construct
from geomdl.visualization import VisMPL as vis
from slice import slice
plt.style.use('dark_background')

def p_spheroid(mu,nu,phi,a = 10):
	x = a*np.sinh(mu)*np.sin(nu)*np.cos(phi)
	y = a*np.sinh(mu)*np.sin(nu)*np.sin(phi)
	z = a*np.cosh(mu)*np.cos(nu)
	return x,y,z
mu = .75
nu = np.linspace(0,np.pi,10)
phi = np.linspace(0,2*np.pi,10)

sph = []

for i in range(0,len(nu)):
	for j in range(0,len(phi)):
		sph.append(p_spheroid(mu,nu[i],phi[j]))


sph = np.array(sph)
# new_sph = np.delete(sph,np.argmax(sph))
# sph = np.delete(sph,0)
# sph = np.delete(sph,len(sph)-1)
# print(new_sph)
fig = plt.figure()
ax = plt.axes(projection = '3d')
layers = []
for i in range(10,90):
	layers.append([sph[i,0],sph[i,1],sph[i,2]])
	ax.scatter(sph[i,0],sph[i,1],sph[i,2])
print(len(layers))
plt.show()
p_ctrlpts = layers
size_u = 8
size_v = 10
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
        size=5
    ),
    dict(
        points=surf_curves['v'][0].evalpts,
        name="v",
        color="black",
        size=5
    )
]
surf.delta = 0.025
surf.vis = vis.VisSurfWireframe()
surf.render(extras=plot_extras)

evalpts = np.array(surf.evalpts)

print(surf.derivatives(0,0)[0][0])