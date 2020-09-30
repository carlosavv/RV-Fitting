import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from geomdl import fitting
from geomdl import convert
from geomdl import construct
from geomdl.visualization import VisMPL as vis
from slice import slice
plt.style.use('seaborn')

def p_spheroid(mu,nu,phi,a = 10):
	x = a*np.sinh(mu)*np.sin(nu)*np.cos(phi)
	y = a*np.sinh(mu)*np.sin(nu)*np.sin(phi)
	z = a*np.cosh(mu)*np.cos(nu)
	return x,y,z
mu = 1
nu = np.linspace(0,np.pi,20)
phi = np.linspace(0,2*np.pi,20)

sph = []

for i in range(0,len(nu)):
	for j in range(0,len(phi)):
		sph.append(p_spheroid(mu,nu[i],phi[j]))


sph = np.array(sph)
# sph = np.delete(sph,0)
# sph = np.delete(sph,len(sph)-1)
fig = plt.figure()
ax = plt.axes(projection = '3d')
# ax.scatter(sph[:,0],sph[:,1],sph[:,2])

n = 8
slice(n,sph)
bins = slice.bins

temp = []
for i in range(4,8):
	for j in range(3,7):
		temp.append([slice.slices[i][:,0],slice.slices[i][:,1],bins[j]*np.ones(len(slice.slices[i][:,2]))])
temp = np.array(temp)
layers = []
for i in range(0,len(temp)):
	for j in range(0,len(temp[i][0])):
		layers.append([temp[:,0][i][j],temp[:,1][i][j],temp[:,2][i][j]])
layers = np.array(layers)
print(layers)

ax.scatter(layers[:,0],layers[:,1],layers[:,2],color = 'g')
plt.show()
print(len(layers))

p_ctrlpts = layers
size_u = 9
size_v = 20
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
surf.delta = 0.025
surf.vis = vis.VisSurface()
surf.render(extras=plot_extras)