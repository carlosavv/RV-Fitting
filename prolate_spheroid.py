import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from geomdl import fitting
from geomdl import convert
from geomdl import construct
from geomdl import exchange 
from geomdl.visualization import VisPlotly as vis
plt.style.use('seaborn-bright')

class Prolate_Spheroid(object):
    # constructor with parameters to generate a prolate spheriod 
    def __init__(self,mu,nu,phi,a):
        self.mu = mu
        self.nu = nu 
        self.phi = phi 
        self.a = a 

    # given the parameters
    def compute_prolate_spheroid(self):
        x = self.a*np.sinh(self.mu)*np.sin(self.nu)*np.cos(self.phi)
        y = self.a*np.sinh(self.mu)*np.sin(self.nu)*np.sin(self.phi)
        z = self.a*np.cosh(self.mu)*np.cos(self.nu)
        return x,y,z

'''

for the ED phase:
    a = 12
    mu = 0.75

for the ES phase:
    a = 10
    mu = 0.5

'''
# a = 12
# mu = 0.75
a = 10
mu = .5
nu = np.linspace(0,np.pi,10)
phi = np.linspace(0,2*np.pi,10)

sph = []

# create a prolate spheroid object and generate a regular point-cloud

for i in range(0,len(nu)):
    for j in range(0,len(phi)):
        s = Prolate_Spheroid(mu,nu[i],phi[j],a)
        sph.append(s.compute_prolate_spheroid())

spheroid = np.array(sph)

# visualize the point-cloud
fig = plt.figure(dpi = 120)
ax = plt.axes(projection = '3d')
layers = []

for i in range(10,70):
    layers.append([spheroid[i,0],spheroid[i,1],spheroid[i,2]])
    ax.scatter(spheroid[i,0],spheroid[i,1],spheroid[i,2],s = 50,color = 'r')
plt.axis('off')

print(len(layers))
# plt.show()

# surface fit parameters
p_ctrlpts = layers
size_u = 6
size_v = 10
degree_u = 3
degree_v = 3


# Do global surface approximation
surf = fitting.approximate_surface(p_ctrlpts, size_u, size_v, degree_u, degree_v)

# convert to NURBS 
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
surf.delta = 0.02
surf.vis = vis.VisSurface()
# surf.render(extras=plot_extras)
# exchange.export_obj(surf,"spheroid_es.obj")

# get the evaluated surface points
evalpts = np.array(surf.evalpts)

# compute the metric tensor

E = []
F = []
G = []
uv = np.linspace(0,1,11)

for i in range(0,len(uv)):
    E.append(np.dot(surf.derivatives(uv[i],uv[i],1)[1][0],surf.derivatives(uv[i],uv[i],1)[1][0]))
    F.append(np.dot(surf.derivatives(uv[i],uv[i],1)[1][0],surf.derivatives(uv[i],uv[i],1)[0][1]))
    G.append(np.dot(surf.derivatives(uv[i],uv[i],1)[0][1],surf.derivatives(uv[i],uv[i],1)[0][1]))

metric_tensor = np.array([E,F,F,G]).T
print(metric_tensor)