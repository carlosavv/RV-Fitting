from cylinder import Cylinder 
from cylinder import compute_metric_tensor 
from cylinder import compute_strain, compute_stretch
import numpy as np 
from geomdl import fitting
from geomdl.visualization import VisMPL as vis 
from geomdl import construct
from geomdl import convert
from geomdl	import exchange
import random as rand
import matplotlib
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import matplotlib.colors as mcolors
plt.style.use("seaborn")


def make_colormap(seq):
    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap('CustomMap', cdict)


N = 11
M = 2
eps1 = 0.1
eps2 = 0.1
z = np.linspace(0,1,N)
theta = np.linspace(0,1,N)


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

r_ed = np.array(r_ed)
fig = plt.figure()
ax = Axes3D(fig)

ax.scatter(r_ed[:,0],r_ed[:,1],r_ed[:,2])

# print(len(drdt_ed))
pts_ed = r_ed
size_u = N
size_v = N
degree_u = 3
degree_v = 3

# Do global surface approximation
surf_ed = fitting.approximate_surface(pts_ed, size_u, size_v, degree_u, degree_v)
surf_ed = convert.bspline_to_nurbs(surf_ed)

surf_ed.delta = 0.025
vis_config = vis.VisConfig(legend=True, axes=False, figure_dpi=120,ctrlpts = False)
surf_ed.vis = vis.VisSurface(vis_config)
from matplotlib import cm


evalpts_ed = np.array(surf_ed.evalpts)

pts_es = r_es

# Do global surface approximation
surf_es = fitting.approximate_surface(pts_es, size_u, size_v, degree_u, degree_v)
surf_es = convert.bspline_to_nurbs(surf_es)

surf_es.delta = 0.025
surf_es.vis = vis.VisSurface()
# surf_es.render()
evalpts_es = np.array(surf_es.evalpts)

# exchange.export_obj(surf_ed,"sin_clyinder_ed.obj")
# exchange.export_obj(surf_es,"sin_clyinder_es.obj")

# compute analytical metric tensor for ed phase
a_E_ed = []
a_F_ed = []
a_G_ed = []
print(len(drdt_ed))
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

print('NURBS (ed) fit metric tensor = ')
# print(metric_tensor_ed)

# compute the NURBS metric tensor (es phase)
metric_tensor_es = compute_metric_tensor(surf_es,uv)

# print('NURBS (es) fit metric tensor = ')
# print(metric_tensor_es)

strain = []
stretch = []
a_strain = []
a_stretch = []
for i in range(0,len(metric_tensor_ed)):
	a_strain.append(compute_strain(a_metric_tensor_ed[i],a_metric_tensor_es[i]))
	a_stretch.append(compute_stretch(a_strain[i],a_metric_tensor_ed[i],a_metric_tensor_es[i]))
	
	strain.append(compute_strain(metric_tensor_ed[i],metric_tensor_es[i]))
	stretch.append(compute_stretch(strain[i],metric_tensor_ed[i],metric_tensor_es[i]))

strain = np.array(strain)
stretch = np.array(stretch)
print(strain)
print(stretch)

c = mcolors.ColorConverter().to_rgb
rvb = make_colormap(
    [c('red'), c('green'), min(stretch[:,1])/max(stretch[:,1]), c('green'), c('blue'), max(stretch[:,1])/max(stretch[:,1]), c('blue')])

import matplotlib as mpl

fig, ax = plt.subplots()

# cmap = mpl.cm.cool
norm = mpl.colors.Normalize(vmin=min(stretch[:,1]), vmax=max(stretch[:,1]))


fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=rvb),
             cax=ax, orientation='vertical')
surf_curves = construct.extract_curves(surf_ed)
plot_extras = [
    dict(points=surf_curves["u"][0].evalpts, name="u", color="white", size=5),
    dict(points=surf_curves["v"][0].evalpts, name="v", color="black", size=5),
]
surf_ed.render( extras = plot_extras)
# surf_es.render(colormap = rvb)