import numpy as np 
import matplotlib.pyplot as plt
from geomdl import fitting
from geomdl.visualization import VisMPL as vis
from geomdl import construct
from geomdl import convert
from geomdl	import exchange
from utils import compute_metric_tensor, compute_strain, compute_stretch
import sys
sys.path.append("../")
from tools import preProcess

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

def compute_torus_xyz(theta,phi,R,r):
	x = (R + r*np.cos(2*np.pi*theta))*np.cos(np.pi/2*phi)
	y = (R + r*np.cos(2*np.pi*theta))*np.sin(np.pi/2*phi)
	z = r*np.sin(2*np.pi*theta)
	return x,y,z

def compute_centralAxis_xyz(phi,R):
	return R*np.cos(np.pi/2*phi),R*np.sin(np.pi/2*phi),0

N = 11
u = np.linspace(0,1,N)
v = np.linspace(0,1,N)
R = 80
r = 30
R_def = 60
r_def = 20 
toroid = []
caxis = []
toroid_def = []
caxis_def = []

for i in range(0,len(v)):
	for j in range(0,len(u)):
		toroid.append(compute_torus_xyz(u[i],v[j],R,r))
		caxis.append(compute_centralAxis_xyz(v[i],R))
		toroid_def.append(compute_torus_xyz(u[i],v[j],R_def,r_def))
		caxis_def.append(compute_centralAxis_xyz(v[i],R_def))

toroid = np.array(toroid)
toroid = preProcess(toroid)
toroid_def = np.array(toroid_def)
toroid_def = preProcess(toroid_def)
caxis = np.array(caxis)
caxis_def = np.array(caxis_def)

p_ctrlpts = toroid
size_u = N
size_v = N
degree_u = 3
degree_v = 3

# Do global surface approximation
surf = fitting.approximate_surface(p_ctrlpts, size_u, size_v, degree_u, degree_v)
surf_def = fitting.approximate_surface(toroid_def, size_u, size_v, degree_u, degree_v)

surf = convert.bspline_to_nurbs(surf)
surf_def = convert.bspline_to_nurbs(surf_def)

# toroid_pcl = np.array(surf.evalpts)
# toroid_cpts = np.array(surf.ctrlpts)

uv = np.linspace(0,1,N)
nurbs_fit_G = compute_metric_tensor(surf,uv)
nurbs_fit_G_def = compute_metric_tensor(surf_def,uv)
E = []
F = 0*np.ones((len(uv)))
G = []
E_def =[]
G_def =[]
for i in range(0,len(uv)):
	E.append((R + r*np.cos(2*np.pi*uv[i]))**2)
	G.append(r**2)

	E_def.append((R_def + r_def*np.cos(2*np.pi*uv[i]))**2)
	G_def.append(r_def**2)
E_fit = nurbs_fit_G[:,0] 
F_fit = nurbs_fit_G[:,1]  
F_fit = nurbs_fit_G[:,2]  
G_fit = nurbs_fit_G[:,3]

E_fit_d = nurbs_fit_G_def[:,0] 
F_fit_d = nurbs_fit_G_def[:,1]  
F_fit_d = nurbs_fit_G_def[:,2]  
G_fit_d = nurbs_fit_G_def[:,3]

#idk why but E and G are switched
nurbs_fit_G = np.array([G_fit/(1/4*np.pi**2),F_fit,F_fit,E_fit/(4*np.pi**2)]).T
analytical_G = np.array([E,F,F,G]).T

nurbs_fit_G_def = np.array([G_fit_d/(1/4*np.pi**2),F_fit_d,F_fit_d,E_fit_d/(4*np.pi**2)]).T
analytical_G_def = np.array([E_def,F,F,G_def]).T

# print(nurbs_fit_G)
# print(analytical_G) 

strain = []
stretch = []
a_strain = []
a_stretch = []

for i in range(0,len(nurbs_fit_G)):
	a_strain.append(compute_strain(analytical_G[i],analytical_G_def[i]))
	a_stretch.append(compute_stretch(a_strain[i],analytical_G[i],analytical_G_def[i]))
	
	strain.append(compute_strain(nurbs_fit_G[i],nurbs_fit_G_def[i]))
	stretch.append(compute_stretch(strain[i],nurbs_fit_G[i],nurbs_fit_G_def[i]))

a_strain = np.array(a_strain)
a_stretch = np.array(a_stretch)

strain = np.array(strain)
stretch = np.array(stretch)

print(stretch)
print(a_stretch)
# fig = plt.figure(dpi = 175)
# ax = plt.axes(projection = '3d')
# # plt.style.use('dark_background')
# ax.scatter(toroid[:,0],toroid[:,1],toroid[:,2], color = 'green')
# ax.plot(caxis[:,0],caxis[:,1],caxis[:,2],color = 'red')
# # ax.scatter(toroid_pcl[:,0],toroid_pcl[:,1],toroid_pcl[:,2],color = "green")
# ax.axis("off")
# plt.show()


# fig = plt.figure(dpi = 175)
# ax = plt.axes(projection = '3d')
# plt.style.use('dark_background')
# ax.scatter(toroid_def[:,0],toroid_def[:,1],toroid_def[:,2], color = 'blue')
# ax.plot(caxis_def[:,0],caxis_def[:,1],caxis_def[:,2],color = 'red')
# # ax.scatter(toroid_pcl[:,0],toroid_pcl[:,1],toroid_pcl[:,2],color = "green")
# ax.axis("off")
# plt.show()

# Extract curves from the approximated surface
# surf_curves = construct.extract_curves(surf)
# plot_extras = [
# 	dict(
# 		points=surf_curves['u'][0].evalpts,
# 		name="u",
# 		color="red",
# 		size= 8
# 	),
# 	dict(
# 		points=surf_curves['v'][0].evalpts,
# 		name="v",
# 		color="black",
# 		size= 8
# 	)
# ]

# surf_curves = construct.extract_curves(surf_def)
# plot_extras = [
# 	dict(
# 		points=surf_curves['u'][0].evalpts,
# 		name="u",
# 		color="red",
# 		size= 8
# 	),
# 	dict(
# 		points=surf_curves['v'][0].evalpts,
# 		name="v",
# 		color="black",
# 		size= 8
# 	)
# ]
import matplotlib as mpl  
import matplotlib.colors as mcolors
c = mcolors.ColorConverter().to_rgb

rvb_u = make_colormap(
    [c('red'), c('green'),0,
     c('green'),c('blue'),1,c('blue')])

# rvb_u = make_colormap(
#     [c('red'), min(a_stretch[:,1])/max(a_stretch[:,1]),c('red'), c('green'),(max(a_stretch[:,1])/max(a_stretch[:,1]) - min(a_stretch[:,1])/max(a_stretch[:,1]))/2,
#      c('green'),c('blue'),max(a_stretch[:,1])/max(a_stretch[:,1]), c('blue')])


surf.delta = 0.02
surf.vis = vis.VisSurface()
surf.render(colormap=rvb_u)

# surf_def.delta = 0.02
# surf_def.vis = vis.VisSurface()
# surf_def.render(colormap=mpl.cm.jet)

fig, ax = plt.subplots()
fig.subplots_adjust()

norm_u = mpl.colors.Normalize(vmin = min(a_stretch[:,0]), vmax = min(a_stretch[:,0]))
norm_v = mpl.colors.Normalize(vmin = min(a_stretch[:,1]), vmax = max(a_stretch[:,1]))

fig.colorbar(mpl.cm.ScalarMappable(norm=norm_v, cmap=rvb_u),
             cax=ax, orientation='vertical')
plt.show()
# exchange.export_obj(surf,"torus.obj")
# exchange.export_obj(surf_def,"torus_def.obj")