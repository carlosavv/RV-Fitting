import sinusoidal_cylinder 
from cylinder import compute_metric_tensor 
from cylinder import compute_strain, compute_stretch
from geomdl.visualization import VisMPL as vis 
from geomdl import construct
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

surf_ed,surf_es = sinusoidal_cylinder()

surf_ed.delta = 0.025
vis_config = vis.VisConfig(legend=True, axes=False, figure_dpi=120,ctrlpts = False)
surf_ed.vis = vis.VisSurface(vis_config)
evalpts_ed = np.array(surf_ed.evalpts)

surf_es.delta = 0.025
surf_es.vis = vis.VisSurface()
evalpts_es = np.array(surf_es.evalpts)

# compute analytical metric tensor for ed phase
a_E_ed = []
a_F_ed = []
a_G_ed = []

for i in range(0,len(drdt_ed)):
	a_E_ed.append(np.dot(drdz_ed[i],drdz_ed[i]))
	a_F_ed.append(np.dot(drdz_ed[i],drdt_ed[i]))
	a_G_ed.append(np.dot(drdt_ed[i],drdt_ed[i]))

a_metric_tensor_ed = np.array([a_E_ed,a_F_ed,a_F_ed,a_G_ed]).T

# compute analytical metric tensor for es phase
a_E_es = []
a_F_es = []
a_G_es = []
for i in range(0,len(drdt_es)):
	a_E_es.append(np.dot(drdz_es[i],drdz_es[i]))
	a_F_es.append(np.dot(drdz_es[i],drdt_es[i]))
	a_G_es.append(np.dot(drdt_es[i],drdt_es[i]))

a_metric_tensor_es = np.array([a_E_es,a_F_es,a_F_es,a_G_es]).T

# compute the NURBS metric tensor (ed phase)
uv = np.linspace(0,1,11)
metric_tensor_ed = compute_metric_tensor(surf_ed,uv)


# compute the NURBS metric tensor (es phase)
metric_tensor_es = compute_metric_tensor(surf_es,uv)


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
# exchange.export_obj(surf_ed,"sin_clyinder_ed.obj")
# exchange.export_obj(surf_es,"sin_clyinder_es.obj")