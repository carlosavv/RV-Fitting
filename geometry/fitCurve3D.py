import numpy as np
import matplotlib.pyplot as plt
from geomdl import fitting
from geomdl import exchange
from geomdl.visualization import VisMPL as vis
'''
todo: fit 3d curves to central axes of the toroid experiments
'''

con_toroid = exchange.import_csv("con_toroid_caxis.csv",delimiter = ',')
print(len(con_toroid))
toroid = exchange.import_csv("toroid_caxis.csv",delimiter = ',')
print(toroid)

curve = fitting.approximate_curve(toroid,degree = 2,ctrlpts_size = 10)
curve1 = fitting.approximate_curve(con_toroid,degree = 2,ctrlpts_size = 10-)
# Plot the interpolated curve
curve.delta = 0.005
curve.vis = vis.VisCurve3D(vis.VisConfig(ctrlpts=False))
curve.render()

# Plot the interpolated curve
curve1.delta = 0.005
curve1.vis = vis.VisCurve3D(vis.VisConfig(ctrlpts=False))
curve1.render()