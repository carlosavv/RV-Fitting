import numpy as np 
import matplotlib.pyplot as plt
from geomdl import fitting
from geomdl import convert
from geomdl import construct
from geomdl.visualization import VisMPL as vis
import sys
np.set_printoptions(threshold=sys.maxsize)
from tools import preProcess
from geomdl import exchange
from geomdl import knotvector
from geomdl import BSpline 
from generateHelix import helix

ctrlpts = np.loadtxt("test1.txt")
np.savetxt("candidate_cpts.csv",ctrlpts,delimiter = ',')
helix = helix()
c = np.array(helix.ctrlpts)
print(c)
fig = plt.figure()
ax = plt.axes(projection = '3d')
ax.scatter(c[:,0],c[:,1],c[:,2])

helix.ctrlpts = ctrlpts
helix.delta = 0.015
helix.vis = vis.VisSurface()
helix.render()
exchange.export_obj(helix, "trial0.obj")


