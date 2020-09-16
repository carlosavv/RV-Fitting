import numpy as np 
import matplotlib.pyplot as plt 
from geomdl.visualization import VisMPL as vis
import scipy.spatial.distance as dist
plt.style.use('seaborn-bright')



'''
todo: compare how the control points move from one frame to the other (toroid experiment)

'''

def distance(a,b):
	return np.sqrt( (a[:,0]-b[:,0])**2 + (a[:,1]-b[:,1])**2 + (a[:,2]-b[:,2])**2)


ed_cpts = np.loadtxt('cpts_N2_RV_P0.csv',delimiter = ',')
es_cpts = np.loadtxt('cpts_N2_RV_P4.csv',delimiter = ',')
print(len(ed_cpts))

fig = plt.figure(dpi = 125)
ax = plt.axes(projection = '3d')
ax.scatter(ed_cpts[:,0],ed_cpts[:,1],ed_cpts[:,2],s = 20,color = 'r')
ax.scatter(es_cpts[:,0],es_cpts[:,1],es_cpts[:,2],s = 20 ,color = 'b')
plt.axis('off')

fig = plt.figure()
ax = plt.axes(projection = '3d')
ax.set_xlabel("x axis")
ax.set_ylabel("y axis")
ax.set_zlabel("z axis")


ax.scatter(ed_cpts[:,0],ed_cpts[:,1],ed_cpts[:,2],s = 50,color = 'r')
ax.scatter(es_cpts[:,0],es_cpts[:,1],es_cpts[:,2],s = 50 ,color = 'b')
for i in range(0,len(ed_cpts)):
	ax.plot3D([ed_cpts[i,0],es_cpts[i,0]],[ed_cpts[i,1],es_cpts[i,1]],[ed_cpts[i,2],es_cpts[i,2]],color = 'k')

fig = plt.figure(dpi = 125)
ax = plt.axes(projection = '3d')
ax.set_xlabel("x axis")
ax.set_ylabel("y axis")
ax.set_zlabel("z axis")

for i in range(0,len(ed_cpts),17):
	ax.plot3D([ed_cpts[i,0],es_cpts[i,0]],[ed_cpts[i,1],es_cpts[i,1]],[ed_cpts[i,2],es_cpts[i,2]],color = 'k')
	ax.scatter(ed_cpts[i,0],ed_cpts[i,1],ed_cpts[i,2],s = 50,color = 'r')
	ax.scatter(es_cpts[i,0],es_cpts[i,1],es_cpts[i,2],s = 50,color = 'b')
# plt.axis('off')
plt.show()


