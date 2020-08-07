import numpy as np 
import matplotlib.pyplot as plt 
from geomdl.visualization import VisMPL as vis
import scipy.spatial.distance as dist
'''
todo: compare how the control points move from one frame to the other (toroid experiment)

'''

def distance(a,b):
	return np.sqrt( (a[:,0]-b[:,0])**2 + (a[:,1]-b[:,1])**2 + (a[:,2]-b[:,2])**2)


con_toroid_cpts= np.loadtxt("con_toroid_cpts.csv",delimiter = ',')

toroid_cpts = np.loadtxt("toroid_cpts.csv",delimiter = ',')
# print(toroid)
toroid = np.loadtxt('tapered_toroid.dat',delimiter=' ')
con_toroid = np.loadtxt('con_tapered_toroid.dat',delimiter=',')
plt.style.use('seaborn-dark')
fig = plt.figure()
ax = plt.axes(projection = "3d")


# ax.quiver(toroid[:,0],toroid[:,1],con_toroid[:,0],con_toroid[:,1])
# ax.scatter(toroid[:,0],toroid[:,1],toroid[:,2],color = 'r')
# ax.scatter(con_toroid[:,0],con_toroid[:,1],con_toroid[:,2], color= 'g', marker = '*')


distances = []
distances_cpts = []

# test = []
for i in range(0,len(toroid)):
	distances.append(dist.euclidean(toroid[i],con_toroid[i]))
for i in range(0,len(toroid_cpts)):
	distances_cpts.append(dist.euclidean(toroid_cpts[i],con_toroid_cpts[i]))
# test = distance(toroid,con_toroid)
print(min(distances))
# print(test[0])
fig = plt.figure(dpi = 150)

plt.plot(distances,color = 'b', label = 'Surface Points')
plt.xlabel('data point')
plt.ylabel('Euclidean Distance')
plt.title('Distance between N1 and N2')
plt.legend()


fig = plt.figure(dpi = 150)
plt.plot(distances_cpts,color = 'r', label = 'Control Points')
plt.xlabel('data point')
plt.ylabel('Euclidean Distance')
plt.title('Distance between N1 and N2')

plt.legend()
# plt.plot(test)
plt.show()