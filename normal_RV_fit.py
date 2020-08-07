import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from slice import slice



def split_into_angles(M,layers):

	'''
	function that splits data in angled segments
	'''
	theta = np.linspace(layers[:,1].min(),layers[:,1].max(),M+1)

	points = []
	for i in range(len(theta)-1):
		points.append(layers[(layers[:, 1] > theta[i]) & (layers[:, 1] < theta[i + 1])])
	data = np.array(points)
	fig = plt.figure()
	ax = plt.axes(projection="3d")
	t = []
	for i in range(len(data)):
		t.append([data[i][:, 0], data[i][:, 1], data[i][:, 2]])
		# ax.scatter(t[i][:, 0], t[i][:, 1], t[i][:, 2])
	data = np.array(t)


	for i in range(len(data)):
		ax.scatter(data[i][0], data[i][1], data[i][2])
	return data


data = np.loadtxt('N2_RV_P0.dat')
fig = plt.figure()
ax = plt.axes(projection = '3d')
# ax.scatter(data[:,0],data[:,1],data[:,2])

N = 5
slice(N, data)
bins = slice.bins
mean_per_slice = []
# slice.slices.pop(0)
# slice.slices.pop(-1)
temp = []
for i in range(0,len(slice.slices)):
	# ax.scatter(slice.slices[i][:,0],slice.slices[i][:,1],slice.slices[i][:,2])
	temp.append([slice.slices[i][:,0],slice.slices[i][:,1],slice.slices[i][:,2]])
	mean_per_slice.append([np.mean(slice.slices[i][:,0]),np.mean(slice.slices[i][:,1]),np.mean(slice.slices[i][:,2])])

ax.scatter(slice.slices[1][:,0],slice.slices[1][:,1],slice.slices[1][:,2])

test = np.array([slice.slices[1][:,0],slice.slices[1][:,1],slice.slices[1][:,2]])
temp = np.array(temp)
layers = []

# store all slices into layers array
for i in range(0,len(temp)):
	for j in range(0,len(temp[i][0])):
		layers.append([temp[:,0][i][j],temp[:,1][i][j],temp[:,2][i][j]])

layers = np.array(layers)
mean_per_slice = np.array(mean_per_slice)



ax.plot(mean_per_slice[:,0],mean_per_slice[:,1],mean_per_slice[:,2],linewidth = '2')
# ax.scatter(data[:,0],data[:,1],data[:,2], color = 'r')

# create starting parameters for helix
M = 10
t = np.linspace(0,2*np.pi/3,M)
a = np.sqrt(test[0,0]**2+test[0,1]**2)
b = 30

s = []
C = a**2+b**2
for i in range(0,len(t)):
    s.append(np.sqrt(C)*t[i])

r = []
T = []
N = []
B = []


for i in range(0,len(s)):
    # generate a helical axis first
    r.append([a*np.cos(s[i]/np.sqrt(C)),a*np.sin(s[i]/np.sqrt(C)),b*s[i]/np.sqrt(C)])

    # create the tangential, normal, and binormal vectors
    T.append([-a/np.sqrt(C)*np.sin(s[i]/np.sqrt(C)),a/np.sqrt(C)*np.cos(s[i]/np.sqrt(C)),b/np.sqrt(C)])
    N.append([-np.cos(s[i]/np.sqrt(C)),-np.sin(s[i]/np.sqrt(C)),0])

B.append(np.cross(T,N))

# store them as numpy arrays for convenience
r = np.array(r)
Ts = np.array(T)
Ns = np.array(N)
Bs = np.array(B[0])

# scatter the T, N, and B vectors
fig = plt.figure()
ax = plt.axes(projection = "3d")

# these scatter points serves as a check to make sure that the T, N , B vectors work 

# ax.plot(r[:,0],r[:,1],r[:,2],color = 'r')
# ax.scatter(r[5,0]+Ts[5,0],r[5,1]+Ts[5,1],r[5,2]+Ts[5,2],color = 'b')
# ax.scatter(r[5,0],r[5,1],r[5,2],color = 'k')
# ax.scatter(r[5,0]-Ts[5,0],r[5,1]-Ts[5,1],r[5,2]-Ts[5,2],color = 'g')

# ax.scatter(Bs[:,0],Bs[:,1],Bs[:,2],color = 'g')
# ax.scatter(Ns[:,0],Ns[:,1],Ns[:,2],color = 'b')



helix = []
phi = np.linspace(0,2*np.pi,M)
d = 10
for i in range(0,len(s)):
    for j in range(0,len(phi)):
        helix.append([  d*Bs[i,0]*np.cos(phi[j])+d*Ns[i,0]*np.sin(phi[j])+r[i,0],
                        d*Bs[i,1]*np.cos(phi[j])+d*Ns[i,1]*np.sin(phi[j])+r[i,1],
                        d*Bs[i,2]*np.cos(phi[j])+d*Ns[i,2]*np.sin(phi[j])+r[i,2]
                    ])

helix = np.array(helix)

ax.scatter(helix[:,0],helix[:,1],helix[:,2])
ax.set_title("Helical Tube Point Cloud")
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()