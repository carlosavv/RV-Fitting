import numpy as np 

def compute_metric_tensor(surf,uv):
	'''
	args: uv - array: linspace from [0,1] to index the metric tensor
	returns: metric tensor array where each column corresponds to the respective index of uv 
	'''
	E = []
	F = []
	G = []
	for i in range(0,len(uv)):
		E.append(np.dot(surf.derivatives(uv[i],uv[i],1)[1][0],surf.derivatives(uv[i],uv[i],1)[1][0]))
		F.append(np.dot(surf.derivatives(uv[i],uv[i],1)[1][0],surf.derivatives(uv[i],uv[i],1)[0][1]))
		G.append(np.dot(surf.derivatives(uv[i],uv[i],1)[0][1],surf.derivatives(uv[i],uv[i],1)[0][1]))

	return np.array([ E,F,F,G ]).T

def compute_strain(m1,m2):
	return 0.5*(m2-m1)

def compute_stretch(eps,m1,m2):
	# print("eps", eps)
	# print("m2", m2)
	lambda_u = 1/( np.sqrt(1 - 2*(eps[0]/m2[0]) ) )
	lambda_v = 1/( np.sqrt(1 - 2*(eps[1]/m2[3]) ) )
	return [lambda_u,lambda_v]
