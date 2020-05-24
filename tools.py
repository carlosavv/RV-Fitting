import numpy as np

def preProcess(X):
	
	xmean = np.mean(X[:,0])
	ymean = np.mean(X[:,1])
	zmean = np.mean(X[:,2])
	mean_data = np.array([xmean,ymean,zmean])
	Xnew = X - mean_data
	return Xnew

