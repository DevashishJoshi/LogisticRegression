import numpy as np
from sigmoid import sigmoid

def calcAccuracy(theta, X, y):
	m = X.shape[0]
	h = sigmoid( np.matmul(X, theta) )
	p = h>=0.5
	return np.sum(p==y)*1.0/m*100
