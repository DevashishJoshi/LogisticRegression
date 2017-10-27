import numpy as np
from sigmoid import sigmoid

def cost(theta, X, y, lambda_):
	m = X.shape[0]
	h = sigmoid( np.matmul(X, theta) )

	J = (-1/m) * np.sum( y*np.log(h) + (1-y)*np.log(1-h), axis=0 )
	
	#Regularization term
	reg = lambda_/(2*m) * np.sum( theta[1:, :]**2, axis=0)
	
	J = J + reg
	return J

def grad(theta, X, y, lambda_):
	m = y.shape[0]
	h = sigmoid( np.matmul(X, theta) )
	
	#Regularization term
	reg = 1.0*lambda_/(2*m) * theta[1:, :]
	reg = reg.T
	
	g = np.array( [(1.0/m) * np.sum( ( X*(h-y) ), axis=0)] )
	g[:, 1:] = g[:, 1:]+reg

	return g
