import numpy as np
from sigmoid import sigmoid
from costAndGrad import grad, cost

def gradDesc(theta, X, y, alpha, num_iters, lambda_):
	m = y.shape[0]
	n = X.shape[1]
	j_history = np.zeros( (num_iters) )
	k=1
	prev_rate=0
	for i in range(0, num_iters):
		theta = theta - np.array( alpha*grad(theta, X, y, lambda_) ).T
		j_history[i] = cost(theta, X, y, lambda_)
	#print "\n\n"
	return {'theta':theta, 'j_history':j_history}
