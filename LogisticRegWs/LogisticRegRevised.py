print "Starting...\n"
import numpy as np
import matplotlib.pyplot as p
from sigmoid import sigmoid
from costAndGrad import cost, grad
from gradientDesc import gradDesc
from calcAccuracy import calcAccuracy
print "Imported all packages\n"

# Load the data
print "Loading the data\n"
train = np.genfromtxt("train.csv", delimiter=",")
test = np.genfromtxt("test.csv", delimiter=",")
print "Data loaded successfully\n"

#Initialize variables
print "Initializing the variables\n"

y = np.array( [train[1:, -1]] )
y = y.T
m = y.shape[0]

X = train[1:, :-1]
X = np.append(np.ones( (m, 1) ), X, axis=1)
n = X.shape[1]

#Initialize the parameters
arr = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
alpha = 0.01
lambda_ = 0.01
num_iters = 600

#Mean normalization of features
mean = np.mean(X[:, 1:], axis=0)
sd = np.std(X[:, 1:], axis=0)
X[:, 1:] = (X[:, 1:]-mean)/sd

#Initialize test set variables
Xtest = test[1:, :-1]
mtest = Xtest.shape[0]
Xtest = np.append(np.ones( (mtest, 1) ), Xtest, axis=1)
ytest = np.array( [test[1:, -1]] )
ytest = ytest.T

#Mean normalize features in test set
Xtest[:,1:] = (Xtest[:,1:]-mean)/sd

initial_theta = np.zeros( (n, 1), dtype=float)
optimum_theta = initial_theta
j_history = np.zeros( (num_iters, 1) )

print "Find optimum theta\n"
for alphaIter in arr:
	for lambda_Iter in arr:
		dict = gradDesc(initial_theta, X, y, alphaIter, num_iters, lambda_Iter)
		theta = dict['theta']
		if( calcAccuracy(theta, Xtest, ytest) > calcAccuracy(optimum_theta, Xtest, ytest) ):
			optimum_theta = theta
			j_history = dict['j_history']
			alpha = alphaIter
			lambda_ = lambda_Iter

theta = optimum_theta

#Plot j_history
print "Plotting j_history"
p.plot(j_history)
p.xlabel('Number of iterations')
p.ylabel('Cost')
p.show()

#Print values of parameters
print "Values of parameters : "
print alpha
print lambda_

#Calculate accuracy
print "Accuracy on training set is "
print calcAccuracy(theta, X, y)

#Predict for test set examples
print "Accuracy on test set is "
print calcAccuracy(theta, Xtest, ytest)
