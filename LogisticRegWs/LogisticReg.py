print "Starting..."
print "Importing all packages"
import numpy as np
import matplotlib.pyplot as p
from sigmoid import sigmoid
from costAndGrad import cost, grad
from gradientDesc import gradDesc
from calcAccuracy import calcAccuracy

# Load the data
print "Loading the data"
train = np.genfromtxt("train.csv", delimiter=",")
test = np.genfromtxt("test.csv", delimiter=",")

print "Initializing the variables"
#Initialize training set variables
y = np.array( [train[1:, -1]] )
y = y.T
m = y.shape[0]

X = train[1:, :-1]
X = np.append(np.ones( (m, 1) ), X, axis=1)
n = X.shape[1]

#Initialize test set variables
Xtest = test[1:, :-1]
mtest = Xtest.shape[0]
Xtest = np.append(np.ones( (mtest, 1) ), Xtest, axis=1)
ytest = np.array( [test[1:, -1]] )
ytest = ytest.T

#Mean normalization of features
mean = np.mean(X[:,1:], axis=0)
sd = np.std(X[:,1:], axis=0)
X[:,1:] = (X[:,1:]-mean)/sd

#Mean normalize features in test set
Xtest[:,1:] = (Xtest[:,1:]-mean)/sd

#Initialize the parameters
alpha = 1
lambda_ = 0
num_iters = 10000

initial_theta = np.zeros((n, 1), dtype=float)

#Find optimum theta
print "Starting gradient descent"
dict = gradDesc(initial_theta, X, y, alpha, num_iters, lambda_)
theta = dict['theta']
j_history = dict['j_history']

#Plot j_history
print "Plotting j_history (Close window to continue)"
p.plot(j_history)
p.xlabel('Number of iterations')
p.ylabel('Cost')
p.show()

#Predict for test set examples
print "Accuracy :"
print calcAccuracy(theta, Xtest, ytest)
