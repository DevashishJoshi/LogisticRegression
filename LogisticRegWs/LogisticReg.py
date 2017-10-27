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
alpha = 1
lambda_ = 0
num_iters = 10000

#Mean normalization of features
mean = np.mean(X[:,1:], axis=0)
sd = np.std(X[:,1:], axis=0)
X[:,1:] = (X[:,1:]-mean)/sd

initial_theta = np.zeros((n, 1), dtype=float)

#Find optimum theta
print "Find optimum theta\n"
dict = gradDesc(initial_theta, X, y, alpha, num_iters, lambda_)
theta = dict['theta']
print theta
#print cost(theta, X, y, lambda_)
j_history = dict['j_history']

#Plot j_history
print ""
print "Plotting j_history"
p.plot(j_history)
p.xlabel('Number of iterations')
p.ylabel('Cost')
p.show()

#Calculate accuracy
print ""
print "Cost :"
print cost(theta, X, y, lambda_)
print "Accuracy on training set is "
print calcAccuracy(theta, X, y)

#Initialize test set variables
Xtest = test[1:, :-1]
mtest = Xtest.shape[0]
Xtest = np.append(np.ones( (mtest, 1) ), Xtest, axis=1)
ytest = np.array( [test[1:, -1]] )
ytest = ytest.T

#Mean normalize features in test set
Xtest[:,1:] = (Xtest[:,1:]-mean)/sd

#Predict for test set examples
print "Cost :"
print cost(theta, Xtest, ytest, lambda_)
print "Accuracy on test set is "
print calcAccuracy(theta, Xtest, ytest)
