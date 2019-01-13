# %load ../../standard_import.txt
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from utils import *
from scipy.special import expit  #Vectorized sigmoid function

datafile = 'ex2data1.txt'
#!head $datafile
cols = np.loadtxt(
    datafile, delimiter=',', usecols=(0, 1, 2),
    unpack=True)  #Read in comma separated data
##Form the usual "X" matrix and "y" vector
X = np.transpose(np.array(cols[:-1]))
y = np.transpose(np.array(cols[-1:]))
m = y.size  # number of training examples
##Insert the usual column of 1's into the "X" matrix
X = np.insert(X, 0, 1, axis=1)

#Divide the sample into two: ones with positive classification, one with null classification
pos = np.array([X[i] for i in range(X.shape[0]) if y[i] == 1])
neg = np.array([X[i] for i in range(X.shape[0]) if y[i] == 0])
#Check to make sure I included all entries
#print "Included everything? ",(len(pos)+len(neg) == X.shape[0])
plotData(pos, neg)

#Quick check that expit is what I think it is
myx = np.arange(-10, 10, .1)
plt.plot(myx, expit(myx))
plt.title("Woohoo this looks like a sigmoid function to me.")
plt.grid(True)
#plt.show()

theta = np.zeros((X.shape[1], 1))
J = costFunc(theta, X, y)
print('cost = ', J)
theta, mincost = optimizeTheta(theta, X, y)
J = costFunc(theta, X, y)
print('after train, cost = ', J)

boundary_xs = np.array([np.min(X[:, 1]), np.max(X[:, 1])])
boundary_ys = (-1. / theta[2]) * (theta[0] + theta[1] * boundary_xs)
plotData(pos, neg)
plt.plot(boundary_xs, boundary_ys, 'b-', label='Decision Boundary')
plt.legend()
#plt.show()

p = predict(theta, X)
print('Train accuracy {}%'.format(100 * sum(p == y.ravel()) / p.size))
