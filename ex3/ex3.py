from scipy.io import loadmat
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from reg import *

data = loadmat('ex3data1.mat')
X = data['X']
y = data['y']

# visulaizing the data
_, axarr = plt.subplots(10, 10, figsize=(10, 10))
for i in range(10):
    for j in range(10):
        axarr[i, j].imshow(X[np.random.randint(X.shape[0])].reshape((20, 20),
                                                                    order='F'))
        axarr[i, j].axis('off')

#plt.show()

# adding the intercept term
m = len(y)
ones = np.ones((m, 1))
X = np.hstack((ones, X))  # add the intercept
(m, n) = X.shape
lmbda = 0.1
k = 10
theta = np.zeros((k, n))  # inital parameters

for i in range(k):
    digit_class = i if i else 10
    theta[i] = opt.fmin_cg(
        f=costFunctionReg,
        x0=theta[i],
        fprime=gradRegularization,
        args=(X, (y == digit_class).flatten(), lmbda),
        maxiter=50)

pred = np.argmax(X @ theta.T, axis = 1)
pred = [e if e else 10 for e in pred]
result = np.mean(pred == y.flatten()) * 100

print('result = {} %'.format(result))
