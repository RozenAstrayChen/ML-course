import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linear_reg import *
'''
data
The first column is the size of the house (in square feet).
the second column is the number of bedrooms
the third column is the price of the house.
'''
data = pd.read_csv('ex1data2.txt', sep = ',', header = None)
X = data.iloc[:,0:2] # read first two columns into X
y = data.iloc[:,2] # read the third column into y
m = len(y) # no. of training samples
data.head()


X = normalize2(X)

ones = np.ones((m, 1))
X = np.hstack((ones, X))
alpha = 0.01
num_iters = 1500
theta = np.zeros((3,1))
y = y[:, np.newaxis]


J = computeCost(X, y, theta)
print('before gradient, cost function = ', J)


'''
theta = gradientDescent(X, y, theta, alpha, iterations)
'''
theta = normalEquation(X, y)
print('theta = ', theta)

J = computeCost(X, y, theta)
print('cost function', J)

testX = np.array([1, 2104,3])
testY = predict(testX, theta)
print('test population is  {}, and predict result is {}'.format(testX, testY))


plt.scatter(X[:,1], y)
plt.xlabel('the size of the house')
plt.ylabel('price of the house')
plt.plot(X[:,1], np.dot(X, theta))
plt.show()

plt.scatter(X[:,2], y)
plt.xlabel('the size of the house')
plt.ylabel('price of the house')
plt.plot(X[:,1], np.dot(X, theta))
plt.show()
