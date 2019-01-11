import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linear_reg import computeCost, gradientDescent, predict

data = pd.read_csv('ex1data1.txt', header = None) #read from dataset
X = data.iloc[:,0] # read first column
y = data.iloc[:,1] # read second column
m = len(y) # number of training example
data.head() # view first few rows of the data

plt.scatter(X, y)
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
#plt.show()


X = X[:, np.newaxis]
y = y[:, np.newaxis]
theta = np.zeros([2, 1])
iterations = 1500
alpha = 0.01
ones = np.ones((m, 1))
print('train', X.shape)
X = np.hstack((ones, X)) # adding the intercept term


J = computeCost(X, y, theta)
print(J)



theta = gradientDescent(X, y, theta, alpha, iterations)
print(theta)

J = computeCost(X, y, theta)
print(J)

testX = np.array([18.9803])
testX = testX[:, np.newaxis]
ones = np.ones((1, 1))
testX = np.hstack((ones, testX)) # adding the intercept term
testY = predict(testX, theta)

print('test population is  {}, and predict result is {}'.format(testX[0][1], testY))

plt.scatter(X[:,1], y)
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.plot(X[:,1], np.dot(X, theta))
plt.show()


