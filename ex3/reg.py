import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


'''
J(θ) = 1/m Sum[-y*log(h(x)) - (1-y)log(1-h(x))] + l/2m sum(θ^2)
'''
def costFunctionReg(theta, X, y, lmbda):
    m = len(y)
    temp1 = np.multiply(y, np.log(sigmoid(np.dot(X, theta)))) 
    temp2 = np.multiply(1-y, np.log(1-sigmoid(np.dot(X, theta))))

    cost = (-1/m)*np.sum(temp1 + temp2) + (lmbda/(2*m))*np.sum(theta[1:]**2) 
    
    return cost


'''
partial derivative + regulared
pJ(θ) = (1/m * sum(h(x)-y)*x) + l/mθ
'''
def gradRegularization(theta, X, y, lmbda):
    m = len(y)
    temp = sigmoid(np.dot(X, theta)) - y
    temp = (np.dot(temp.T, X).T / m) + (theta * (lmbda / m))
    temp[0] = temp[0] - theta[0] * lmbda/m
    return temp
