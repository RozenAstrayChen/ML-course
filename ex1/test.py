import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
'''
X = np.asarray([[1],[2],[4],[0]])
y = np.asarray([[0.5],[1],[2],[0]])
m = len(y)
#print(X)
ones = np.ones((m, 1))
#print(ones)
X = np.hstack((ones, X)) # adding the intercept term
#print('train', X)
theta = np.zeros([1,2])



def normal_equation(theta, X, y):
    temp = np.linalg.pinv(X.dot(X.T))
    print(temp.shape)
    theta = temp.dot(X)
    theta = theta.T.dot(y)

    return theta

def predict(theta, X):
    y = X.dot(theta)
    return y

theta = normal_equation(theta, X, y)
testX = np.array([1,1])
testY = predict(theta, testX)
print(theta)
'''
a = np.asarray([[1,2],[3,4],[5,6]])
print(a.shape)