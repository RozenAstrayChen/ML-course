import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def computeCost(X, y, theta):
    m = len(y)
    #  MSE
    temp = np.dot(X, theta) - y
    return np.sum(np.power(temp, 2)) / (2*m)

'''
This is ex1data2 cost function which is recommend in pdf
J(θ)) = 1/2m * (Xθ - y)^T * (Xθ - y)
'''
def computeCost2(X, y, theta):
    m = len(y)
    temp = np.dot(X, theta) - y
    J = np.dot(temp.T, temp) / 2*m

    return J

# theta = theta - (alpha/m) * (X' * (X * theta - y ));
def gradientDescent(X, y, theta, alpha, iterations):
    m = len(y)
    for _ in range(iterations):
        # [97x2] * [2x1] = [97*1]
        temp = np.dot(X, theta) - y
        # mutlipie transpoce T
        # [2x97] * [97*1] = [2*1]
        temp = np.dot(X.T, temp)
        #print('transpoce', len(temp))
        theta = theta - (alpha/m) * temp
    return theta
'''
[[-2.60208521e-17]
 [ 9.52411140e-01]
 [-6.59473141e-02]]
'''
def normalEquation(X, y):
    temp = np.dot(X.T, X)
    temp = np.linalg.pinv(temp)
    theta = np.dot(temp, X.T).dot(y)
    return theta
    
    '''
    X_transpose = np.transpose(X)
    theta = np.linalg.inv(X_transpose.dot(X))
    theta = theta.dot(X_transpose)
    theta = theta.dot(y)
    return theta
    '''

def normalize(x):
    print(x.min().shape)
    s1 = x.max() - x.min()
    mu = x.mean()
    x = (x - mu) / s1
    return x

def normalize2(x):
    return (x - np.mean(x))/np.std(x)


def predict(x, theta):
    prediction = np.dot(x, theta)

    return prediction