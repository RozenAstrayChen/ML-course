import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.special import expit #Vectorized sigmoid function
from scipy import optimize

'''
#Logistic hypothesis function
'''
def h(theta, X): 
    return expit(np.dot(X, theta))

def costFunc(theta, X, y, l=0.):
    m = len(y)
    #note to self: *.shape is (rows, columns)
    term1 = np.dot(-np.array(y).T,np.log(h(theta,X)))
    term2 = np.dot((1-np.array(y)).T,np.log(1-h(theta,X)))
    #regterm = (lambda/2) * np.sum(np.dot(theta[1:].T,theta[1:])) #Skip theta0
    return float( (1./m) * ( np.sum(term1 - term2)  ) )
    #return float( (1./m) * ( np.sum(term1 - term2) + regterm ) )

def gradientDescent(theta, X, y):
    m = y.size
    #for _ in range(iterator):
    h = sigmoid(X.dot(theta))
    grad = (1/m)*X.T.dot(h-y)
        
    return(grad.flatten())

def loaddata(file, delimeter):
    data = np.loadtxt(file, delimiter=delimeter)
    print('Dimensions: ',data.shape)
    print(data[1:6,:])
    return(data)

def plotData(pos, neg):
    plt.figure(figsize=(10,6))
    plt.plot(pos[:,1],pos[:,2],'k+',label='Admitted')
    plt.plot(neg[:,1],neg[:,2],'yo',label='Not admitted')
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    plt.legend()
    plt.grid(True)


def predict(theta, X, threshold=0.5):
    return h(theta, X) >= 0.5

def optimizeTheta(theta, X, y, l=0.):
    result = optimize.fmin(costFunc, x0=theta, args=(X, y, l), maxiter=400, full_output=True)
    return result[0], result[1]



