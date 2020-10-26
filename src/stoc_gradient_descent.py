#file for Stochastic gradient descent
import pandas as pd
import numpy as np
import pre_processing
from numpy.linalg import inv
from matplotlib import pyplot
import random

def error(y,yi):
    return (np.mean((y-yi)**2))**(1/2)


def sgd(x, y, rate):
    beta=np.ones((4,1))
    for iteration in range(0,10000):
        i= random.randint(0,x.shape[0]-1)
        xi = x[i].reshape((1,4))
        yi = y[i].reshape((1,1))
        diff=xi.dot(beta)-yi
        beta = beta-rate*(xi.T.dot(diff))
        if(iteration%50==0):
            print(error(y,x.dot(beta)))
    return beta
    



data=pd.read_csv("insurance.txt").to_numpy()
training_data=pre_processing.standardization(data)
train_error=[]
test_error=[]
for i in range(0,20):
    (training_data,testing_data)=pre_processing.train_test_split(data)
    x=training_data[:,0:3]
    y_train_org=training_data[:,3]
    ones=np.ones((x.shape[0],1))
    x=np.append(ones,x,axis=1)
    y_train_org=np.reshape(y_train_org,(x.shape[0],1))
    learning_rate=0.1/(x.shape[0])
    beta=sgd(x,y_train_org,learning_rate)
    y_train=x.dot(beta)
    train_error.append(error(y_train,y_train_org))
    x_test=testing_data[:,0:3]
    y_test_org=testing_data[:,3]
    ones=np.ones((x_test.shape[0],1))
    x_test=np.append(ones,x_test,axis=1)
    y_test=x_test.dot(beta)
    test_error.append(error(y_test_org,y_test))
    print("\n")

train_error_mean=np.mean(train_error)
train_error_var=np.var(train_error)
train_error_min=min(train_error)
test_error_mean=np.mean(test_error)
test_error_var=np.var(test_error)
test_error_min=min(test_error)
    
