
import pandas as pd
import numpy as np
import pre_processing
from numpy.linalg import inv
from matplotlib import pyplot

def error(y,yi):
    return (np.mean((y-yi)**2))**(1/2)

def pred_y(beta,x):
    return (x.dot(beta))
data=pd.read_csv("insurance.txt").to_numpy()
training_data=pre_processing.standardization(data)
train_error=[]
test_error=[]
for i in range(0,20):
    (training_data,testing_data)=pre_processing.train_test_split(data)
    x=training_data[:,0:3]
    y_org=training_data[:,3]
    ones=np.ones((936,1))
    x=np.append(ones,x,axis=1)
    count=0
    beta=np.ones((4,1))
    rate=0.1/936
    y_org=np.reshape(y_org,(936,1))
    y_train=x.dot(beta)
    diff=y_train-y_org
    for j in range(0,10000):
        beta=beta-rate*(x.T.dot(diff))
        diff=x.dot(beta)-y_org
        count=count+1
        if(count%50==0):
            print(error(y_org,x.dot(beta)))
    print("\n")
    y_train=x.dot(beta)
    train_error.append(error(y_train,y_org))
    x_test=testing_data[:,0:3]
    ones=np.ones((x_test.shape[0],1))
    x_test=np.append(ones,x_test,axis=1)
    y_test_org=testing_data[:,3]
    y_test=x_test.dot(beta)
    test_error.append(error(y_test,y_test_org))

train_error_mean=np.mean(train_error)
train_error_var=np.var(train_error)
train_error_min=min(train_error)
test_error_mean=np.mean(test_error)
test_error_var=np.var(test_error)
test_error_min=min(test_error)
    
    

    
    



