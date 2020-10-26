
import pandas as pd
import numpy as np
import pre_processing
from numpy.linalg import inv
from matplotlib import pyplot

def error(y_given,y_pred,m):
    error=((y_given-y_pred)*(y_given-y_pred))/(2*m)
    error=np.sum(error)
    return error

def pred_y(beta,x):
    return (x.dot(beta))
data=pd.read_csv("insurance.txt").to_numpy()
(training_data,testing_data)=pre_processing.train_test_split(data)
training_data=pre_processing.normalization(training_data)
x=training_data[:,0:3]
y=training_data[:,3]
ones=np.ones((936,1))
x=np.append(ones,x,axis=1)
count=0
beta=np.ones((4,1))
rate=0.1/936
y=np.reshape(y,(936,1))
diff=x.dot(beta)-y
for i in range(0,10000):
    beta=beta-rate*(x.T.dot(diff))
    diff=x.dot(beta)-y
    count=count+1
    if(count%50==0):
        print(error(y,x.dot(beta),936))

y_pred=x.dot(beta)
    

    
    



