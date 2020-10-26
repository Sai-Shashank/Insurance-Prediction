
import pandas as pd
import numpy as np
import pre_processing
from numpy.linalg import inv
from matplotlib import pyplot

def error(y,yi):
    return (np.mean((y-yi)**2))**(1/2)
          
data=pd.read_csv("insurance.txt").to_numpy()
data=pre_processing.standardization(data)
all_errors_train=[]
all_errors_test=[]
for i in range(0,20):
    (training_data,testing_data)=pre_processing.train_test_split(data)
    x=training_data[:,0:3]
    y_org=training_data[:,3]
    y_org=np.reshape(y_org,(len(y_org),1))
    ones=np.ones((x.shape[0],1))
    x=np.append(ones,x,axis=1)
    beta=inv(x.T.dot(x)).dot(x.T).dot(y_org)
    y_train=x.dot(beta)
    all_errors_train.append(error(y_org,y_train))
    y_org_test=testing_data[:,3]
    x_test=testing_data[:,0:3]
    ones=np.ones((x_test.shape[0],1))
    x_test=np.append(ones,x_test,axis=1)
    y_test=x_test.dot(beta)
    all_errors_test.append(error(y_org_test,y_test))

train_error_mean=np.mean(all_errors_train)
train_error_var=np.var(all_errors_train)
train_error_min=min(all_errors_train)
test_error_mean=np.mean(all_errors_test)
test_error_var=np.var(all_errors_test)
test_error_min=min(all_errors_test)
    


