
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd

def normalization(data):
        # data is a numpy multidim array
        minval=np.amin(data,axis=0)     # axis=0 returns an array containing the smallest element for each column
        maxval=np.amax(data,axis=0)
        # Iterate over each column (feature) and compute the normalized value of each value in feature column
        for i in range(0,len(minval)):
            data[:,i]=(data[:,i]-minval[i])/(maxval[i]-minval[i])
        return data


def standardization(data):
        # data is a numpy multidim array
        meanval=np.mean(data,axis=0)    # axis=0 returns an array containing the smallest element for each column
        std_dev=np.std(data,axis=0)
        # Iterate over each column (feature) and compute the normalized value of each value in feature column
        for i in range(0,len(meanval)):
            data[:,i]=(data[:,i]-meanval[i])/(std_dev[i])  
        return data


def train_test_split(data,degree):
        np.random.shuffle(data)    # Shuffle datapoints randomly
        poly = PolynomialFeatures(degree)
        x=data[:,:-1] # Slice all rows and all columns except the last one
        x_shuffled = x[0:936,:]
        y=data[:,-1] # Slice all rows and the last column
        y=np.reshape(y,(len(y),1))
        x=poly.fit_transform(x) 
        data=np.append(x,y,axis=1)
        x_temp=data[:,1:] # Slice all rows and all columns except the first one (all values as 1; normalization not needed)
        #x_temp=standardization(x_temp)
        x_temp=normalization(x_temp)
        x=data[:,0]
        x=x.reshape((len(x),1))
        data=np.append(x,x_temp,axis=1) # Normalized data
        # Split training, validation and testing data in 70:20:10 ratio
        training_set=data[0:936,:]      
        validation_set=data[936:1204,:]
        testing_set=data[1204:,:]
        return (training_set,validation_set,testing_set,x_shuffled)
"""
data=pd.read_csv("insurance.txt").to_numpy()
data=np.delete(data,2,1)
training_set,validation_set,testing_set=train_test_split(data,8)
"""





