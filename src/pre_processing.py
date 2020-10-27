
import numpy as np

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


def train_test_split(data):
        np.random.shuffle(data)
        training_set=data[0:936,:]
        testing_set=data[936:,:]
        return (training_set,testing_set)

