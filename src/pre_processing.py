
import numpy as np

def normalization(data):
        processdata=data
        minval=np.amin(data,axis=0)
        maxval=np.amax(data,axis=0)
        for i in range(0,len(minval)):
            processdata[:,i]=(data[:,i]-minval[i])/(maxval[i]-minval[i])
        return processdata


def standardization(data):
        meanval=np.mean(data,axis=0)
        std_dev=np.std(data,axis=0)
        for i in range(0,len(meanval)):
            data[:,i]=(data[:,i]-meanval[i])/(std_dev[i])  
        return data


def train_test_split(data):
        np.random.shuffle(data)
        training_set=data[0:936,:]
        testing_set=data[936:,:]
        return (training_set,testing_set)

