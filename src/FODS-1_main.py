# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 18:05:51 2020

@author: Shashank
"""
import pandas as pd
import numpy as np
#import pre_processing
from numpy.linalg import inv
from matplotlib import pyplot

training_data=pd.read_csv("insurance.txt").to_numpy()
#(training_data,testing_data)=pre_processing.train_test_split(data)
#training_data=pre_processing.normalization(training_data)
x=training_data[:,0:3]
y=training_data[:,3]
ones=np.ones((1338,1))
x=np.append(ones,x,axis=1)
beta=inv(x.T.dot(x)).dot(x.T).dot(y)
y_train=x.dot(beta)


