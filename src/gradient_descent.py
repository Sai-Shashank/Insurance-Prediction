
import pandas as pd
import numpy as np
import pre_processing
from numpy.linalg import inv
import matplotlib
import matplotlib.pyplot as plt

def error(y,yi):
    return (np.mean((y-yi)**2))**(1/2)

def gd(x, y_org, rate, num_iter, isPlot):
    '''
    Function to apply Gradient descent on training data, and find beta 
    '''
    origrate = rate
    errorToPlot = []
    beta=np.ones((4,1))
    rate=rate/(x.shape[0])
    y_train=x.dot(beta)
    diff=y_train-y_org
    count=0
    for j in range(0,num_iter):
        beta=beta-rate*(x.T.dot(diff))
        diff=x.dot(beta)-y_org
        count=count+1
        if isPlot == 1:
            errorToPlot.append(error(y_org,x.dot(beta)))
        if(count%50==0):
            print(error(y_org,x.dot(beta)))
    if isPlot == 1:
        plt.plot(np.arange(num_iter), errorToPlot, label = "Learning rate = " + str(origrate))
        
    print("\n")
    
    return beta

def process(data, learning_rate, num_iter, isPlot):
    # Training data part
    (training_data,testing_data)=pre_processing.train_test_split(data)
    x=training_data[:,0:3]
    y_org=training_data[:,3]
    ones=np.ones((x.shape[0],1))
    x=np.append(ones,x,axis=1)
    y_org=np.reshape(y_org,(x.shape[0],1))

    beta = gd(x, y_org, learning_rate, num_iter, isPlot)
    
    y_train = x.dot(beta)
    error_in_train = error(y_train,y_org)
    
    # Test data part
    x_test=testing_data[:,0:3]
    ones=np.ones((x_test.shape[0],1))
    x_test=np.append(ones,x_test,axis=1)
    y_test_org=testing_data[:,3]

    y_test=x_test.dot(beta)
    error_in_test = error(y_test,y_test_org)
  
    return error_in_train, error_in_test   

def main():
    data=pd.read_csv("insurance.txt").to_numpy()
    _ = pre_processing.standardization(data)
    learning_rate = float(input("Enter Learning rate: "))
    num_iter = 10000

    train_error=[] # A list of rmse for each model for training data
    test_error=[] # A list of rmse for each model for test data

    isPlot = 0
    for i in range(0,20):
        print("Model " + str(i+1))
        (error_in_train, error_in_test) = process(data, learning_rate, num_iter, isPlot)
        train_error.append(error_in_train)
        test_error.append(error_in_test)
    
    # Plot of RMSE vs iteration
    plt.figure()
    
    #plt.subplot(2,2,1)
    dummy = process(data, 0.1, num_iter,1)
    #plt.title('Learning Rate = 0.1')

    #plt.subplot(2,2,2)
    dummy = process(data, 0.01, num_iter,1)
    #plt.title('Learning Rate = 0.01')

    #plt.subplot(2,2,3)
    dummy = process(data, 0.001, num_iter,1)
    #plt.title('Learning Rate = 0.001')

    #plt.subplot(2,2,4)
    dummy = process(data, 0.0001, num_iter,1)
    #plt.title('Learning Rate = 0.0001')
    plt.xlabel('Epochs')
    plt.ylabel('RMSE')
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.close()
    train_error_mean=np.mean(train_error)
    train_error_var=np.var(train_error)
    train_error_min=min(train_error)
    test_error_mean=np.mean(test_error)
    test_error_var=np.var(test_error)
    test_error_min=min(test_error)


if __name__ == "__main__":
    main()
    
    

    
    



