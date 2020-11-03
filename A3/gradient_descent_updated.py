
import pandas as pd
import numpy as np
import pre_processing
import matplotlib
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import PolynomialFeatures

def error(y,yi):
    return (np.mean((y-yi)**2,dtype=np.float64))**(1/2)

def gd(x, y_org, rate, num_iter,lam,Islasso):
    '''
    Function to apply Gradient descent on training data, and find beta 
    '''
    beta=np.ones((x.shape[1],1))
    rate=rate/(x.shape[0])
    y_train=x.dot(beta)
    diff=y_train-y_org
    count=0
    costs = []
    for j in range(0,num_iter):
        if(Islasso):
            #beta=beta - rate*((x.T).dot(diff))
            #beta=beta-rate*(x.T.dot(diff)) + lam*np.ones((x.shape[1],1))
            beta=beta-rate*(x.T.dot(diff)) + (lam/(x.shape[0]))*np.sign(beta)
        else:
            #beta=beta-rate*(x.T.dot(diff))
            beta=beta-rate*(x.T.dot(diff)+2*lam*beta)
        diff=x.dot(beta)-y_org
        count=count+1
        costs.append(error(y_org,x.dot(beta)))
        #print(beta)
        if(count%50==0):
            print(error(y_org,x.dot(beta)))
            
            
    plt.plot(np.arange(num_iter), costs, label = 'λ = ' + "{:.2f}".format(lam))
    return beta

def process(data, learning_rate, num_iter,degree,Islasso):
    # Training data part
    (training_data,validation_data,testing_data)=pre_processing.train_test_split(data,degree)
    x=training_data[:,:-1]
    y_org=training_data[:,-1]
    y_org=np.reshape(y_org,(x.shape[0],1))
    error_in_train=[]
    error_in_test=[]
    error_in_validation=[]
    lambdas=[]

    plt.figure()
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Learning Rate = '+str(learning_rate) )
    for j in np.arange(0.1, 1.0, 0.1):
        '''
        if(j==0):
            lam=0
            lambdas.append(lam)
        else:
            lam=random.random()
            lambdas.append(lam)
        '''
        print("for lam = " + "{:.2f}".format(j))
        beta = gd(x, y_org, learning_rate, num_iter,j,1)
        
        y_train = x.dot(beta)
        error_in_train.append(error(y_train,y_org))
        
        x_validate=validation_data[:,:-1]
        y_validate_org=validation_data[:,-1]
        y_validate=x_validate.dot(beta)
        error_in_validation.append(error(y_validate_org,y_validate))
        # Test data part
        x_test=testing_data[:,:-1]
        y_test_org=testing_data[:,-1]
    
        y_test=x_test.dot(beta)
        error_in_test.append(error(y_test,y_test_org))
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.close()
    return error_in_train,error_in_validation,error_in_test,lambdas 

def main():
    data=pd.read_csv("insurance.txt").to_numpy()
    _ = pre_processing.standardization(data)
    data=np.delete(data,2,1)
    learning_rate = float(input("Enter Learning rate: "))
    num_iter = 1000

    all_train_error=[] # A list of rmse for each model for training data
    all_test_error=[] # A list of rmse for each model for test data
    all_validation_error=[]
    Islasso=1
    for i in range(7,8):
        print("Model " + str(i+1))
        (error_in_train,error_in_validation,error_in_test,lambdas) = process(data, learning_rate, num_iter,i+1,Islasso)
        all_train_error.append(error_in_train)
        all_validation_error.append(error_in_validation)
        all_test_error.append(error_in_test)
        print(all_train_error)
        
    
"""
    train_error_mean=np.mean(train_error)
    train_error_var=np.var(train_error)
    train_error_min=min(train_error)
    test_error_mean=np.mean(test_error)
    test_error_var=np.var(test_error)
    test_error_min=min(test_error)
"""
    
   
    


if __name__ == "__main__":
    main()
    
    

    
    



