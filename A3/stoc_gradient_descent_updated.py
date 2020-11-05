
import pandas as pd
import numpy as np
import pre_processing
import matplotlib
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import PolynomialFeatures

def error(y,yi):
    return (np.mean((y-yi)**2,dtype=np.float64))**(1/2)

def gd(x, y_org, rate, num_iter,lam,Islasso,Isdegree):
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
        i= random.randint(0,x.shape[0]-1)
        xi = x[i].reshape((1,x.shape[1]))
        yi = y_org[i].reshape((1,y_org.shape[1]))
        diff=xi.dot(beta)-yi
        if(Islasso==1 and Isdegree==1):
            #beta=beta - rate*((x.T).dot(diff))
            #beta=beta-rate*(x.T.dot(diff)) + lam*np.ones((x.shape[1],1))
            beta=beta-rate*(xi.T.dot(diff)) -(rate*(lam*(xi.shape[0]))*np.sign(beta))
        elif(Islasso==0 and Isdegree==1):
            #beta=beta-rate*(x.T.dot(diff))
            beta=beta-rate*(xi.T.dot(diff)-rate*2*lam*beta*x.shape[0])
        else:
            beta=beta - rate*((xi.T).dot(diff))
        diff=x.dot(beta)-y_org
        count=count+1
        costs.append(error(y_org,x.dot(beta)))
        #print(beta)
        if(count%50==0):
            print("{:.5f}".format(error(y_org,x.dot(beta))))
            
            
    #plt.plot(np.arange(num_iter), costs, label = 'λ = ' + "{:.2f}".format(lam))
    return beta

def process(data, learning_rate, num_iter,degree,Islasso,Isdegree):
    # Training data part
    (training_data,validation_data,testing_data,x_shuffled)=pre_processing.train_test_split(data,degree)
    x=training_data[:,:-1]
    y_org=training_data[:,-1]
    y_org=np.reshape(y_org,(x.shape[0],1))
    lambdas=[]
    betas=[]
    error_in_train=0
    temp=[]
    error_in_validation=0
    error_in_test=0
    for j in range(0,10):
        lam=random.random()
        lambdas.append(lam)
        beta = gd(x, y_org, learning_rate, num_iter,lam,Islasso,Isdegree)
        betas.append(beta)
        
    x_validate=validation_data[:,:-1]
    y_validate_org=validation_data[:,-1]
    for beta in betas:
        y_validate=x_validate.dot(beta)
        temp.append(error(y_validate_org,y_validate))
    min_index=temp.index(min(temp))
    min_lambda=lambdas[min_index]
    min_beta=betas[min_index]
    
    error_in_train=error(x.dot(min_beta),y_org)
    error_in_validation=error(x_validate.dot(min_beta),y_validate_org)
    # Test data part
    x_test=testing_data[:,:-1]
    y_test=testing_data[:,-1]
    error_in_test=error(x_test.dot(min_beta),y_test)
    return min_lambda,min_beta,error_in_train,error_in_validation,error_in_test

def main():
    data=pd.read_csv("insurance.txt").to_numpy()
    data=np.delete(data,2,1)
    learning_rate = float(input("Enter Learning rate: "))
    num_iter = 20000

    all_train_error=[] # A list of rmse for each model for training data
    all_test_error=[] # A list of rmse for each model for test data
    all_validation_error=[]
    all_errors=[]
    Islasso=1
    for i in range(0,10):
        print("Model " + str(i+1))
        lambdas,beta,error_in_train,error_in_validation,error_in_test= process(data, learning_rate, num_iter,i+1,Islasso,1)
        all_errors.append([i+1,lambdas,error_in_train,error_in_validation,error_in_test])
    all_errors=np.array(all_errors)
    df1=pd.DataFrame({'Degree':all_errors[:,0],'lambda':all_errors[:,1],'train-error':all_errors[:,2],'validation-error':all_errors[:,3],'test-error':all_errors[:,4]})
    
    
    all_errors=[]
    Islasso=0
    for i in range(0,10):
        print("Model " + str(i+1))
        lambdas,beta,error_in_train,error_in_validation,error_in_test= process(data, learning_rate, num_iter,i+1,Islasso,1)
        all_errors.append([i+1,lambdas,error_in_train,error_in_validation,error_in_test])
    all_errors=np.array(all_errors)
    df2=pd.DataFrame({'Degree':all_errors[:,0],'lambda':all_errors[:,1],'train-error':all_errors[:,2],'validation-error':all_errors[:,3],'test-error':all_errors[:,4]})
    
    
    all_errors=[]
    for i in range(0,10):
        print("Model " + str(i+1))
        lambdas,beta,error_in_train,error_in_validation,error_in_test= process(data, learning_rate, num_iter,i+1,Islasso,0)
        all_errors.append([i+1,0,error_in_train,error_in_validation,error_in_test])
    all_errors=np.array(all_errors)
    df3=pd.DataFrame({'Degree':all_errors[:,0],'lambda':all_errors[:,1],'train-error':all_errors[:,2],'validation-error':all_errors[:,3],'test-error':all_errors[:,4]})
    
    print("Lasso Regression With Regularisation")
    print(df1)
    print("Ridge Regression With Regularisation")
    print(df2)
    print("No Regularisation")
    print(df3)
    df1.to_csv('df1.csv')
    df2.to_csv('df2.csv')
    df3.to_csv('df3.csv')
    
    
    
    
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
    
    

    
    



