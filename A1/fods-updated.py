import random
import numpy as np
from scipy.special import gamma
import matplotlib.pyplot

def coin_toss():
    num=random.random()
    if(num<=0.7):
        return 0
    else:
        return 1

def beta_distribution(x,a,b):
    '''
    Finds the PDF the beta distribution based on inputs a and b on data x 
    '''
    y=np.zeros(len(x))
    x2=1-x
    z=gamma(a+b)/(gamma(a)*gamma(b))
    first=np.power(x,a-1)
    second=np.power(x2,b-1)
    y=np.multiply(first,second)
    y=y*z
    matplotlib.pyplot.plot(x,y)
    matplotlib.pyplot.show()
    # matplotlib.pyplot.savefig('plots.png')
    matplotlib.pyplot.close()

def bob(x, dataset, a, b):
    '''
    Taking values one at a time and finding posterior probabilities and Maximum Likelihood Estimator
    '''
    for i in range(0,160):
        if(dataset[i]==0):
            a=a+1
        else:
            b=b+1
        beta_distribution(x,a,b)
        mew=a/(a+b)

    zeros=np.where(dataset==0)[0]
    zeros=len(zeros)

    mewtwo=(a+zeros)/(a+b+160)
    print(mewtwo)

def lisa(x, dataset, a, b):
    '''
    Likelihood calculated over entire dataset
    '''
    zeros=np.where(dataset==0)[0]
    zeros=len(zeros)
    ones = 160 - zeros
    beta_distribution(x, a + zeros, b + ones)

def main():
    x=np.linspace(0,1,100)
    dataset=np.zeros(160)
    for i in range(0,160):
        dataset[i]=coin_toss()

    a=2
    b=3
    mew=0.4
    beta_distribution(x, a, b)

    # bob(x, dataset, a, b)
    lisa(x, dataset, a, b)

if __name__ == "__main__":
    main()