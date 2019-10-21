import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#read data
def readData(filename):
    df=pd.read_csv(filename)
    return df.values
#reading x
x=readData('Linear_X_Train.csv')
#reading y
y=readData('Linear_Y_Train.csv')
x=x.reshape((3750,))
y=y.reshape((3750,))
#print(x)
#plot the data set
plt.scatter(x,y,color='r')
plt.show()
#normalisation
x=x-x.mean()/x.std()
plt.scatter(x,y)
plt.show()
#define hypothesis
def hypothesis(theta,x):
    return theta[0]+theta[1]*x
#error/cost function
def error(x,y,theta):
    
    m=x.shape[0]
    
    total_error=0
    for i in range(m):
        total_error+=(y[i]-hypothesis(theta,x[i]))**2
         
    return 0.5*total_error
#function for update
#calculating derivatives
def gradient(x,y,theta):
    
    grad=np.array([0.0,0.0])
    m=x.shape[0]
    for i in range(m):
        grad[0]+=(hypothesis(theta,x[i])-y[i]) 
        grad[1]+=(hypothesis(theta,x[i])-y[i])*x[i]
        
    return grad
#applying gradient descent
def gradient_descend(x,y,learning_rate,maxIter):
    
    theta=np.array([0.0,0.0])
    err=[]
    for i in range(maxIter):
        grad=gradient(x,y,theta)
        ce=error(x,y,theta)
        theta[0]=theta[0]-learning_rate*grad[0]
        theta[1]=theta[1]-learning_rate*grad[1]
        err.append(ce)
        
    return theta,err
theta,err=gradient_descend(x,y,learning_rate=0.001,maxIter=1000)
print(theta[0],theta[1])
plt.scatter(x,y,color='g')
plt.plot(x,hypothesis(theta,x),color='r')
plt.show()
plt.plot(err)
plt.show()
















        
        
        
        






        
     


























































print(x)
