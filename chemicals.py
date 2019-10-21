import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
#read data
def readData(filename):
    df=pd.read_csv(filename)
    return df.values
#reading x
x_train=readData('Logistic_X_Train.csv')
#reading y
y_train=readData('Logistic_Y_Train.csv')
x_test=readData('Logistic_X_Test.csv')


#hypothesis
def hypothesis(x,w,b):
    hx=np.dot(x,w) + b
    return sigmoid(hx)
#sigmoid function
def sigmoid(h):
    return 1.0/(1.0 + np.exp(-1.0*h))
#error function
def error(x,y,w,b):
    err = 0.0
    for i in range(x.shape[0]):
        hx = hypothesis(x[i], w, b)
        err += y[i]*np.log2(hx) + (1-y[i])*np.log2(1-hx)

    return err
#gradient
def get_grad(x,y,w,b):
    
    grad_b = 0.0
    grad_w = np.zeros(w.shape)
    for i in range(x.shape[0]):
        hx = hypothesis(x[i],w,b)
        grad_b += (y[i]-hx)
        grad_w += (y[i]-hx)*x[i]
        
    return [grad_w, grad_b]
#gradient ascent
def gradient_ascent(x,y,w,b,learning_rate=0.01):
    
    err=error(x,y,w,b)
    [grad_w, grad_b] = get_grad(x,y,w,b)
    w = w+learning_rate*grad_w
    b = b+learning_rate*grad_b
    return err, w, b

w = 2*np.random.random((x_train.shape[1],))
b = 5*np.random.random()

loss = [ ]
for i in range(1000):
    l , w, b = gradient_ascent(x_train,y_train,w,b,0.001)
    loss.append(l) 
plt.plot(loss)
plt.show()
    
    
    
    