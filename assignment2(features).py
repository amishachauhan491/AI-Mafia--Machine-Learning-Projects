import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
#read data

df=pd.read_csv('Train.csv')
x_train = df.iloc[:,0:5].values
#x_train=x_train.reshape((8000,))



y_train = df.iloc[:,5].values
#y_train= y_train.reshape((1600,))
#print(x)
#plot the data set
#plt.scatter(x_train,y_train,color='r')
#plt.show()
#normalisation
#x_train=x_train-x_train.mean()/x_train.std()
#plt.scatter(x_train,y_train)
#plt.show()


from sklearn.linear_model import LinearRegression

# Create linear regression object
regr = LinearRegression()


# Train the model using the training sets 
regr.fit(x_train, y_train)

dff=pd.read_csv('Test.csv')
X_test = df.iloc[:,0:5].values
# Make predictions using the testing set
answer = regr.predict(X_test)
plt.scatter(x_train[:,0],y_train,color='red')
print(answer)
plt.show()




