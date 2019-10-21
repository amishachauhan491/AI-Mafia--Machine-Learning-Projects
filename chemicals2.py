import numpy as np 
import pandas as pd 

read_x_train = pd.read_csv('Logistic_X_Train.csv')
read_y_train = pd.read_csv('Logistic_Y_Train.csv')
read_x_test = pd.read_csv('Logistic_X_Test.csv')


#joining

data =  read_x_train.join(read_y_train)




x = data.iloc[0:,0:3].values
y = data.iloc[0:,3].values



# Feature scaling...It basically helps to normalise the data within a particular range 
#it also helps in speeding up the calculations in an algorithm.
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

x_train = sc.fit_transform(x)
read_x_test = sc.fit_transform(read_x_test)





from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(x_train, y)
y_pred = logreg.predict(read_x_test)
print(y_pred)
