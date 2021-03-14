import numpy as np
import matplotlib.pyplot as plt 

import pandas as pd  
import seaborn as sns

data = pd.read_csv("housing.csv")

names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
data = pd.read_csv("housing.csv", header=None, names=names, delimiter='\s+')

X = pd.DataFrame(np.c_[data['LSTAT'], data['RM'], data['PTRATIO']], columns = ['LSTAT', 'RM','PTRATIO'])
Y = data['MEDV']

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=5)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

lin_model = LinearRegression()
lin_model.fit(X_train, Y_train)

from sklearn.metrics import r2_score
y_train_predict = lin_model.predict(X_train)
y_test_predict = lin_model.predict(X_test)

rmse = (np.sqrt(mean_squared_error(Y_test, y_test_predict)))
r2 = r2_score(Y_test, y_test_predict)

print("The model performance for testing set")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))

print('Adjusted R Squared Value is ',1 - (1-r2_score(Y_test, y_test_predict))*(len(Y_test)-1)/(len(Y_test)-X_train.shape[1]-1))