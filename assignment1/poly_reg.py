from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import numpy as np
import matplotlib.pyplot as plt 

import pandas as pd  
import seaborn as sns

data = pd.read_csv("housing.csv")

names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
data = pd.read_csv("housing.csv", header=None, names=names, delimiter='\s+')
X = pd.DataFrame(np.c_[data['LSTAT']], columns = ['LSTAT'])
Y = data['MEDV']

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=5)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)


def polynomial_regression_model(degree):
 
  
  poly_features = PolynomialFeatures(degree=degree)
  
 
  X_train_poly = poly_features.fit_transform(X_train)
  
 
  poly_model = LinearRegression()
  poly_model.fit(X_train_poly, Y_train)
  
 
  y_train_predicted = poly_model.predict(X_train_poly)
  
 
  y_test_predict = poly_model.predict(poly_features.fit_transform(X_test))
  
 
  rmse_train = np.sqrt(mean_squared_error(Y_train, y_train_predicted))
  r2_train = r2_score(Y_train, y_train_predicted)
  
 
  rmse_test = np.sqrt(mean_squared_error(Y_test, y_test_predict))
  r2_test = r2_score(Y_test, y_test_predict)
  
  print("The model performance for the training set")
  print("RMSE of training set is {}".format(rmse_train))
  print("R2 score of training set is {}".format(r2_train))
  
  print("\n")
  
  print("The model performance for the test set")
  print("RMSE of test set is {}".format(rmse_test))
  print("R2 score of test set is {}".format(r2_test))
    
  plt.scatter(X_test, Y_test, color = 'blue')
  plt.plot(X_test, y_test_predict, color = 'red')
  plt.title('House Price vs LSTAT')
  plt.xlabel('LSTAT')
  plt.ylabel('House Price')
  plt.show()
  
polynomial_regression_model(2)
polynomial_regression_model(20)