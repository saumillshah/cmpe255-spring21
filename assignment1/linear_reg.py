import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
import pandas as pd  
import seaborn as sns

data = pd.read_csv("housing.csv")
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
data = pd.read_csv("housing.csv", header=None, names=names, delimiter='\s+')

X = pd.DataFrame(np.c_[data['LSTAT']], columns = ['LSTAT'])
Y = data['MEDV']


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=5)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

lin_model = LinearRegression()
lin_model.fit(X_train, Y_train)

from sklearn.metrics import r2_score
y_train_predict = lin_model.predict(X_train)
rmse = (np.sqrt(mean_squared_error(Y_train, y_train_predict)))
r2 = r2_score(Y_train, y_train_predict)

print("The model performance for training set")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print("\n")

y_test_predict = lin_model.predict(X_test)
rmse = (np.sqrt(mean_squared_error(Y_test, y_test_predict)))
r2 = r2_score(Y_test, y_test_predict)

print("The model performance for testing set")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))

import matplotlib.pyplot as plt
plt.scatter(X_test, Y_test, color = 'blue')
plt.plot(X_test, y_test_predict, color = 'red')
plt.title('Linear Reggresion')
plt.xlabel('LSTAT')
plt.ylabel('MEDV')
plt.savefig('Linear_Regression_Best_Fit_Line.png', dpi=400)
plt.show()