#import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#from sklearn.impute import SimpleImputer
#from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score

dataset = pd.read_csv("Data_Sets/GOOG.csv")
print(dataset.columns)

X = dataset.iloc[:,1:4].values
Y = dataset.iloc[:,4].values


X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2, random_state=0)

regressor = LinearRegression()
regressor.fit(X_train,Y_train)
predicted_linear_regression = regressor.predict(X_test)

regressor_rfr = RandomForestRegressor(n_estimators=500, random_state=0)
regressor_rfr.fit(X_train,Y_train)
predicted_random_forest_regressor = regressor_rfr.predict(X_test)

regressor_dt = DecisionTreeRegressor()
regressor_dt.fit(X_train, Y_train)
predicted_decision_tree_regressor = regressor_dt.predict(X_test)
print('Hello')

print("Linear Regression Accuracy", accuracy_score(Y_test,predicted_linear_regression))
print("Decision Tree Regression Accuracy", accuracy_score(Y_test,predicted_decision_tree_regressor))
print("Random Forest Regression Accuracy", accuracy_score(Y_test,predicted_random_forest_regressor))