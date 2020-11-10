import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split 
from sklearn.compose import ColumnTransformer

dataset = pd.read_csv("Data_Sets/50_Startups.csv")
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4].values


ct = ColumnTransformer([("State", OneHotEncoder(), [3])], remainder='passthrough')
X = ct.fit_transform(X)
print(X)
# Avoiding the Dummy Variable Trap
#X = X[:, 1:]

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

y_pred = regressor.predict(X_test)


import math
from sklearn.metrics import mean_squared_error

rmse = math.sqrt(mean_squared_error(y_test,y_pred))
print(rmse)


from statistics import mean 
print(mean(y_test))

rmse/mean(y_test)