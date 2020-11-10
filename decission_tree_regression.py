import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from sklearn.impute import SimpleImputer
#from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#from sklearn.preprocessing import StandardScaler
#from sklearn.model_selection import train_test_split 
#from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

dataset = pd.read_csv("Data_Sets/Position_Salaries.csv")
print(dataset)

X = dataset.iloc[:,1:2].values
Y = dataset.iloc[:,2].values

regressor = DecisionTreeRegressor()
regressor.fit(X, Y)

X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X,Y,color='r')
plt.plot(X_grid, regressor.predict(X_grid),color='b')