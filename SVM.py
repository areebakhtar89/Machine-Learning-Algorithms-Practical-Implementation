import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from sklearn.impute import SimpleImputer
#from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
#from sklearn.model_selection import train_test_split 
#from sklearn.linear_model import LinearRegression

dataset = pd.read_csv("Position_Salaries.csv")
print(dataset)

X = dataset.iloc[:,1:2].values
Y = dataset.iloc[:,2].values

#sc_X = StandardScaler()
#x = sc_X.fit_transform(X)
#sc_Y = StandardScaler()
#y = sc_Y.fit_transform(Y)

from sklearn.svm import SVR

regressor = SVR(kernel='poly')
regressor.fit(X, Y)

regressor.predict([[10]])
