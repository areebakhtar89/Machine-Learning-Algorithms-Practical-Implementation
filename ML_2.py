import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

data_set = pd.read_csv('data.csv')
print(data_set)
print('-'*80)
print(data_set.isnull())
print('-'*80)

X = data_set.iloc[:,:-1].values
Y = data_set.iloc[:,-1].values

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
X[:,1:] = imputer.fit_transform(X[:,1:])

ct = ColumnTransformer([("Country", OneHotEncoder(), [0])], remainder='passthrough')
X = ct.fit_transform(X)
print('-----------X_transform---------------')
print(X)
label_encoder_Y = LabelEncoder()
Y = label_encoder_Y.fit_transform(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size=0.25, random_state = 0)

X_sc = StandardScaler()
X_test = X_sc.fit_transform(X_test)
X_train = X_sc.fit_transform(X_train)
print('-------------------X_train-----------------')
print(X_train)
print('-------------------X_test------------------')
print(X_test)
print('-------------------Y_train------------------')
print(Y_train)
print('-------------------Y_test-------------------')
print(Y_test)
print('--'*40)
