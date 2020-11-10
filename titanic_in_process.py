import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

df = pd.read_csv('Data_Sets/Titanic_full.csv')
print(df.head(40))
print(df.columns)
#for i in df.columns:
#    print(type(i))
X = df.drop(['Survived', 'Name', 'Name_wiki', 'Ticket'], axis=1)
Y = df['Survived']
ct = ColumnTransformer([(['Sex','Cabin',])])

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

X = imputer.fit_transform(X)
Y = imputer.fit_transform(Y)
