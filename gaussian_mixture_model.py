import seaborn as sns
#from sklearn.model_selection import train_test_split
#from sklearn.linear_model import LinearRegression
#from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
#import pandas as pd
#from sklearn.metrics import accuracy_score
from sklearn.mixture import GaussianMixture

df = sns.load_dataset('iris')
print(df)
sns.set()
sns.pairplot(df, hue='species', size=1.5)

X = df.iloc[:,:-1]
Y = df.iloc[:,-1]

pca = PCA(n_components=2)
X_2D = pca.fit_transform(X)

df['PCA1'] = X_2D[:,0]
df['PCA2'] = X_2D[:,1]

model = GaussianMixture(n_components=3, covariance_type='full')
model.fit(X)
y_gmm = model.predict(X)
df['cluster'] = y_gmm

sns.lmplot('PCA1', 'PCA2', hue='species', data=df, col='cluster')
sns.lmplot('PCA1', 'PCA2', hue='species', data=df, col='cluster', fit_reg=False)