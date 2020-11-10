import pandas as pd
#import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch

df = pd.read_csv('Data_Sets/Mall_Customers.csv')

X = df.iloc[:,[3,4]].values

dendogram = sch.dendrogram(sch.linkage(X,method='ward'))
plt.title('Dendogram')
plt.xlabel('Customers')
plt.ylabel('Distance')
plt.show()

hc = AgglomerativeClustering(n_clusters=5)
y_hc = hc.fit_predict(X)

plt.scatter(X[y_hc==0,0],X[y_hc==0,1],c='red',label='cluster 1')
plt.scatter(X[y_hc==1,0],X[y_hc==1,1],c='yellow',label='cluster 2')
plt.scatter(X[y_hc==2,0],X[y_hc==2,1],c='blue',label='cluster 3')
plt.scatter(X[y_hc==3,0],X[y_hc==3,1],c='orange',label='cluster 4')
plt.scatter(X[y_hc==4,0],X[y_hc==4,1],c='green',label='cluster 5')
#plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], c='black', label='centroid')
plt.title('Clusters of customers')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()
plt.show()