import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

df = pd.read_csv('Data_Sets/Mall_Customers.csv')

X = df.iloc[:,[3,4]].values

wcss=[]
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, random_state=20)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1,11),wcss)
plt.title('The elbow method')
plt.xlabel('No of clusters')
plt.ylabel('WCSS')
plt.show()
                        
kmeans = KMeans(n_clusters=5, random_state=20)
y_means = kmeans.fit_predict(X)

plt.scatter(X[y_means==0,0],X[y_means==0,1],c='red',label='cluster 1')
plt.scatter(X[y_means==1,0],X[y_means==1,1],c='yellow',label='cluster 2')
plt.scatter(X[y_means==2,0],X[y_means==2,1],c='blue',label='cluster 3')
plt.scatter(X[y_means==3,0],X[y_means==3,1],c='orange',label='cluster 4')
plt.scatter(X[y_means==4,0],X[y_means==4,1],c='green',label='cluster 5')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], c='black', label='centroid')
plt.title('Clusters of customers')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()
plt.show()