#importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importing the mail dataset with pandas
df=pd.read_csv('Mall_Customers.csv')
X=df.iloc[:,[3,4]].values

#using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',n_init=10,random_state=0)
    kmeans.fit(X);
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title("elbow method")
plt.xlabel('no.of clusters')
plt.ylabel('wcss values')
plt.show()

#applying kmeans to the mail dataset
kmeans=KMeans(n_clusters=5,init='k-means++',n_init=10,random_state=0)
y_kmeans=kmeans.fit_predict(X)

#visualising the clusters
plt.scatter(X[y_kmeans==0,0],X[y_kmeans==0,1],c='red',label='Cluster 1')
plt.scatter(X[y_kmeans==1,0],X[y_kmeans==1,1],c='blue',label='Cluster 2')
plt.scatter(X[y_kmeans==2,0],X[y_kmeans==2,1],c='green',label='Cluster 3')
plt.scatter(X[y_kmeans==3,0],X[y_kmeans==3,1],c='cyan',label='Cluster 4')
plt.scatter(X[y_kmeans==4,0],X[y_kmeans==4,1],c='red',label='Cluster 5')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],c='yellow',label='Centriod')
plt.xlabel("Annual income")
plt.ylabel("Spending Scores")
plt.legend()
plt.show()