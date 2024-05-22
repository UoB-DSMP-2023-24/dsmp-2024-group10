import pandas as pd
import numpy as np
import umap
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

df_AA=pd.read_csv('df_AG--RotationEncodingBL62.txt_EncodingMatrix.txt', sep="\t", header=None)
df_BB=pd.read_csv('df_BG--RotationEncodingBL62.txt_EncodingMatrix.txt', sep="\t", header=None)
df_AB=pd.read_csv('df_AB--RotationEncodingBL62.txt_EncodingMatrix.txt', sep="\t", header=None)

df_AB = df_AB[df_AB.iloc[:, 6] == 'HomoSapiens']
print(np.shape(df_AB))
data_AB_human = df_AB.iloc[:,14:]
labels, unique_values = pd.factorize(df_AB.iloc[:, 10])
print(labels)
mapping_dict = {i: val for i, val in enumerate(unique_values)}
print(mapping_dict[labels[0]])
data_ab_human = data_AB_human.to_numpy()
print(np.shape(data_ab_human))


reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2)
print("Before UMAP:", np.shape(data_ab_human))  # 确认降维前的数据形状
# 使用UMAP降维
embedding = reducer.fit_transform(data_ab_human)
print("After UMAP:", np.shape(embedding))  # 确认降维后的数据形状
plt.scatter(embedding[:, 0], embedding[:, 1], s=5, c=labels, cmap='Spectral')
plt.title('Human alpha beta')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.show()

kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(embedding)
cluster_labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Plotting the UMAP reduced data with cluster labels and centroids
plt.scatter(embedding[:, 0], embedding[:, 1], s=5, c=cluster_labels, cmap='Spectral', alpha=0.6)
plt.scatter(centroids[:, 0], centroids[:, 1], c='black', s=100, marker='x')  # Red x marks centroids
plt.title('Human alpha beta Clustering with K-means')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.show()

from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt

Z = linkage(embedding, 'ward')

k = 7

clusters = fcluster(Z, k, criterion='maxclust')

plt.figure(figsize=(10, 7))
dendrogram(Z)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample index')
plt.ylabel('Distance')

# 画出对应于指定聚类数的切割线
max_d = max([Z[i][2] for i in range(len(Z) - k, len(Z)-1)])
# plt.axhline(y=max_d, c='k', ls='--', lw=0.8)  # 添加一条水平虚线

plt.show()