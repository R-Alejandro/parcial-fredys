from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

datos = pd.read_csv("CryptoCoins.csv")

X = datos.iloc[:100,[9,10]].values
Xu = datos.iloc[:100,[9]].values
Xd = datos.iloc[:100,[10]].values

nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)
distances, indices = nbrs.kneighbors(X)

kdt = KDTree(X, leaf_size=30, metric='euclidean')
kdtFind = kdt.query(X, k=2, return_distance=False)

print("\nindices \n",indices)
print("\ndistancias \n",distances)
print("\nConecciones entre los puntos vecinos \n",nbrs.kneighbors_graph(X).toarray())
print("\nKDTree and BallTree Classes to find nearest neighbors \n",kdtFind)

# plt.scatter(Xu, Xd, color='red')
# plt.title("Nearest Neighbor")
# plt.xlabel("Open")
# plt.ylabel("Close")
# plt.show()
correct = 0

# print(indices)
# print(kdt.query(X, k=2, return_distance=False))



for i in range(len(X)):
    if int(indices[i,1]) == int(kdtFind[i,1]):
        correct += 1

#precision del modelo
print(f"precision de Nearest Neighbor:\n {correct*0.01}")