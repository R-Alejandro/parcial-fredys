from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KDTree
import numpy as np
import pandas as pd

datos = pd.read_csv("CryptoCoins.csv")

X = datos.iloc[:100,[9,10]].values

nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)
distances, indices = nbrs.kneighbors(X)

kdt = KDTree(X, leaf_size=30, metric='euclidean')


print("indices -> ",indices)
print("distancias -> ",distances)
print("Conecciones entre los puntos vecinos -> ",nbrs.kneighbors_graph(X).toarray())
print("KDTree and BallTree Classes -> ",kdt.query(X, k=2, return_distance=False))