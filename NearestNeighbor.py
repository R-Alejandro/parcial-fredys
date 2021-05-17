from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KDTree
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


dataset = pd.read_csv("CryptoCoins.csv")

x_train = dataset.iloc[:100, 2:3].values #:30
y_train = dataset.iloc[:100, 5].values #:30
x_test = dataset.iloc[110:210, 2:3].values #70:100
y_test = dataset.iloc[110:210, 5].values# 70:100

x_train = x_train.astype(int)
y_train = y_train.astype(int)
x_test = x_test.astype(int)
y_test = y_test.astype(int)

# Test2
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(x_train, y_train)
y_predic = neigh.predict(x_test)

plt.scatter(x_train, y_train, color='red')
plt.plot(x_train, neigh.predict(x_train), color='blue')
plt.title("Nearest Neighbor")
plt.xlabel("Open")
plt.ylabel("Close")
plt.show()

print(f"precision de arbol de decision:\n {round(neigh.score(np.array(y_test).reshape(-1,1), np.array(y_predic).reshape(-1,1)) * 100, 2)}")