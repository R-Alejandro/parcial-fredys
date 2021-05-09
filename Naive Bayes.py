import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.datasets import make_blobs
from sklearn.naive_bayes import GaussianNB
import pandas as pd

datos = pd.read_csv("CryptoCoins.csv")
# filas, columnas
X = datos.iloc[:20,[9,10]].values #Open
y = GaussianNB()
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='RdBu')
plt.show()

# model = GaussianNB()
# model.fit(X, y)
# rng = np.random.RandomState(0)
# Xnew = [-6, -14] + [14, 18] * rng.rand(2000, 2)
# ynew = model.predict(Xnew)
# plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='RdBu')
# lim = plt.axis()
# plt.scatter(Xnew[:, 0], Xnew[:, 1], c=ynew, s=20, cmap='RdBu', alpha=0.1)
# plt.axis(lim);
# yprob = model.predict_proba(Xnew)
# yprob[-8:].round(2)
# print(yprob)