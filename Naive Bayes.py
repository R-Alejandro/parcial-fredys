import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.datasets import make_blobs
from sklearn.naive_bayes import GaussianNB
import pandas as pd

datos = pd.read_csv("CryptoCoins.csv")
# filas, columnas
X = datos.iloc[:100,[9,10]].values #Open
# y = datos.iloc[:20,[10]]
S, y = make_blobs(100, 2, centers=2, random_state=2, cluster_std=1.5)

plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='RdBu')
plt.show()

model = GaussianNB()
model.fit(X, y)
rng = np.random.RandomState(0)
Xnew = [-6, -14] + [14, 18] * rng.rand(10000, 1)
ynew = model.predict(Xnew)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='RdBu')
lim = plt.axis()
plt.scatter(Xnew[:, 0], Xnew[:, 1], c=ynew, s=20, cmap='RdBu', alpha=0.1)
plt.axis(lim)
yprob = model.predict_proba(Xnew)
yprob[-8:].round(2)
print(yprob)
correct = 0


for i in range(len(yprob)):
    correct += yprob[i][1] * (yprob[i][0] * 100)

print(f"Precision del modelo (Porcentaje) {(correct/float(len(yprob)))*100}") 