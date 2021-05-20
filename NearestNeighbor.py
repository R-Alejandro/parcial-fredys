# Se importan las librerias correspondientes
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Importamos la libreria
dataset = pd.read_csv("CryptoCoins.csv")

# Se definen x_train, y_train, x_test y y_test
x_train = dataset.iloc[:100, 2:3].values #:30
y_train = dataset.iloc[:100, 5].values #:30
x_test = dataset.iloc[110:210, 2:3].values #70:100
y_test = dataset.iloc[110:210, 5].values# 70:100

# Se cambia el tipo de la variable
x_train = x_train.astype(int)
y_train = y_train.astype(int)
x_test = x_test.astype(int)
y_test = y_test.astype(int)

# Neighbor
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(x_train, y_train)
y_predic = neigh.predict(x_test)

# Tabla
plt.scatter(x_train, y_train, color='red')
plt.plot(x_train, neigh.predict(x_train), color='blue')
plt.title("Nearest Neighbor")
plt.xlabel("Valor inicial")
plt.ylabel("Valor final")
plt.show()

# Se calacula la presicion
print(f"precision de Nearest Neighbor:\n {round(neigh.score(np.array(y_test).reshape(-1,1), np.array(y_predic).reshape(-1,1)) * 100, 2)}")