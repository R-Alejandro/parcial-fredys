import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import svm
from sklearn import datasets
############################################################################
datos = pd.read_csv("./parcial/CryptoCoins.csv")
# filas, columnas
x = datos.iloc[:100, 2:3].values
y = datos.iloc[:100, 5].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

x_train = x_train.astype(int)
y_train = y_train.astype(int)
x_test = x_test.astype(int)
y_test = y_test.astype(int)

clasificador_lineal = svm.SVC(kernel='linear')
clasificador_lineal.fit(x_train, y_train)

#x_test = np.array([numero a evaluar])
y_predic = clasificador_lineal.predict(x_test)
print(y_predic)

plt.scatter(x_train, y_train, color='red')
plt.plot(x_train, clasificador_lineal.predict(x_train), color='blue')
plt.title("Arbol de decision regresion")
plt.xlabel("Open")
plt.ylabel("Close")
plt.show()
#evaluamos la precision
print(f"precision de arbol de decision:\n {round(clasificador_lineal.score(np.array(y_test).reshape(-1,1), np.array(y_predic).reshape(-1,1)) * 100, 2)}")