import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("CryptoCoins.csv")
x_train = dataset.iloc[:100, 2:3].values #:30
y_train = dataset.iloc[:100, 5].values #:30
x_test = dataset.iloc[110:210, 2:3].values #70:100
y_test = dataset.iloc[110:210, 5].values# 70:100

x_train = x_train.astype(int)
y_train = y_train.astype(int)
x_test = x_test.astype(int)
y_test = y_test.astype(int)

from sklearn.tree import DecisionTreeClassifier
regresor = DecisionTreeClassifier()
regresor.fit(x_train,y_train)

y_predic = regresor.predict(x_test)
print(y_predic)

plt.scatter(x_train, y_train, color='red')
plt.plot(x_train, regresor.predict(x_train), color='blue')
plt.title("Arbol de decision regresion")
plt.xlabel("Valor inicial")
plt.ylabel("Valor final")
plt.show()

#regularizar los resultados 
x_regulado = np.arange(min(x_train), max(x_train), 0.01)
x_regulado = x_regulado.reshape(len(x_regulado), 1)
plt.scatter(x_train, y_train, color='red')
plt.plot(x_regulado, regresor.predict(x_regulado), color='blue')
plt.title("Arbol de decision regresion")
plt.xlabel("Valor inicial")
plt.ylabel("Valor final")
plt.show()

#precision del modelo
print(f"precision de arbol de decision:\n {round(regresor.score(np.array(y_test).reshape(-1,1), np.array(y_predic).reshape(-1,1)) * 100, 2)}")