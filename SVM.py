import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import svm
from sklearn import datasets
############################################################################
datos = pd.read_csv("./parcial/CryptoCoins.csv")
# filas, columnas
datos = datasets.load_wine()
x_train, x_test, y_train, y_test = train_test_split(datos.data, datos.target, test_size=0.3)

""" x_train = np.array([x_train])
x_train = x_train.transpose()
print(x_train.shape)
print(y_train.shape) """
clasificador_lineal = svm.SVC(kernel='linear')
clasificador_lineal.fit(x_train, y_train)

y_predic = clasificador_lineal.predict(x_test)
print(y_predic)

#evaluamos la precision
from sklearn import metrics
prec = metrics.precision_score(y_test, y_predic, pos_label='positive',average='micro')
print(f"precision: {prec}")