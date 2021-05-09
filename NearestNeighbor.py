import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
import pandas as pd
# Importamos las librerias necesarias para las graficas y el algoritmo

# Impoortamos la data 
dataframe = pd.read_csv("CryptoCoins.csv")

# Determinamos X e Y de entradas, y los set y los test de entrenamiento
X = dataframe.iloc[60:90,2].values #Open
y = dataframe.iloc[60:90,5].values #Close

n_neighbors = 15

# tamaño del paso en la malla
h = .02
 
# Create color maps
cmap_light = ListedColormap(['#FF1AAA', '#AAFAA1', '#AAAAA1'])
cmap_bold = ListedColormap(['#FF0000', '#19FA05', '#AAAA99'])
 
for weights in ['uniform', 'distance']:
  # creamos una instancia de Clasificador de Vecinos y encajamos los datos.
  clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
  clf.fit(X, y)
 
  #Plotea el límite de decisión. Para ello, asignaremos un color a cada uno de ellos.
  # puntos de la malla [x_min, x_max]x[y_min, y_max].
  x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
  y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
  xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
  np.arange(y_min, y_max, h))
  Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
 
  # Pon el resultado con colores
  Z = Z.reshape(xx.shape)
  plt.figure()
  plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
 
  # plotea tambien los puntos de entrenamiento
  plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,edgecolor='k', s=20)
  plt.xlim(xx.min(), xx.max())
  plt.ylim(yy.min(), yy.max())
  plt.title("3-Class classification (k = %i, weights = '%s')"
            % (n_neighbors, weights))
 
plt.show()