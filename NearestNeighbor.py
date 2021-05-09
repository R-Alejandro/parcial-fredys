import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
# Importamos las librerias necesarias para las graficas y el algoritmo

# Impoortamos la data 
dataframe = pd.read_csv("CryptoCoins.csv")

# Determinamos X e Y de entradas, y los set y los test de entrenamiento
X = dataframe.iloc[60:90,2].values #Open
y = dataframe.iloc[60:90,5].values #Close

punto_nuevo = {'Open': [1692.000000], 'Close': [1692.000000]}

ax = plt.axes()

ax.scatter(dataframe.loc[dataframe['Genero'] == 'm', 'Altura'],c="blue",label="Mujer")

ax.scatter(punto_nuevo['Open'],punto_nuevo['Close'],c="black")

plt.xlabel("Open")
plt.ylabel("Close")
ax.legend()
plt.show()

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
X = dataframe[['Masa', 'Altura']]
y = dataframe[['Genero']]
knn.fit(X, y)
prediccion = knn.predict(punto_nuevo)
print(prediccion)