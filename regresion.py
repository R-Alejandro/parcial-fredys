import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import pandas as pd
############################################################################
datos = pd.read_csv("./parcial/CryptoCoins.csv")
# filas, columnas
x = datos.iloc[60:90,2].values #Open
y = datos.iloc[60:90,5].values #Close
############################################################################

#loistic regression (regresion lineal)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3)
r = linear_model.LinearRegression()
x_train= np.array(x_train).reshape(-1,1)
y_train=np.array(y_train).reshape(-1,1)
r.fit(x_train, y_train)
y_predict = r.predict(np.array(x_test).reshape(-1,1))
plt.figure(figsize=(10,7))
plt.scatter(x_test, y_test)
plt.title("Open vs Close")
plt.xlabel("Open")
plt.ylabel("Close")
plt.plot(x_test, y_predict, color='red', linewidth=5)
plt.show()

print(f"valor de la pendiente a: {r.coef_}")
print(f"valor del coeficiente b: {r.intercept_}")

#medir la precision del modelo
print(f"precision del modelo (porcentaje): {r.score(x_train, y_train)}")

############################################################################
#regresion polinomial
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2)
x_train_poly = poly.fit_transform(x_train)
x_test_poly = poly.fit_transform(np.array(x_test).reshape(-1,1))

pr = linear_model.LinearRegression()
pr.fit(x_train_poly, y_train) #entrena el modelo
Y_pred = pr.predict(x_test_poly)

plt.scatter(x_test, y_test)
plt.title("Regresion polinomial")
plt.xlabel("Open")
plt.ylabel("Close")
plt.plot(x_test, Y_pred, color='red')
plt.show()

print(f"Precision del modelo (Porcentaje){pr.score(x_train_poly, y_train)}") 



