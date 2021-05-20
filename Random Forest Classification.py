from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset = pd.read_csv("CryptoCoins.csv")

x_train = dataset.iloc[:100, 2:3].values #:30
y_train = dataset.iloc[:100, 5].values #:30
x_test = dataset.iloc[110:210, 2:3].values #70:100
y_test = dataset.iloc[110:210, 5].values# 70:100

x_train = x_train.astype(int)
y_train = y_train.astype(int)
x_test = x_test.astype(int)
y_test = y_test.astype(int)

clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(x_train, y_train)

y_predic = clf.predict(x_test)

plt.scatter(x_train, y_train, color='red')
plt.plot(x_train, clf.predict(x_train), color='blue')
plt.title("Random Forest Classification")
plt.xlabel("Valor inicial")
plt.ylabel("Valor final")
plt.show()

# print(metrics.classification_report(expected, predicted))
# print(metrics.confusion_matrix(expected, predicted))
print(f"precision de arbol de decision:\n {round(clf.score(np.array(y_test).reshape(-1,1), np.array(y_predic).reshape(-1,1)) * 100, 2)}")
