"""Ejercicio integrador - Clase 3"""

import numpy as np


# Item 1 - Carga del dataset
# data0 = np.genfromtxt('clase3v2.csv', delimiter = ';')
from data import Data
data = Data(r'./clase3v2.csv')

# Item 2 - Particion del dataset en train/test con 80/20 usando la función del Ej4
pct_train = 0.8
pct_val = 0
pct_test = 0.2
train, _, test = data.split(pct_train, pct_val, pct_test)
#train, _, test = define_subsets(data, pct_train, pct_val, pct_test)


# Item 3 - Remover NaNs con las técnicas de los Ejercicios 2 y 3
train1 = data.removeNaNs(train)
test1 = data.removeNaNs(test)
train2 = data.replaceNaNs(train)
test2 = data.replaceNaNs(test)

x_train1 = train1[:, :-1]
y_train1 = train1[:, -1]
x_test1 = test1[:, :-1]
y_test1 = test1[:, -1]

x_train2 = train2[:, :-1]
y_train2 = train2[:, -1]
x_test2 = test2[:, :-1]
y_test2 = test2[:, -1]


# Item 4 - Utilizar PCA para retener las primeras 3 CP
from sklearn.decomposition import PCA
comp = 3
pca1_train = PCA(n_components=comp).fit(x_train1)
pca1_test = PCA(n_components=comp).fit(x_test1)
x_train_pca1 = pca1_train.transform(x_train1)
x_test_pca1 = pca1_test.transform(x_test1)

pca2_train = PCA(n_components=comp).fit(x_train2)
pca2_test = PCA(n_components=comp).fit(x_test2)
x_train_pca2 = pca2_train.transform(x_train2)
x_test_pca2 = pca2_test.transform(x_test2)


# Items 5 y 6: en archivos aparte
from metricas import MSE
from modelos import RegresionLineal

# Item 7.a - Entrenamiento de la regresión lineal sobre los conjuntos de train
# (train1 y train2). 
reg_lin1 = RegresionLineal()
reg_lin1.fit(x_train_pca1, y_train1)
y_pred1 = reg_lin1.predict(x_test_pca1)

reg_lin2 = RegresionLineal()
reg_lin2.fit(x_train_pca2, y_train2)
y_pred2 = reg_lin1.predict(x_test_pca2)


# Item 7.b - Calcular MSE sobre los sets correspondientes de validacion
mse = MSE()
mse_modelo1 = mse(y_test1.flatten(), y_pred1)
mse_modelo2 = mse(y_test2.flatten(), y_pred2)

print(f"MSE modelo 1 (dataset con valores NaN removidos): {mse_modelo1:.4f}")
print(f"MSE modelo 2 (dataset con valores NaN reemplazados por la media de los valores restantes de la columna): {mse_modelo2:.4f}")
