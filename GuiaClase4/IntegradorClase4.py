"""Ejercicio integrador - Clase 4."""

import numpy as np

## ITEM 1: Genero un dataset con la misma funcion provista en clase
# Función a utilizar: y = |x^2-4|
from data import Data

# Genero todos los datos de una sola vez
n_samples = 2000
dataset = Data(n_samples)

# Ploteo la funcion original y la funcion corrompida con ruido blanco Gaussiano
dataset.plot_original()


## ITEM 2: Genero multiples regresiones
from metricas import MSE

def f_hat(x, W):
    d = len(W) - 1
    return np.sum(W * np.power(x, np.expand_dims(np.arange(d, -1, -1), 1)).T, 1)

# Ratio para definir conjunto de entrenamiento
pct_train = 0.8

# Defino cantidad de variantes en la partición de los conjuntos de train y test
n_variantes = 10
rand_seeds = np.arange(n_variantes)*100
XX_train = []
YY_train = []
XX_test = []
YY_test = []
Y_pred_train = []
Y_pred_test = []

# Defino cantidad de regresiones a computar para cada una de las particiones previamente definidas
n_regresiones = 30
regresiones_train = []
regresiones_test = []

# Defino arreglo para almacenar los valores de MSE de cada modelo
mse = MSE()
mse_train = np.zeros((n_regresiones, n_variantes))
mse_test = np.zeros((n_regresiones, n_variantes))


for i in range(n_variantes):
    
    # Divido el dataset en conjuntos de train y test
    x_train, x_test, y_train, y_test = dataset.split(pct_train, rand_seeds[i])
    XX_train.append(x_train)
    YY_train.append(y_train)
    XX_test.append(x_test)
    YY_test.append(y_test)
    Y_pred_train_i = []
    Y_pred_test_i = []
    regresiones_train_i = []
    regresiones_test_i = []
    
    for j in range(n_regresiones):
        
        # Entrenamiento de modelos de regresion usando conjunto de train
        w_train_j = np.polyfit(x_train, y_train, j)
        regresiones_train_i.append(w_train_j)
        
        # Predicciones sobre los conjuntos de train y test para el modelo entrenado
        y_pred_train_j = f_hat(x_train, w_train_j)
        Y_pred_train_i.append(y_pred_train_j)
        y_pred_test_j = f_hat(x_test, w_train_j)
        Y_pred_test_i.append(y_pred_test_j)
            
        # Computo errores de train y test
        mse_train_ij = mse(y_train, y_pred_train_j)
        mse_test_ij = mse(y_test, y_pred_test_j)
        mse_train[j][i] = mse_train_ij
        mse_test[j][i] = mse_test_ij

    Y_pred_train.append(Y_pred_train_i)
    Y_pred_test.append(Y_pred_test_i)
    regresiones_train.append(regresiones_train_i)
    regresiones_test.append(regresiones_test_i)


## ITEM 3: Obtengo resultados de los modelos entrenados
# Tabla de valores de MSE
import pandas as pd
tabla_mse_train = pd.DataFrame.from_records(mse_train)
tabla_mse_test = pd.DataFrame.from_records(mse_test)

filas = []
columnas = []
for i in range(n_regresiones): filas.append("Grado "+str(i))
for i in range(n_variantes): columnas.append("Seed "+str(rand_seeds[i]))

# Tabla de errores de train
tabla_mse_train.columns = columnas
tabla_mse_train.index = filas
print(tabla_mse_train)

# Tabla de errores de test
tabla_mse_test.columns = columnas
tabla_mse_test.index = filas
print(tabla_mse_test)



# Minimo MSE de train y test
min_MSE_train = np.min(mse_train)
min_MSE_test = np.min(mse_test)
min_train_ij = np.where(mse_train==min_MSE_train)
min_test_ij = np.where(mse_test==min_MSE_test)
print(f"El mínimo valor de MSE en test es {min_MSE_train:.4f}, obtenido con la partición N°{min_train_ij[1][0]:d} (seed={rand_seeds[min_train_ij[1][0]]:d}) \n y con un polinomio de grado {min_train_ij[0][0]:d}")
print(f"El mínimo valor de MSE en test es {min_MSE_test:.4f}, obtenido con la partición N°{min_test_ij[1][0]:d} (seed={rand_seeds[min_test_ij[1][0]]:d}) \n y con un polinomio de grado {min_test_ij[0][0]:d}")

# Ploteo de gráficas
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Graficas para conjunto de train, en la variante donde ocurrió el menor error
variante = min_train_ij[1][0]
regresion = min_train_ij[0][0]
fig, axs = plt.subplots(1, sharey = True, figsize=(15, 9))
axs.plot(XX_train[variante], YY_train[variante], ".", color = "Black")
colores = list(mcolors.CSS4_COLORS.keys())
x_range = np.linspace(-4, 4, n_samples)

for i in range(n_regresiones):
    axs.plot(x_range, f_hat(x_range, regresiones_train[variante][i]), linewidth=3.0, color=colores[i], label='Regresion Grado '+str(i))
    axs.legend()


# Graficas de error de train y test
fig, axs = plt.subplots(1, 2, sharey = True, figsize=(15, 9))
variante = min_train_ij[1][0]
regresion = min_train_ij[0][0]
axs[0].plot(np.arange(n_regresiones), mse_train[:, variante], linewidth=3.0, color = "Black")
axs[0].axvline(x = regresion)
axs[0].set_title('Error en conjunto de train')
variante = min_test_ij[1][0]
regresion = min_test_ij[0][0]
axs[1].plot(np.arange(n_regresiones), mse_test[:, variante], linewidth=3.0, color = "Black")
axs[1].axvline(x = regresion)
axs[1].set_title('Error en conjunto de test')

