from Util.ReadFileYela import get_dataFile
from Util import Plotter
from Neural_Network.Data import Data
from Neural_Network.Model import NN_Model
import numpy as np


distancia=200
edad=19
anio=2010
genero=int(0)
#escalando los valores
distancia=(distancia - 6.135217051724262 ) / ( 269.7739244046257 - 6.135217051724262 )
edad=(edad - 17 ) / ( 57 - 17 )
anio=(anio - 2010 ) / ( 2019 - 2010 )
genero=0

prediccion = [genero,edad,anio,distancia]
prediccion=np.array(prediccion)
arrpr=[prediccion]
arrpr=(np.array(arrpr)).T
arrResp=np.zeros(1)
arrResp2=[arrResp]
arrResp2=(np.array(arrResp2)).T
mi_prediccion=Data(arrpr,arrResp2)
print(prediccion.shape)
print(prediccion)


# Cargando conjunto de datos
train_X, train_Y, val_X, val_Y = get_dataFile()

# Definir los conjuntos de datos
train_set = Data(train_X, train_Y)
val_set = Data(val_X, val_Y)

# Se define las dimensiones de las capas
#capas1 = [Cantidad de variables que tiene el problema, capa 1, capa 2, Capa de salida]
#se tendría una red neuronal de 3 capas, la capa de entrada NO se toma en cuenta
capas1 = [train_set.n, 10, 5, 1]

# Se define el modelo
nn1 = NN_Model(train_set, capas1, alpha=0.001, iterations=5000, lambd=0, keep_prob=0.50)
nn2 = NN_Model(train_set, capas1, alpha=0.01, iterations=5000, lambd=0.7, keep_prob=1)

# Se entrena el modelo
nn1.training(False)
nn2.training(False)

# Se analiza el entrenamiento
Plotter.show_Model([nn1, nn2])

#print(train_set.shape)

print('Entrenamiento Modelo 1')
nn1.predict(train_set)
print('Validacion Modelo 1')
nn1.predict(val_set)

#"""
print('########################')
print('Entrenamiento Modelo 2')
nn2.predict(train_set)
print('Validacion Modelo 2')
nn2.predict(val_set)
#"""

#"""
print('########################')
print('probando un dato x')
print(nn2.mi_predict(mi_prediccion))
#"""

