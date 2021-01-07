#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
import numpy as np
from nodo import Nodo
import csv
from Util.ReadFileYela import get_dataFile
from Util import Plotter
from Neural_Network.Data import Data
from Neural_Network.Model import NN_Model

alphas=[]
lambdas=[]
maxIterations=[]
kps=[]
# Cargando conjunto de datos
train_X, train_Y, val_X, val_Y = get_dataFile()

# Definir los conjuntos de datos
train_set = Data(train_X, train_Y)
val_set = Data(val_X, val_Y)

# Se define las dimensiones de las capas
#capas1 = [Cantidad de variables que tiene el problema, capa 1, capa 2, Capa de salida]
#se tendría una red neuronal de 3 capas, la capa de entrada NO se toma en cuenta
capas1 = [train_set.n, 20, 15, 5, 1]

#CONSTANTES DEL ALGORITMO
maximo_generaciones = 4 #Número máximo de generaciones que va a tener el algoritmo
#suma_anterior = 1 #Para guardar la suma de la población anterior

"""
*   Función que crea la población
"""
def inicializarPoblacion():
    poblacion = []

    #La población inicial ya la definió el ingeniero en la tabla
    #Individuo 1-10
    for i in range(10):
        alpha = random.randint(0,9)
        lambd = random.randint(0,9)
        max_i = random.randint(0,9)
        kp = random.randint(0,9)
        individuo = Nodo([alpha, lambd, max_i, kp], evaluarFitness([alpha, lambd, max_i, kp]))
        poblacion.append(individuo)
        pass
    return poblacion #Retorno la población ya creada

"""
*   Función que verifica si el algoritmo ya llegó a su fin
"""
def verificarCriterio(generacion):
    #Si ya llegó al máximo de generaciones lo detengo
    if generacion >= maximo_generaciones:
        return True
    return None
    """ 
    #Tomar la razón de 85% entre la suma del fitness de la generación anterior y la actual
    #se deja como numerador la suma menor
    if suma_anterior < suma_actual:
        numerador = suma_anterior
        denominador = suma_actual
    else:
        numerador = suma_actual
        denominador = suma_anterior

    razon = numerador / denominador

    #Actualizo la suma_anterior
    suma_anterior = suma_actual

    #Verifico si ya se llegó a una razón mayor o igual a 85%
    return True if razon >= 0.95 else None
    #"""
"""
*   Función que convierte un número binario a decimal
*   el número viene en un arreglo como este [0, 1, 1, 1, 0]
"""    
def convertirBinario(arreglo):
    base = 2 #Base 2 porque es binario
    decimal = 0
    for valor in arreglo:
        decimal = decimal * base + valor

    return decimal

"""
*   Función que evalúa qué tan buena es una solución, devuelve el valor fitness de la solución
*   @solucion = el número viene en un arreglo como este [0, 1, 1, 1]
"""
def evaluarFitness(solucion):
    #f(x) = 25x - x^2
    """print(alphas[solucion[0]])
    print(maxIterations[solucion[2]])
    print(lambdas[solucion[1]])
    print(kps[solucion[3]])"""
    nn1 = NN_Model(train_set, capas1, alpha=alphas[solucion[0]], iterations=maxIterations[solucion[2]], lambd= lambdas[solucion[1]], keep_prob=kps[solucion[3]])
    nn1.training(False)
    #print('Entrenamiento Modelo 1')
    #nn1.predict(train_set)
    print('Validacion Modelo')
    res=nn1.predict(val_set)
    #Retorno el valor fitness
    return res


"""
*   Función que toma a los mejores padres para luego crear una nueva generación
"""
def seleccionarPadres(poblacion):
    #Los padres se seleccionan por torneo
    #Mejor entre individuo 1 y 2 y mejor entre individuo 3 y 4
    padres = []

    #Mejor entre individuo 1 y 2
    individuo1 = poblacion[0]
    individuo2 = poblacion[1]
    padres.append(individuo2 if individuo2.fitness > individuo1.fitness else individuo1)

    #Mejor entre individuo 3 y 4
    individuo3 = poblacion[2]
    individuo4 = poblacion[3]
    padres.append(individuo4 if individuo4.fitness > individuo3.fitness else individuo3)

    #Mejor entre individuo 5 y 6
    individuo5 = poblacion[4]
    individuo6 = poblacion[5]
    padres.append(individuo6 if individuo6.fitness > individuo5.fitness else individuo5)

    #Mejor entre individuo 7 y 8
    individuo7 = poblacion[6]
    individuo8 = poblacion[7]
    padres.append(individuo8 if individuo8.fitness > individuo7.fitness else individuo7)

    #Mejor entre individuo 9 y 10
    individuo9 = poblacion[8]
    individuo10 = poblacion[9]
    padres.append(individuo10 if individuo10.fitness > individuo9.fitness else individuo9)

    return padres


"""
*   Función que toma dos soluciones padres y las une para formar una nueva solución hijo
*   Se va a alternar los bits de ambos padres
*   Se va a tomar un bit del padre 1, un bit del padre 2 y así sucesivamente
"""
def porcentaje(percent=50):
    return random.randrange(100) < percent

def cruzar(padre1, padre2):
    #padre1 = [1, 0, 1, 1]
    #padre2 = [0, 0, 1, 0]
    #hijo = [1, 0, 1, 0]
    hijo = []
    for i in range(4):
        if porcentaje:
            hijo.append(padre1[i])
            pass
        else:
            hijo.append(padre2[i])
            pass
        pass
    return hijo #Retorno al hijo ya cruzado


"""
*   Función que toma una solución y realiza la mutación
*   Se va a cambiar el bit con valor 0 más a la izquierda por 1
"""
def mutar(solucion):
    valorACambiar=random.randint(0,3)
    valoNuevo=random.randint(0,9)
    solucion[valorACambiar]=valoNuevo
    return solucion #Retorno la misma solución, solo que ahora mutó


"""
*   Función que toma a los mejores padres y genera nuevos hijos
"""
def emparejar(padres):

    #Se van a generar 2 nuevos hijos, se tienen 2 padres
    
    #Genero al hijo 1
    hijo1 = Nodo()
    hijo1.solucion = cruzar(padres[0].solucion, padres[1].solucion)
    hijo1.solucion = mutar(hijo1.solucion)
    hijo1.fitness = evaluarFitness(hijo1.solucion)

    #Genero al hijo 2
    hijo2 = Nodo()
    hijo2.solucion = cruzar(padres[2].solucion, padres[3].solucion)
    hijo2.solucion = mutar(hijo2.solucion)
    hijo2.fitness = evaluarFitness(hijo2.solucion)
    
    #Genero al hijo 3
    hijo3 = Nodo()
    hijo3.solucion = cruzar(padres[4].solucion, padres[0].solucion)
    hijo3.solucion = mutar(hijo3.solucion)
    hijo3.fitness = evaluarFitness(hijo3.solucion)

    #Genero al hijo 4
    hijo4 = Nodo()
    hijo4.solucion = cruzar(padres[1].solucion, padres[3].solucion)
    hijo4.solucion = mutar(hijo4.solucion)
    hijo4.fitness = evaluarFitness(hijo4.solucion)
    
    #Genero al hijo 5
    hijo5 = Nodo()
    hijo5.solucion = cruzar(padres[0].solucion, padres[2].solucion)
    hijo5.solucion = mutar(hijo5.solucion)
    hijo5.fitness = evaluarFitness(hijo5.solucion)

    #Creo un arreglo de hijos para luego ordenarlos
    hijos = [hijo1, hijo2, hijo3, hijo4, hijo5]

    #La nueva población se hará de la siguiente manera:
    #El mejor padre, el segundo mejor hijo, el segundo mejor padre, el mejor hijo
    nuevaPoblacion = [padres[0], hijos[4], padres[1], hijos[3], padres[2], hijos[2], padres[3], hijos[1], padres[4], hijos[0]]

    random.shuffle(nuevaPoblacion)

    return nuevaPoblacion

"""
*   Método para imprimir los datos de una población
"""
def imprimirPoblacion(poblacion):
    for individuo in poblacion:
        print('Individuo: ', individuo.solucion, ' Fitness: ', individuo.fitness)




"""
*   Método que ejecutará el algoritmo genético para obtener
*   los coeficientes del filtro
"""
def ejecutar():
    #np.seterr(over='raise')
    print("Algoritmo corriendo")

    generacion = 0
    poblacion = inicializarPoblacion()
    fin = None

    #Imprimo la población
    print('*************** GENERACION ', generacion, " ***************")
    imprimirPoblacion(poblacion)

    while(fin == None):
        padres = seleccionarPadres(poblacion)
        poblacion = emparejar(padres)
        generacion += 1 #Lo pongo aquí porque en teoría ya se creó una nueva generación
        fin = verificarCriterio(generacion)
        #generacion += 1

        #Imprimo la población
        print('*************** GENERACION ', generacion, " ***************")
        imprimirPoblacion(poblacion)

    #print('Cantidad de generaciones:', generacion)
    #imprimirPoblacion(poblacion) #Población final

    #Obtengo la mejor solución y la muestro
    arregloMejorIndividuo = sorted(poblacion, key=lambda item: item.fitness, reverse=True)[:1] #Los ordena de menor a mayor
    mejorIndividuo = arregloMejorIndividuo[0]

    print('\n\n*************** MEJOR INDIVIDUO***************')
    print('Individuo: ', mejorIndividuo.solucion, ' Fitness: ', mejorIndividuo.fitness)
    

with open('./datasets/AlgoritmoGeneticoY.csv') as File:
    reader = csv.reader(File)
    encabezado=True
    for row in reader:
        if encabezado:
            encabezado=False
            continue
        alphas.append(float(row[0]))
        lambdas.append(float(row[1]))
        maxIterations.append(int(row[2]))
        kps.append(float(row[3]))
        pass
        #print(row[0])
    print("ya")


#Corro el algoritmo
ejecutar()