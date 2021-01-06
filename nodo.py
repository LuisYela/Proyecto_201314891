#!/usr/bin/env python
# -*- coding: utf-8 -*-

class Nodo:
    #solucion = []
    #fitness = 0 #Valor fitness
    #x = 0 #Para la tarea se guarda el valor de x

    #Le defino parámetros al constructor y le pongo valores por defecto por si no se envían
    def __init__(self, solucion = [], fitness = 0):
        self.solucion = solucion
        self.fitness = fitness