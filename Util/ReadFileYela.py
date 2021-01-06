import csv
import numpy as np
import math
distanciaDepartamentos = {}
distancias = []
edades = []
anios = []

def escalar(arreglo):
    #recive un arreglo asi [genero,edad,anio,distancia]
    # se escalan solo la edad, el anio y la distancia
    arregloNuevo=[]
    nuevaEdad=0
    nuevoAnio=0
    nuevaDist=0
    for linea in arreglo:
        nuevaLinea=[]
        nuevaLinea.append(linea[0])
        #print(linea[1])
        #print(linea[2])
        #print(linea[3])
        nuevaEdad=(linea[1] - min(edades) ) / ( max(edades) - min(edades) )
        nuevoAnio=(linea[2] - min(anios) ) / ( max(anios) - min(anios) )
        nuevaDist=(linea[3] - min(distancias) ) / ( max(distancias) - min(distancias) )
        nuevaLinea.append(nuevaEdad)
        nuevaLinea.append(nuevoAnio)
        nuevaLinea.append(nuevaDist)
        nuevaLinea=np.array(nuevaLinea)
        arregloNuevo.append(nuevaLinea)
        pass
    arregloNuevo=np.array(arregloNuevo)
    return arregloNuevo

            
def get_dataFile():
    set_y_origin = []
    set_x_origin = []

    try:
        with open('../datasets/Dataset.csv') as File:
            reader = csv.reader(File)
            encabezado=True
            for row in reader:
                if encabezado or row[6]=='Jocotenango' or row[6]=='San Miguel Duenias'or row[6]=='Comapa' or row[6]=='San Miguel Due?as':
                    encabezado=False
                    continue
                if row[0]=='Activo':
                    #si no se ha trasladado es 1
                    temp=np.ones(1)
                    set_y_origin.append(temp)
                    pass
                else:
                    #si se ha trasladado es 0
                    temp=np.zeros(1)
                    set_y_origin.append(temp)
                    pass
                temp=[]
                #distancia
                temp.append(distanciaDepartamentos[row[6]])
                #genero
                if row[1]=='MASCULINO':
                    #si es hombre es 1
                    temp.append(1)
                    pass
                else:
                    #si es mujer es 0
                    temp.append(0)
                    pass
                #edad
                temp.append(int(row[2]))
                #anio
                temp.append(int(row[7]))
                #convierto en array
                temp=np.array(temp)
                #guardo en mi lista de X
                set_x_origin.append(temp)
                #creo las listas para que se me haga facil el escalamiento
                distancias.append(distanciaDepartamentos[row[6]])
                edades.append(int(row[2]))
                anios.append(int(row[7]))
                #print(row[0])
    except OSError as err:
        #si entro aqui es por que esta leyendo mal el nombre en el archivo y no deberia pasar por que ya corregi nombres
        print('error de nombres'+ err)
        pass
    set_y_origin=np.array(set_y_origin)
    set_y_origin=set_y_origin.T
    
    set_x_origin=np.array(set_x_origin)
    #print('sin escalamiento de variables')
    #print(set_x_origin)
    set_x_origin=escalar(set_x_origin)
    #print('con escalamiento de variables')
    #print(set_x_origin)

    #haciendo la transpuesta
    set_x_origin=set_x_origin.T
    #se separan los datos de entrenamiento de los datos de prueba
    
    slice_point = int(set_x_origin.shape[1] * 0.7)
    train_set = set_x_origin[:, 0:slice_point ]
    test_set = set_x_origin[:, slice_point:]
    train_set_y = set_y_origin[:, 0:slice_point ]
    test_set_y = set_y_origin[:, slice_point:]
    #"""
    print("-------------------------------------------TRAIN-------------------------------------------")
    print(train_set.shape)
    print(train_set_y.shape)
    print("-------------------------------------------TEST-------------------------------------------")
    print(test_set.shape)
    print(test_set_y.shape)
    #print(set_y_origin)
    #print(set_y_origin.shape)
    #print(len(set_y_origin))
    #"""

    #return 0
    return train_set, train_set_y, test_set, test_set_y


def distancia(latitud, longitud):
    latU = 14.589246
    lonU = -90.551449

    rad = math.pi/180
    dlat = latU - latitud
    dlon = lonU - longitud
    R = 6372.795477598
    a = (math.sin(rad*dlat/2))**2 + math.cos(rad*latU)*math.cos(rad*latitud)*(math.sin(rad*dlon/2))**2
    distancia = 2*R*math.asin(math.sqrt(a))
    return distancia


with open('../datasets/Municipios.csv') as File:
    reader = csv.reader(File)
    encabezado=True
    for row in reader:
        if encabezado:
            encabezado=False
            continue
        distanciaDepartamentos[row[2]] = distancia(float(row[3]),float(row[4]))
        #print(float(row[3]))
        #print(row[3])
        pass
        #print(row[0])
    #print(distanciaDepartamentos)

get_dataFile()