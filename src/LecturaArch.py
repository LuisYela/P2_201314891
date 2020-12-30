import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import random

def load_dataset(numero):

    direccionImgUSAC = '../ImagenesDeEntrenamiento/Conjunto_Img_Pr2/USAC'
    direccionImgLandivar = '../ImagenesDeEntrenamiento/Conjunto_Img_Pr2/Landivar'
    direccionImgMariano = '../ImagenesDeEntrenamiento/Conjunto_Img_Pr2/Mariano'
    direccionImgMarroquin = '../ImagenesDeEntrenamiento/Conjunto_Img_Pr2/Marroquin'

    contenidoImgUSAC = os.listdir(direccionImgUSAC)
    contenidoImgLandivar = os.listdir(direccionImgLandivar)
    contenidoImgMariano = os.listdir(direccionImgMariano)
    contenidoImgMarroquin = os.listdir(direccionImgMarroquin)

    namesImgsUSAC = []
    namesImgsLandivar = []
    namesImgsMariano = []
    namesImgsMarroquin = []

    ImgsUSAC = []
    ImgsLandivar = []
    ImgsMariano = []
    ImgsMarroquin = []

    class datosAnalisis:
        datos_x = []
        datos_y = 0

        aux1=np.zeros((121))

        #datos_x=np.concatenate((ImgsUSAC, ImgsLandivar, ImgsMariano, ImgsMarroquin))
        #datos_y=np.concatenate((aux1,aux2), axis=0)
        
        def __init__(self, n, s):
            self.datos_x=n
            self.datos_y=s
        
        def print_x(self):
            print (self.datos_x)
            
        def print_y(self):
            print (self.datos_y)


    for ficheroUSAC in contenidoImgUSAC:
        if os.path.isfile(os.path.join(direccionImgUSAC, ficheroUSAC)) and (ficheroUSAC.endswith('.jpg') or ficheroUSAC.endswith('.png')):
            namesImgsUSAC.append(ficheroUSAC)
            train_set_x_aux = cv2.imread('../ImagenesDeEntrenamiento/Conjunto_Img_Pr2/USAC/'+ficheroUSAC)
            imagen = train_set_x_aux
            #print('Original: ', train_set_x_aux.shape)
            #print('Con reshape: ', imagen.shape)
            nuevaDatoUSAC= datosAnalisis(imagen,1)
            nuevaDatoLandivar= datosAnalisis(imagen,0)
            nuevaDatoMariano= datosAnalisis(imagen,0)
            nuevaDatomarroquin= datosAnalisis(imagen,0)
            ImgsUSAC.append(nuevaDatoUSAC)
            ImgsLandivar.append(nuevaDatoLandivar)
            ImgsMariano.append(nuevaDatoMariano)
            ImgsMarroquin.append(nuevaDatomarroquin)
    #print("---------------USAC---------------")
    #print(ImgsUSAC)

    for ficheroLandivar in contenidoImgLandivar:
        if os.path.isfile(os.path.join(direccionImgLandivar, ficheroLandivar)) and (ficheroLandivar.endswith('.jpg') or ficheroLandivar.endswith('.png')):
            namesImgsLandivar.append(ficheroLandivar)
            train_set_x_aux = cv2.imread('../ImagenesDeEntrenamiento/Conjunto_Img_Pr2/Landivar/'+ficheroLandivar)
            imagen = train_set_x_aux
            #print('Original: ', train_set_x_aux.shape)
            #print('Con reshape: ', imagen.shape)
            nuevaDatoUSAC= datosAnalisis(imagen,0)
            nuevaDatoLandivar= datosAnalisis(imagen,1)
            nuevaDatoMariano= datosAnalisis(imagen,0)
            nuevaDatomarroquin= datosAnalisis(imagen,0)
            ImgsUSAC.append(nuevaDatoUSAC)
            ImgsLandivar.append(nuevaDatoLandivar)
            ImgsMariano.append(nuevaDatoMariano)
            ImgsMarroquin.append(nuevaDatomarroquin)
    #print("---------------Landivar---------------")
    #print(namesImgsLandivar)


    for ficheroMariano in contenidoImgMariano:
        if os.path.isfile(os.path.join(direccionImgMariano, ficheroMariano)) and (ficheroMariano.endswith('.jpg') or ficheroMariano.endswith('.png')):
            namesImgsMariano.append(ficheroMariano)
            train_set_x_aux = cv2.imread('../ImagenesDeEntrenamiento/Conjunto_Img_Pr2/Mariano/'+ficheroMariano)
            imagen = train_set_x_aux
            #print('Original: ', train_set_x_aux.shape)
            #print('Con reshape: ', imagen.shape)
            nuevaDatoUSAC= datosAnalisis(imagen,0)
            nuevaDatoLandivar= datosAnalisis(imagen,0)
            nuevaDatoMariano= datosAnalisis(imagen,1)
            nuevaDatomarroquin= datosAnalisis(imagen,0)
            ImgsUSAC.append(nuevaDatoUSAC)
            ImgsLandivar.append(nuevaDatoLandivar)
            ImgsMariano.append(nuevaDatoMariano)
            ImgsMarroquin.append(nuevaDatomarroquin)
    #print("---------------Marinao---------------")
    #print(namesImgsMariano)

    for ficheroMarroquin in contenidoImgMarroquin:
        if os.path.isfile(os.path.join(direccionImgMarroquin, ficheroMarroquin)) and (ficheroMarroquin.endswith('.jpg') or ficheroMarroquin.endswith('.png')):
            namesImgsMarroquin.append(ficheroMarroquin)
            train_set_x_aux = cv2.imread('../ImagenesDeEntrenamiento/Conjunto_Img_Pr2/Marroquin/'+ficheroMarroquin)
            imagen = train_set_x_aux
            #print('Original: ', train_set_x_aux.shape)
            #print('Con reshape: ', imagen.shape)
            nuevaDatoUSAC= datosAnalisis(imagen,0)
            nuevaDatoLandivar= datosAnalisis(imagen,0)
            nuevaDatoMariano= datosAnalisis(imagen,0)
            nuevaDatomarroquin= datosAnalisis(imagen,1)
            ImgsUSAC.append(nuevaDatoUSAC)
            ImgsLandivar.append(nuevaDatoLandivar)
            ImgsMariano.append(nuevaDatoMariano)
            ImgsMarroquin.append(nuevaDatomarroquin)
    #print("---------------Marroquin---------------")
    #print(namesImgsMarroquin)

    """print(("--------------------------------USAC--------------------------------"))
    print(len(ImgsUSAC))
    print(("--------------------------------Landivar--------------------------------"))
    print(len(ImgsLandivar))
    print(("--------------------------------Mariano--------------------------------"))
    print(len(ImgsMariano))
    print(("--------------------------------Marroquin--------------------------------"))
    print(len(ImgsMarroquin))"""

    random.shuffle(ImgsUSAC)
    random.shuffle(ImgsLandivar)
    random.shuffle(ImgsMariano)
    random.shuffle(ImgsMarroquin)

    imagenes=[]
    resultado=[]
    if numero==1:
        result=[]
        result_y=[]
        for fila in ImgsUSAC:
            #print(fila.datos_y)
            imagenes.append(fila.datos_x)
            result.append(fila.datos_y)
        imagenes= np.array(imagenes)
        resultado.append(np.array(result))
        resultado=np.array(resultado)
        #print(resultado)
        #print(len(imagenes))
        #print(imagenes.shape)
        #print(len(resultado))
        #print(resultado.shape)
        #return train_set_x, train_set_y, test_set_x, test_set_y, ['1', '0']
        slice_point = int(imagenes.shape[0] * 0.7)
        train_set = imagenes[0:slice_point, : ]
        test_set = imagenes[slice_point:, :]
        train_set_y_origin = resultado[:, 0:slice_point ]
        test_set_y_origin = resultado[:, slice_point:]
        #print(train_set.shape)
        #print(test_set.shape)
        #print(train_set_y_origin.shape)
        #print(test_set_y_origin.shape)
        return train_set, train_set_y_origin, test_set, test_set_y_origin, ['1', '0']

    if numero==2:
        result=[]
        result_y=[]
        for fila in ImgsLandivar:
            #print(fila.datos_y)
            imagenes.append(fila.datos_x)
            result.append(fila.datos_y)
        imagenes= np.array(imagenes)
        resultado.append(np.array(result))
        resultado=np.array(resultado)
        #print(resultado)
        #print(len(imagenes))
        #print(imagenes.shape)
        #print(len(resultado))
        #print(resultado.shape)
        #return train_set_x, train_set_y, test_set_x, test_set_y, ['1', '0']
        slice_point = int(imagenes.shape[0] * 0.7)
        train_set = imagenes[0:slice_point, : ]
        test_set = imagenes[slice_point:, :]
        train_set_y_origin = resultado[:, 0:slice_point ]
        test_set_y_origin = resultado[:, slice_point:]
        #print(train_set.shape)
        #print(test_set.shape)
        #print(train_set_y_origin.shape)
        #print(test_set_y_origin.shape)
        return train_set, train_set_y_origin, test_set, test_set_y_origin, ['1', '0']
        
    if numero==3:
        result=[]
        result_y=[]
        for fila in ImgsMariano:
            #print(fila.datos_y)
            imagenes.append(fila.datos_x)
            result.append(fila.datos_y)
        imagenes= np.array(imagenes)
        resultado.append(np.array(result))
        resultado=np.array(resultado)
        #print(resultado)
        #print(len(imagenes))
        #print(imagenes.shape)
        #print(len(resultado))
        #print(resultado.shape)
        #return train_set_x, train_set_y, test_set_x, test_set_y, ['1', '0']
        slice_point = int(imagenes.shape[0] * 0.7)
        train_set = imagenes[0:slice_point, : ]
        test_set = imagenes[slice_point:, :]
        train_set_y_origin = resultado[:, 0:slice_point ]
        test_set_y_origin = resultado[:, slice_point:]
        #print(train_set.shape)
        #print(test_set.shape)
        #print(train_set_y_origin.shape)
        #print(test_set_y_origin.shape)
        return train_set, train_set_y_origin, test_set, test_set_y_origin, ['1', '0']

    if numero==4:
        result=[]
        result_y=[]
        for fila in ImgsMarroquin:
            #print(fila.datos_y)
            imagenes.append(fila.datos_x)
            result.append(fila.datos_y)
        imagenes= np.array(imagenes)
        resultado.append(np.array(result))
        resultado=np.array(resultado)
        #print(resultado)
        #print(len(imagenes))
        #print(imagenes.shape)
        #print(len(resultado))
        #print(resultado.shape)
        #return train_set_x, train_set_y, test_set_x, test_set_y, ['1', '0']
        slice_point = int(imagenes.shape[0] * 0.7)
        train_set = imagenes[0:slice_point, : ]
        test_set = imagenes[slice_point:, :]
        train_set_y_origin = resultado[:, 0:slice_point ]
        test_set_y_origin = resultado[:, slice_point:]
        #print(train_set.shape)
        #print(test_set.shape)
        #print(train_set_y_origin.shape)
        #print(test_set_y_origin.shape)
        return train_set, train_set_y_origin, test_set, test_set_y_origin, ['1', '0']

#load_dataset(1)