#!/usr/bin/env python
# -*- coding: utf-8 -*-

import LecturaArch
from Logistic_Regression.Model import Model
from Logistic_Regression.Data import Data
from Logistic_Regression import Plotter
import numpy as np

ONLY_SHOW = False #Veo si quiero mostrar una imagen del conjunto de datos

#Cargando conjuntos de datos
train_set_x_origU, train_set_yU, test_set_x_origU, test_set_yU, classesU = LecturaArch.load_dataset(1)
train_set_x_origL, train_set_yL, test_set_x_origL, test_set_yL, classesL = LecturaArch.load_dataset(2)
train_set_x_origM, train_set_yM, test_set_x_origM, test_set_yM, classesM = LecturaArch.load_dataset(3)
train_set_x_origR, train_set_yR, test_set_x_origR, test_set_yR, classesR = LecturaArch.load_dataset(4)

"""if ONLY_SHOW:
    #index = 14 #Gato
    index = 14 #No Gato
    index = 59 #Gato
    Plotter.show_picture(train_set_x_origU[index])
    print(classesU[train_set_yU[0][index]])
    exit()
"""

# Convertir imagenes a un solo arreglo
train_set_xU = train_set_x_origU.reshape(train_set_x_origU.shape[0], -1).T
test_set_xU = test_set_x_origU.reshape(test_set_x_origU.shape[0], -1).T

train_set_xL = train_set_x_origL.reshape(train_set_x_origL.shape[0], -1).T
test_set_xL = test_set_x_origL.reshape(test_set_x_origL.shape[0], -1).T

train_set_xM = train_set_x_origM.reshape(train_set_x_origM.shape[0], -1).T
test_set_xM = test_set_x_origM.reshape(test_set_x_origM.shape[0], -1).T

train_set_xR = train_set_x_origR.reshape(train_set_x_origR.shape[0], -1).T
test_set_xR = test_set_x_origR.reshape(test_set_x_origR.shape[0], -1).T

#Vemos cómo queda ahora la estructura de una imagen
#(12288, 209) En este caso tiene 209 registros y cada registro tiene 12288 valores
#En el caso de las notas cada registro tenía solo 3 valores, que eran las 3 notas
#Por lo tanto, nuestro modelo va a tener 12288 + 1 Coeficientes, el + 1 es por B0
#print(train_set_x.shape)

# Vean la diferencia de la conversion
print('Original: ', train_set_x_origU.shape)
print('Con reshape: ', train_set_xU.shape)

#print('tamaño train_set_x_orig: ', len(train_set_x_orig))
#print('tamaño train_set_x: ', len(train_set_x))

#temp = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1)
#print('Prueba: ', temp.shape)

#print('train_set_x')
#print(train_set_x)
#exit()


# Definir los conjuntos de datos
train_setU = Data(train_set_xU, train_set_yU, 255)
test_setU = Data(test_set_xU, test_set_yU, 255)

train_setL = Data(train_set_xL, train_set_yL, 255)
test_setL = Data(test_set_xL, test_set_yL, 255)

train_setM = Data(train_set_xM, train_set_yM, 255)
test_setM = Data(test_set_xM, test_set_yM, 255)

train_setR = Data(train_set_xR, train_set_yR, 255)
test_setR = Data(test_set_xR, test_set_yR, 255)

#print("estoy entrando a analizar")

# Se entrenan los modelos
#model1 = Model(train_set, test_set, reg=False, alpha=0.0001, lam=0)
#model1.training()

#model2 = Model(train_set, test_set, reg=True, alpha=0.01, lam=1) #Se puede ver en la gráfica que hay SOBRE-AJUSTE
#model2 = Model(train_set, test_set, reg=True, alpha=0.01, lam=150) #Aquí también se puede ver sobre-ajuste

#model2 = Model(train_set, test_set, reg=True, alpha=0.001, lam=300) #Se ajusta mejor con la regulariación de 300, pero se tarda más

#modelU1 = Model(train_setU, test_setU, reg=False, alpha=0.001, lam=100) #Baja más quitandole la regularización
#modelU1.training()
#modelU2 = Model(train_setU, test_setU, reg=False, alpha=0.000001, lam=150) #Baja más quitandole la regularización
#modelU2.training()
#modelU3 = Model(train_setU, test_setU, reg=False, alpha=0.001, lam=300) #Baja más quitandole la regularización
#modelU3.training()
#modelU4 = Model(train_setU, test_setU, reg=False, alpha=0.0001, lam=350) #Baja más quitandole la regularización
#modelU4.training()
#modelU5 = Model(train_setU, test_setU, reg=False, alpha=0.00001, lam=450) #Baja más quitandole la regularización
#modelU5.training()

#modelL1 = Model(train_setU, test_setU, reg=False, alpha=0.001, lam=100) #Baja más quitandole la regularización
#modelL1.training()
#modelL2 = Model(train_setU, test_setU, reg=False, alpha=0.000001, lam=150) #Baja más quitandole la regularización
#modelL2.training()
#modelL3 = Model(train_setU, test_setU, reg=False, alpha=0.0001, lam=400) #Baja más quitandole la regularización
#modelL3.training()
#modelL4 = Model(train_setU, test_setU, reg=False, alpha=0.0001, lam=350) #Baja más quitandole la regularización
#modelL4.training()
#modelL5 = Model(train_setU, test_setU, reg=False, alpha=0.00001, lam=450) #Baja más quitandole la regularización
#modelL5.training()

#modelM1 = Model(train_setU, test_setU, reg=False, alpha=0.001, lam=100) #Baja más quitandole la regularización
#modelM1.training()
#modelM2 = Model(train_setU, test_setU, reg=False, alpha=0.000001, lam=150) #Baja más quitandole la regularización
#modelM2.training()
#modelM3 = Model(train_setU, test_setU, reg=False, alpha=0.0001, lam=400) #Baja más quitandole la regularización
#modelM3.training()
#modelM4 = Model(train_setU, test_setU, reg=False, alpha=0.0001, lam=350) #Baja más quitandole la regularización
#modelM4.training()
#modelM5 = Model(train_setU, test_setU, reg=False, alpha=0.00001, lam=100) #Baja más quitandole la regularización
#modelM5.training()

#modelR1 = Model(train_setU, test_setU, reg=False, alpha=0.001, lam=100) #Baja más quitandole la regularización
#modelR1.training()
#modelR2 = Model(train_setU, test_setU, reg=False, alpha=0.000001, lam=500) #Baja más quitandole la regularización
#modelR2.training()
#modelR3 = Model(train_setU, test_setU, reg=False, alpha=0.0001, lam=400) #Baja más quitandole la regularización
#modelR3.training()
#modelR4 = Model(train_setU, test_setU, reg=False, alpha=0.0001, lam=350) #Baja más quitandole la regularización
#modelR4.training()
#modelR5 = Model(train_setU, test_setU, reg=False, alpha=0.00001, lam=100) #Baja más quitandole la regularización
#modelR5.training()

modelU = Model(train_setU, test_setU, reg=False, alpha=0.001, lam=150) #Baja más quitandole la regularización
modelU.training()
modelL = Model(train_setL, test_setL, reg=False, alpha=0.0001, lam=200) #Baja más quitandole la regularización
modelL.training()
modelM = Model(train_setM, test_setM, reg=False, alpha=0.0001, lam=100) #Baja más quitandole la regularización
modelM.training()
modelR = Model(train_setR, test_setR, reg=False, alpha=0.00001, lam=600) #Baja más quitandole la regularización
modelR.training()

#MODELOS ELEJIDOS

# Se grafican los entrenamientos
#Plotter.show_Model([model1, model2])
#Plotter.show_Model([model1])

#Plotter.show_Model([modelR1])
#Plotter.show_Model([modelR2])
#Plotter.show_Model([modelR3])
#Plotter.show_Model([modelR4])
#Plotter.show_Model([modelR5])
#Plotter.show_Model([modelR1, modelR2, modelR3, modelR4, modelR5])

#Plotter.show_Model([modelU])

#Plotter.show_Model([modelL])

#Plotter.show_Model([modelM])

#Plotter.show_Model([modelR])

Plotter.show_Model([modelU, modelL, modelM, modelR])
