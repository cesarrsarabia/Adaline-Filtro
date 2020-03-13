import numpy as np
from numpy import arange, sin, pi
import matplotlib
from pylab import xlim
from pylab import ylim
import random
import math
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

#Variables para generar señal con ruido
muestras = 10000
señal_ruido = []
Tiempo = 300
t = []
t = np.linspace(0, Tiempo, muestras)
señal_ruido = 3*np.sin(t/(3*np.pi)) +  np.random.rand(muestras) 


#Valores modificables tam_muestra y eta
muestra_tomada = 12 
eta= 0.003
#############
inicioEntradas = 0
w = np.random.rand(muestra_tomada,1)
b = np.random.rand(1)
y = 0


#Guarda señal con ruido
a = []
for i in señal_ruido:
    a.append(i)

f=open('ruido.txt','w')
f.write(str(señal_ruido))
f.close()


fig, ax = plt.subplots()
fig.patch.set_facecolor('xkcd:white')
plt.rcParams['axes.facecolor'] = 'white'
xlim([-0, Tiempo])
ylim([-3, 5])


arrayEntrenamiento = []
arrayEntrenamiento = señal_ruido[inicioEntradas:muestra_tomada]
c_posicion = 0
final = []

#Obtiene señal de txt
with open('ruido.txt','r') as f:
        MuestasRuido = f.read()
f.close()
RuidoString = np.array(MuestasRuido)



def get_punto_salida(w_2,b_2,entradas_2):
    c_posicion = 0
    y2 = 0
    for x in entradas_2:
        
        y2 = w_2[c_posicion]*entradas_2[c_posicion] + b_2
        c_posicion = c_posicion + 1
    return y2

#Entrenamiento
for x in range(0,int(len(señal_ruido)-1),1):
    entradas = señal_ruido[x:muestra_tomada]
    if muestra_tomada+1 >= int(len(señal_ruido)):
        y = get_punto_salida(w,b,entradas)
        final.append(y)
        error = señal_ruido[muestra_tomada] - y
        muestra_tomada = muestra_tomada + 1
        
        for index,value in enumerate(arrayEntrenamiento):
            w[index] += eta * error * value
            b += eta*error
        break
    else:
        y = get_punto_salida(w,b,entradas)
        final.append(y)
        error = señal_ruido[muestra_tomada+1] - y
        muestra_tomada = muestra_tomada + 1
        for index,value in enumerate(arrayEntrenamiento):
            w[index] += eta * error * value
            b += eta*error


#Fin Entrenamiento         
array_filtrado = []
array_filtrado = np.append(arrayEntrenamiento,final)

#Grafica la senoidal con ruido y la senoidal filtrada encima
plt.subplot(2, 1, 2)
plt.plot(t,señal_ruido,'-c')
plt.plot(t,array_filtrado,'-m')
plt.xlabel("Tiempo")
plt.ylabel("Señal")
plt.grid(True)

#Grafica señal con ruido sola
plt.subplot(2, 2, 1)
ruido_patch = mpatches.Patch(color='cyan', label='Señal con Ruido')
plt.legend(bbox_to_anchor=(0,1.2),handles=[ruido_patch],loc='upper left',borderaxespad=0.)
plt.plot(t,señal_ruido,':c')
plt.grid(True)


#Grafica señal limpia sola
plt.subplot(2, 2, 2)
clean_patch = mpatches.Patch(color='purple', label='Señal Limpia')
plt.legend(bbox_to_anchor=(0,1.2),handles=[clean_patch],loc='upper left',borderaxespad=0.)
plt.plot(t,array_filtrado,':m')

#Muestra Graficas
plt.grid(True)
plt.show()
