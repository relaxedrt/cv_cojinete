#Importamos la libreria necesaria para tratas imagenes
import cv2
import numpy as np

#ELEGIMOS EL UMBRAL DEL COLOR NEGRO EN HSV
umbral_bajo = (0,0,0)
umbral_alto = (0,0,0)

#Seleccionamos los vertices del recorte
x1 = 150
x2 = 350
y1 = 250
y2 = 1008

# Creamos un kernel basado en una matriz de 15x15
kernel = np.ones((22, 22), np.uint8)

#Leemos la imagen en concreto
img = cv2.imread("samples/IMG_3270.jpg")

#Escalamos la imagen
fde = 0.25
nuevas_dimensiones = (int(img.shape[1] * fde), int(img.shape[0] * fde))
scaled = cv2.resize(img, nuevas_dimensiones)

#Recortamos la imagen 
recorte = scaled[y1:y2, x1:x2]

#La convertimos a blanco y negro
img_bw = cv2.cvtColor(recorte, cv2.COLOR_BGR2GRAY)

#Erosionamos y dilatamos la imagen en blanco y negro
img_erosion = cv2.erode(img_bw, kernel, iterations=1)
img_dilation = cv2.dilate(img_erosion, kernel, iterations=1)

#Aplicamos threshold a la imagen
_, imagen_umbral = cv2.threshold(img_dilation, 30, 255, cv2.THRESH_BINARY)

#Buscamos los contronos de la imagen
cnts,_ = cv2.findContours(imagen_umbral, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#Dibujamos los contronos como prueba
cv2.drawContours(recorte, cnts, -1, (0,255,0), 2)

#La mostramos por pantalla
cv2.imshow('Imagen', imagen_umbral)
cv2.waitKey(0)
cv2.destroyAllWindows()