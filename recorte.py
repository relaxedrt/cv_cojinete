#Importamos la libreria necesaria para tratas imagenes
import cv2
import numpy as np

#Seleccionamos los vertices del recorte
x1 = 200
x2 = 300
y1 = 250
y2 = 960

#Creamos un array donde colocaremos los valores de x e y
posiciones = []

# Creamos dos kernel basado en una matriz
kernel_erode = np.ones((9, 9), np.uint8)
kernel_dilate = np.ones((10, 10), np.uint8)

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

#Aplicamos threshold a la imagen
_, imagen_umbral = cv2.threshold(img_bw, 20, 255, cv2.THRESH_BINARY)

#Erosionamos y dilatamos la imagen en blanco y negro
img_erosion = cv2.erode(imagen_umbral, kernel_erode, iterations=1)
img_dilation = cv2.dilate(img_erosion, kernel_dilate, iterations=1)

#Creamos la mascara
_, mask = cv2.threshold(img_dilation, 15, 255, cv2.THRESH_BINARY)

#Buscamos los contornos
contornos1,hierarchy1 = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

if len(contornos1) == 3:
    #Hay el numero correcto de taladros
    print("Pieza con el número correcto de taladros.")
    for c in contornos1:
        #Calculamos la posición del taladro
        epsilon = 0.01*cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        x, y, w, h = cv2.boundingRect(approx)
        #Solo si tiene mas de 4 lados lo usamos como taladro, asi nos quitamos bordes
        if len(approx) > 4:
            cv2.putText(recorte, "Taladro", (x, y -5), 1, 1, (0,255,0), 2)
            #Dibujamos los contronos como prueba
            cv2.drawContours(recorte, c, -1, (0,255,0), 2)
            #Guardamos la coordenada x e y de la esquina superior izquierda
            posiciones.append([ x, y])

    #Calculamos la distancia entre los dos puntos
    d = np.sqrt(((posiciones[0][0] - posiciones[1][0]) * (posiciones[0][0] - posiciones[1][0])) + ((posiciones[0][1] - posiciones[1][1]) * (posiciones[0][1] - posiciones[1][1])))
    print(f"La pieza tiene {d} pixels entre centros")
    #Dibujamos la linea
    cv2.line(recorte, posiciones[0], posiciones[1], (0,255,0), 2)
    #Escribimos la distancia
    #cv2.putText(recorte, "yes", ((posiciones[0][0] / posiciones[1][0]) + 3, (posiciones[0][1] / posiciones[1][1])), 1, 1, (0,255,0), 2)

#La mostramos por pantalla
cv2.imshow('Imagen', recorte)
cv2.waitKey(0)
cv2.destroyAllWindows()