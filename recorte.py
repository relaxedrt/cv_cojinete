#Importamos la libreria necesaria para tratas imagenes
import cv2
import numpy as np

#Seleccionamos los vertices del recorte
x1 = 190
x2 = 300
y1 = 250
y2 = 960

#Medida standar entre centros
distok = 116.0
umbral_dist = 0.5
pxmm = 4.99137931

#Creamos un array donde colocaremos los valores de x e y
posiciones = []

# Creamos dos kernel basado en una matriz
kernel_erode = np.ones((9, 9), np.uint8)
kernel_dilate = np.ones((10, 10), np.uint8)

#Leemos la imagen en concreto
img = cv2.imread("samples/IMG_3267.jpg")

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

if len(contornos1) > 1:
    #Hay el numero correcto de taladros
    for c in contornos1:
        #Calculamos la posición del taladro
        epsilon = 0.01*cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        x, y, w, h = cv2.boundingRect(approx)
        #Solo si tiene mas de 10 lados lo usamos como taladro, asi nos quitamos bordes
        if len(approx) > 10:
            M = cv2.moments(c)
            if M['m00'] != 0:
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
            cv2.putText(recorte, "Taladro", (x, y -5), 1, 1, (0,0,0), 2)
            #Dibujamos los contronos como prueba
            cv2.drawContours(recorte, c, -1, (0,255,0), 2)
            #Guardamos la coordenada x e y de la esquina superior izquierda
            posiciones.append([ cx, cy])

    #Comprobamos que haya habido 2 circulos
    if len(posiciones) == 2:
        print("Pieza con el número correcto de taladros.")
        #Calculamos la distancia entre los dos puntos
        d = np.sqrt(((posiciones[0][0] - posiciones[1][0]) * (posiciones[0][0] - posiciones[1][0])) + ((posiciones[0][1] - posiciones[1][1]) * (posiciones[0][1] - posiciones[1][1])))
        #Escribimos la distancia
        midx = (posiciones[0][0] / posiciones[1][0]) + 3
        midy = (posiciones[0][1] / posiciones[1][1])
        #cv2.putText(recorte, "", (midx, midy), 1, 1, (0,0,0), 2)
        if ((d/pxmm) > (distok + umbral_dist)) or ((d/pxmm) < (distok - umbral_dist)):
            #Dibujamos la linea
            cv2.line(recorte, posiciones[0], posiciones[1], (0,0,255), 2)
            print("Medición entre centros erronea.")
            print(f"La pieza tiene {d/pxmm} pixels entre centros.")
        else :
            #Dibujamos la linea
            cv2.line(recorte, posiciones[0], posiciones[1], (0,255,0), 2)
            print("Medición entre centros correcta.")
            print(f"La pieza tiene {d/pxmm} pixels entre centros.")
    else:
        print("La deteccion ha fallado.")
        print(f"No se han encontrado la cantidad de taladros correcta.")
else:
    print("La deteccion ha fallado.")
#La mostramos por pantalla
cv2.imshow('Imagen', recorte)
cv2.waitKey(0)
cv2.destroyAllWindows()