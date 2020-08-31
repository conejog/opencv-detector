import numpy as np
import argparse
import cv2
from os.path import dirname, join
#Para reproducir un .mp3 importamos playsound
#import playsound
import winsound
import time

frequency = 2500  #Frequencia del beep 2500 Hertz
duration = 500  #Duracion del beep. 1000 ms == 1 segundo

#Constructor y sus argumentos. Confidencia seteada en 0.3 por default.
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--confidence", type=float, default=0.3,
	help="minimum probability to filter weak detections")
protoPath = join(dirname(__file__), "MobileNetSSD_deploy.prototxt")
modelPath = join(dirname(__file__), "MobileNetSSD_deploy.caffemodel")
args = vars(ap.parse_args())
#Se setean los labels de cada clase y se le asigna un color random para referencia. Se pueden cambiar a gusto.
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
#Cargamos el modelo
print("Modelo cargado. Iniciamos deteccion.")
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

#Cargamos imagen, ajustamos a 300x300 y normalizamos para el modelo.

#Levantamos la webcam de la notebook 
cap = cv2.VideoCapture(0)

#Este codigo funciona tambien con una camara remota. Se debera conocer la url de rtsp para poder configurarla.
#Eliminar usuario:password@ si la camara no requiere autenticacion.
#cap = cv2.VideoCapture('rtsp://usuario:password@192.168.0.0:554/video.mp4')
while 1: 
        ret, img = cap.read() 
        image = img
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843,
                (300, 300), 127.5)
        # enviamos el blob y procesamos las detecciones. Descomentar la siguiente linea para informar el proceso.
        #print("Procesando detecciones.")
        net.setInput(blob)
        detections = net.forward()
        for i in np.arange(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                #Filter detecciones por valor de confidencia seteado
                if confidence > args["confidence"]:
                        #Si coincide lo marcamos en la imagen
                        idx = int(detections[0, 0, i, 1])
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")
                        #Informamos la deteccion. 
                        label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
                        #Esta linea filtra por gatos, perros o personas.
                        if CLASSES[idx] == 'cat' or CLASSES[idx] == 'dog'  or CLASSES[idx] == 'person':             
                            #Descomentar si decidimos utilizar un .mp3 (y comentar el uso de winsound.Beep
                            #playsound('alarma.mp3', False)                            
                            winsound.Beep(frequency, duration)
                            print("Detectado {}".format(label))
                            cv2.rectangle(image, (startX, startY), (endX, endY),
                                    COLORS[idx], 2)
                            y = startY - 15 if startY - 15 > 15 else startY + 15
                            cv2.putText(image, label, (startX, y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
                            #almaceno una captura en la carpeta shots, si no se quiere guardar la imagen comentar las proximas 3 lineas
                            timestr = CLASSES[idx] + "-" + time.strftime("%Y%m%d-%H%M%S") + '.jpg' 
                            filename = "shots/" + timestr
                            cv2.imwrite(filename, image)  
        #Mostramos imagen 
        cv2.imshow("Output", image)
        k = cv2.waitKey(50) & 0xff
        #Para salir presionar tecla ESC
        if k == 27: 
            break

#Cerramos ventana y limpiamos memoria
cap.release() 
cv2.destroyAllWindows() 
