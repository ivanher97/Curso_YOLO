# ---------------------------------------
# Importamos las librerías necesarias
# ---------------------------------------
import cv2                      # OpenCV para visión por computadora
import numpy as np              # NumPy para cálculos numéricos
import yaml                     # YAML para leer archivos de configuración de clases
from yaml.loader import SafeLoader  # Cargador seguro de archivos YAML

# ---------------------------------------
# Cargar nombres de clases desde un archivo .txt en formato YAML
# ---------------------------------------
# Este archivo debe contener algo como:
# names:
#   - persona
#   - coche
#   - perro
with open('data.txt', mode='r') as f:
    data_yaml = yaml.load(f, Loader=SafeLoader)  # Cargamos el YAML de forma segura

labels = data_yaml['names']  # Obtenemos la lista de nombres de las clases

# ---------------------------------------
# Cargar el modelo YOLO exportado en formato ONNX
# ---------------------------------------
# Asegúrate de haber exportado el modelo de YOLOv5 a formato .onnx previamente
yolo = cv2.dnn.readNetFromONNX('modelo.onnx')  # <-- CAMBIA esto por la ruta a tu archivo .onnx

# Definimos que usaremos OpenCV como backend y la CPU como destino de cómputo
yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# ---------------------------------------
# Cargar una imagen desde disco
# ---------------------------------------
img = cv2.imread('imagen.jpg')  # <-- CAMBIA esto por la ruta a tu imagen

# Hacemos una copia para preservar la imagen original sin modificaciones
image = img.copy()

# ---------------------------------------
# Mostrar la imagen original (opcional)
# ---------------------------------------
cv2.imshow('Imagen original', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ---------------------------------------
# Convertir la imagen en una imagen cuadrada
# ---------------------------------------
# El modelo YOLOv5 espera una imagen de entrada cuadrada (ej: 640x640)
row, col, d = image.shape  # Obtenemos alto, ancho y profundidad de la imagen

max_rc = max(row, col)  # Calculamos el tamaño más grande para hacerla cuadrada
input_img = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)  # Creamos una imagen negra cuadrada

# Pegamos la imagen original en la esquina superior izquierda de la cuadrada
input_img[0:row, 0:col] = image

# ---------------------------------------
# Preprocesar la imagen para que sea compatible con el modelo
# ---------------------------------------
INP_WH_MODEL = 640  # Tamaño que espera el modelo (640x640)

# Convertimos la imagen a blob:
# - escala de píxeles: 1/255 para pasar de 0–255 a 0–1
# - cambio de tamaño: 640x640
# - conversión de BGR a RGB: swapRB=True
blob = cv2.dnn.blobFromImage(
    input_img, 1/255, (INP_WH_MODEL, INP_WH_MODEL),
    swapRB=True, crop=False
)

# ---------------------------------------
# Enviar la imagen al modelo y obtener predicciones
# ---------------------------------------
yolo.setInput(blob)         # Enviamos el blob al modelo
pred = yolo.forward()       # Ejecutamos el modelo y guardamos las predicciones

# ---------------------------------------
# Mostrar la imagen cuadrada enviada al modelo (opcional)
# ---------------------------------------
cv2.imshow('Imagen cuadrada para el modelo', input_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ---------------------------------------
# Procesar las predicciones del modelo
# ---------------------------------------
detections = pred[0]  # Lista de todas las predicciones que hizo el modelo

boxes = []        # Coordenadas de las cajas (x, y, ancho, alto)
confidences = []  # Nivel de confianza de cada detección
classes = []      # ID de clase de cada detección

# Factores de escala para convertir coordenadas normalizadas a píxeles
image_w, image_h = input_img.shape[:2]
x_factor = image_w / INP_WH_MODEL
y_factor = image_h / INP_WH_MODEL

# ---------------------------------------
# Recorremos cada detección y filtramos las válidas
# ---------------------------------------
for i in range(len(detections)):
    rw = detections[i]  # Una fila del resultado (una predicción)

    confidence = rw[4]  # Confianza general de que hay un objeto

    if confidence > 0.4:  # Solo procesamos si es confiable
        class_score = rw[5:].max()     # Probabilidad más alta entre las clases
        class_id = rw[5:].argmax()     # Clase que obtuvo la puntuación más alta

        if class_score > 0.25:  # También filtramos por puntuación de clase
            cx, cy, w, h = rw[0:4]  # Coordenadas normalizadas del centro y tamaño

            # Convertimos a coordenadas absolutas en píxeles
            left = int((cx - 0.5 * w) * x_factor)
            top = int((cy - 0.5 * h) * y_factor)
            ancho = int(w * x_factor)
            alto = int(h * y_factor)

            # Guardamos todo
            box = np.array([left, top, ancho, alto])
            confidences.append(confidence)
            boxes.append(box)
            classes.append(class_id)

# ---------------------------------------
# Aplicamos Non-Maximum Suppression para eliminar duplicados
# ---------------------------------------
# Esto evita que aparezcan múltiples cajas sobre el mismo objeto
boxes_np = np.array(boxes).tolist()
confidences_np = np.array(confidences).tolist()

# Aplicamos NMS
index = cv2.dnn.NMSBoxes(
    boxes_np, confidences_np,
    0.25,    # Umbral de confianza mínimo
    0.45     # Umbral de solapamiento (IOU) máximo permitido
).flatten()

# ---------------------------------------
# Dibujar las detecciones finales sobre la imagen
# ---------------------------------------
for i in index:
    x, y, w, h = boxes_np[i]  # Coordenadas de la caja
    bb_conf = int(confidences_np[i] * 100)  # Confianza como porcentaje

    classes_id = classes[i]                 # ID de clase
    class_name = labels[classes_id]         # Nombre de la clase

    text = f'{class_name}: {bb_conf}%'      # Texto a mostrar

    # Dibujamos la caja verde
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Fondo blanco para el texto
    cv2.rectangle(image, (x, y - 30), (x + w, y), (255, 255, 255), -1)

    # Escribimos el texto negro sobre el fondo blanco
    cv2.putText(image, text, (x, y - 10),
                cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 0, 0), 1)

# ---------------------------------------
# Mostrar la imagen con las detecciones finales
# ---------------------------------------
cv2.imshow('Detecciones YOLOv5', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
