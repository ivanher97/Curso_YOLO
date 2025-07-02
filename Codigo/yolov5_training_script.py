import os

# Cambia al directorio donde está tu entrenamiento de YOLO
os.chdir('/content/drive/MyDrive/YOLO_training')

# Lista el contenido del directorio actual
os.system('ls')

# Cambia al subdirectorio de YOLOv5
os.chdir('yolov5')

# Lista archivos dentro del directorio yolov5
os.system('ls')

# Instalación de dependencias necesarias para entrenar YOLOv5
# os.system('pip install -r requirements.txt')

# Entrenamiento del modelo YOLOv5 con imágenes de 640x640, batch size 8 y 50 épocas, usando pesos preentrenados 'yolov5s.pt'.
# El archivo 'data.yaml' define las rutas y clases del dataset. Los resultados se guardarán bajo 'runs/train/Model'.
# Nota: Corregir '--eproch' a '--epochs'.

# os.system('python train.py --data data.yaml --weights yolov5s.pt --img 640 --batch-size 8 --name Model --epochs 50')

# Continúa el entrenamiento de un modelo previamente entrenado usando los pesos guardados en 'best.pt'.
# Esto permite hacer fine-tuning sobre el mismo dataset u otro similar.
# Se entrena por 50 épocas más, usando imágenes de 640x640 y un batch size de 8.

# os.system('python train.py --data data.yaml --weights runs/train/Model/weights/best.pt --img 640 --batch-size 8 --name Model --epochs 50')

# Exporta el modelo YOLOv5 entrenado al formato ONNX (Open Neural Network Exchange), 
# permitiendo usarlo en producción en otras plataformas como móviles, servidores, o dispositivos embebidos.

# os.system('python export.py --weights runs/train/Model/weights/best.pt --include onnx --simplify --opset 12')
