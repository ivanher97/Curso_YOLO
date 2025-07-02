# --- Código extraído del notebook ---

import os # para interactuar con el SO
from glob import glob # sirve para buscar archivos con patrones, como .xml
import pandas as pd # libreria para manejar datos tabulates y se renombra como pd en el codigo
from functools import reduce # aplica una funcion acumulativa Ejemplo : reduce(lambda x, y: x + y, [1, 2, 3, 4]) --> # Resultado final = 10
from xml.etree import ElementTree as et # sirve para leer y analizar archivos XML

# Cargar todos los archivos .xml de la carpeta y almacenarlos en una lista
xml_list = glob('./data_images/*.xml')

# Limpiar los nombres de los archivos --> replace \\ con /
xml_list = list(map(lambda x: x.replace('\\', '/'), xml_list))

xml_list

# Esta funcion se la aplicamos a todas las imagenes que tenemos en la carpeta
def extract_text(filename):
    # Leer el archivo xml, de cada uno extraemos: filename, size(ancho,alto), object(name, xmin, xmax, ymin, ymax)
    tree = et.parse(filename)
    root = tree.getroot()
    
    # Sacar nombre del archivo
    img_name = root.find('filename').text
    
    # Sacar ancho y alto de la imagen
    img_w = root.find('size').find('width').text
    img_h = root.find('size').find('height').text
    
    # Sacar del objeto los datos, cada imagen puede tener mas de un oobjeto
    obj = root.findall('object')

    datos =[]
    for o in obj:
        name = o.find('name').text
        
        box = o.find('bndbox')
        xmin = box.find('xmin').text
        xmax = box.find('xmax').text
        ymin = box.find('ymin').text
        ymax = box.find('ymax').text
        datos.append([img_name, img_w, img_h, name, xmin, xmax, ymin, ymax])
    
    return datos

parser_all = list(map(extract_text, xml_list))

data = reduce(lambda x, y : x + y, parser_all)

# Construimos una tabla con los datos
df = pd.DataFrame(data, columns = ['filename', 'img_w', 'img_h', 'name', 'xmin', 'xmax', 'ymin', 'ymax' ])
df.head()

# Filas x columnas de la tabla
df.shape

# Contamos los objetos que hay de cada tipo en la tabla
df['name'].value_counts()

df.info()

# Modificamos el tipo de variable para los datos que queremos manejar
cols = ['img_w', 'img_h', 'xmin', 'xmax', 'ymin', 'ymax']
df[cols] = df[cols].astype(int)
df.info()

 # center_x y center_y
df['center_x'] = ((df['xmin'] + df['xmax']) / 2) / df['img_w']
df['center_y'] = ((df['ymin'] + df['ymax']) / 2) / df['img_h']
# ancho y alto
df['w'] = (df['xmax'] - df['xmin']) / df['img_w']
df['h'] = (df['ymax'] - df['ymin']) / df['img_h']

df.head()

images = df['filename'].unique()
print(len(images))

# Vamos a dividir las imagenes en un 80% entrenamiento y un 20% test
img_df = pd.DataFrame(images, columns = ['filename'])
# Mezcla y elige aleatoriamente el 80% de las imagenes
img_train = tuple(img_df.sample(frac = 0.8)['filename'])

# Obtenemos el 20% de las imagenes restantes
img_test = tuple(img_df.query(f'filename not in {img_train}')['filename'])

# Creamos copias explicitas de los dataframes para evitar problemas mas adelante
train_df = df.query(f'filename in {img_train}').copy()
test_df = df.query(f'filename in {img_test}').copy()

train_df.head()

# Convertimos los tipos de objetos en id, ya que necesitamos informacion numerica no texto
def labels_encoding(x):
    labels = {'person':0, 'car':1, 'chair':2, 'bottle':3, 'pottedplant':4, 'bird':5, 'dog':6, 'sofa':7, 'bicycle':8, 'horse':9, 'boat':10,
              'motorbike':11, 'cat':12, 'tvmonitor':13, 'cow':14, 'sheep':15, 'aeroplane':16, 'train':17, 'diningtable':18, 'bus':19}
    return labels[x]

train_df['id'] = train_df['name'].apply(labels_encoding)
test_df['id'] = test_df['name'].apply(labels_encoding)

test_df.head(10)

# Guardar imagenes y labels en texto
import os
from shutil import move

train_folder = 'data_images/train'
test_folder = 'data_images/test'

os.mkdir(train_folder)
os.mkdir(test_folder)

cols = ['filename', 'id', 'center_x', 'center_y', 'w', 'h']
group_obj_tr = train_df[cols].groupby('filename')
group_obj_ts = test_df[cols].groupby('filename')

# Guardar las imagenes en las carpetas correspondientes y los respectivos .txt con la informacion
def save_data(filename, folder_path, group_obj):
    # Mover imagenes
    src = os.path.join('data_images', filename)
    dts = os.path.join(folder_path, filename)
    move(src, dts)

    # Guardar la informacion de los labels
    text_filename = os.path.join(folder_path,
                                 os.path.splitext(filename)[0] + '.txt')
    group_obj.get_group(filename).set_index('filename').to_csv(text_filename, sep = ' ', index=False, header=False)

filename_series = pd.Series(group_obj_tr.groups.keys())

filename_series.apply(save_data, args = (train_folder, group_obj_tr))
filename_series_test = pd.Series(group_obj_ts.groups.keys())
filename_series_test.apply(save_data, args = (test_folder, group_obj_ts))


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



