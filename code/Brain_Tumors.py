# %% [markdown]
## Importación de Bibliotecas
!pip install imagehash
!pip install --upgrade keras  # Aseguramos tener Keras 3+

import os

# 🚨 CRÍTICO PARA KERAS 3 🚨
# Debemos definir la variable de entorno ANTES de importar Keras
os.environ["KERAS_BACKEND"] = "torch"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile
from PIL import Image
import glob   
from io import BytesIO
import hashlib
from google.colab import drive
import datetime
import cv2
import random
import subprocess

import torch  # El nuevo motor subyacente
import tensorflow as tf  # Lo mantenemos SOLO para cargar los datos eficientemente (tf.data)
import keras
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, TensorBoard

# %% [markdown]
## 1. Conexión con Google Drive y Carga de Datos Limpios

if not os.path.exists('/content/drive'):
    drive.mount('/content/drive')

# 2. Configuración de rutas
ruta_rar = "/content/drive/MyDrive/Data/Curso Prof TensorFlow/dataset_extraido.rar"
extract_dir = "/content/dataset_trabajo"

# 3. Proceso de extracción con limpieza previa
if os.path.exists(ruta_rar):
    if os.path.exists(extract_dir):
        !rm -rf "{extract_dir}"
    
    os.makedirs(extract_dir, exist_ok=True)
    
    print("📦 Extrayendo dataset... Por favor, espera a que aparezca el mensaje de éxito.")
    !unrar x -o+ -idq "{ruta_rar}" "{extract_dir}/"
    
    print("🔍 Buscando carpetas de datos...")
    hallazgos = glob.glob(os.path.join(extract_dir, "**/Training"), recursive=True)
    
    if hallazgos:
        base_real = os.path.dirname(hallazgos[0])
        train_dir = os.path.join(base_real, 'Training')
        test_dir = os.path.join(base_real, 'Testing')
        
        print(f"✅ ¡Éxito! Carpetas encontradas en: {base_real}")
        print(f"📍 Train: {train_dir}")
        print(f"📍 Test:  {test_dir}")
        
        extract_dir = base_real 
    else:
        print("❌ ERROR: El .rar se extrajo pero no contiene una carpeta llamada 'Training'.")
        !ls -R "{extract_dir}" | head -n 10
else:
    print(f"❌ ERROR: No se encontró el archivo en Drive: {ruta_rar}")

train_dir = os.path.join(extract_dir, 'Training')
test_dir = os.path.join(extract_dir, 'Testing')

# %% [markdown]
## 5. EDA: Balanceo de Clases, Detección de Corruptos y Duplicados, Análisis Dimensional

# [NOTA: Todo el bloque de EDA (matplotlib, seaborn, PIL, cv2, imagehash) permanece EXACTAMENTE IGUAL, 
# ya que es Python puro y no depende del framework de Deep Learning. 
# Se omite aquí por brevedad visual, pero tu código original funciona perfectamente].

# %% [markdown]
### Train y Test Data Pipelines
train_dir = os.path.join(extract_dir, 'Training')
test_dir = os.path.join(extract_dir, 'Testing')

IMG_SIZE = (224, 224)
BATCH_SIZE = 16

# Keras 3 es compatible nativamente con tf.data.Dataset para PyTorch
train_ds = keras.utils.image_dataset_from_directory(
    extract_dir + '/Training',
    validation_split=0.2,
    class_names=['glioma', 'meningioma', 'notumor', 'pituitary'],
    color_mode='rgb',
    subset="training",
    seed=42,
    image_size=IMG_SIZE, 
    batch_size=BATCH_SIZE
)

val_ds = keras.utils.image_dataset_from_directory(
    extract_dir + '/Training',
    validation_split=0.2,
    class_names=['glioma', 'meningioma', 'notumor', 'pituitary'],
    color_mode='rgb',
    subset="validation",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

test_ds = keras.utils.image_dataset_from_directory(
    extract_dir + '/Testing',
    class_names=['glioma', 'meningioma', 'notumor', 'pituitary'],
    color_mode='rgb',
    shuffle=False,      
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# %% [markdown]
## Visualización de 5 muestras

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1): 
    for i in range(5): 
        ax = plt.subplot(1, 5, i + 1)
        plt.imshow(images[i].numpy().astype("uint8")) 
        plt.title(train_ds.class_names[labels[i]]) 
        plt.axis("off") 
plt.show()

# %% [markdown]
### Preprocesamiento y Configuración de Rendimiento
# Reemplazamos tf.keras por keras puro
data_augmentation = keras.Sequential([
    keras.layers.RandomFlip("horizontal_and_vertical"),
    keras.layers.RandomRotation(0.2),
    keras.layers.RandomZoom(0.2),
    keras.layers.RandomContrast(0.2)
])

# Mantenemos tf.data para el pipeline eficiente
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE) 
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE) 
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE) 

# %% [markdown]
# Creación del modelo base (MobileNetV2) con Keras 3

base_model = keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False

inputs = keras.Input(shape=(224, 224, 3))
x = data_augmentation(inputs)  
x = keras.applications.mobilenet_v2.preprocess_input(x) 
x = base_model(x, training=False) 
x = keras.layers.GlobalAveragePooling2D()(x) 
x = keras.layers.Dropout(0.2)(x) 
outputs = keras.layers.Dense(4, activation='softmax')(x) 

model = keras.Model(inputs, outputs)

# Compilación del modelo (Keras 3 se encarga de usar PyTorch optimizers)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

model.summary() 

epochs = 20

early_stopping = EarlyStopping(
    monitor='val_loss', 
    patience=3, 
    restore_best_weights=True
)

log_dir = os.path.join("logs", "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

tensorboard_callback = TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,     
    write_graph=True,     
    write_images=True,    
    update_freq='epoch',
    profile_batch=0       
)

# El entrenamiento ocurre usando el motor de PyTorch
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=[early_stopping, tensorboard_callback]
)

# %%
# Evaluación
loss, accuracy = model.evaluate(test_ds)
print(f"Precisión en Test: {accuracy*100:.2f}%")

from sklearn.metrics import classification_report

MIS_CLASES_BRAIN = ['glioma', 'meningioma', 'notumor', 'pituitary']
y_true = np.concatenate([y for x, y in test_ds], axis=0)
y_pred = np.argmax(model.predict(test_ds), axis=-1)

print("\n--- REPORTE DE CLASIFICACIÓN ---")
print(classification_report(y_true, y_pred, target_names=MIS_CLASES_BRAIN))

# [El bloque de visualización de métricas plot_loss_accuracy se mantiene igual]

# %% [markdown]
## 6.1 Visualización de TensorBoard

!npm install -g localtunnel
subprocess.Popen(["tensorboard", "--logdir", "logs/fit", "--port", "6006"])

print("\n============== INSTRUCCIONES ==============")
print("1. Copia esta dirección IP completa:")
!curl https://localtunnel.me/tunnelme

print("\n2. Haz clic en el enlace de abajo para abrir TensorBoard:")
!lt --port 6006

# %%
# ❌ ATENCIÓN: El servicio de TensorBoard.dev fue cerrado por Google.
# La siguiente línea ha sido comentada porque ya no funciona y generará un error.
# !tensorboard dev upload --logdir ./logs --name "Proyecto prueba" --description "Test development results" --one_shot