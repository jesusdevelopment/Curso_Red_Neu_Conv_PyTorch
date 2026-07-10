 # %% [markdown]
## Importación de Bibliotecas y Configuración de Entorno
!pip install imagehash
!pip install --upgrade keras  # Aseguramos tener Keras 3+

import os

# 🚨 CRÍTICO PARA KERAS 3 🚨
# La variable de entorno DEBE definirse ANTES de cualquier import de ML
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
import imagehash  # Para la detección de duplicados

# Importaciones de Machine Learning (seguras bajo backend Torch)
import keras
from keras import layers
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, TensorBoard
from keras.applications.densenet import preprocess_input

import tensorflow as tf  # Solo para tf.data (Data Pipelines)
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize

# %% [markdown]
## 1. Conexión con Google Drive y Carga de Datos

if not os.path.exists('/content/drive'):
    drive.mount('/content/drive')

# Configuración de rutas
ruta_rar = "/content/drive/MyDrive/Data/Curso Prof TensorFlow/dataset_extraido.rar"
extract_dir = "/content/dataset_trabajo"

# Proceso de extracción con limpieza previa
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

# %% [markdown]
## 5. EDA: Balanceo, Limpieza y Análisis Dimensional

print("\n--- INICIANDO EDA Y LIMPIEZA DE DATOS ---")

# 5.1 Balanceo de Clases
sets = ['Training', 'Testing']
MIS_CLASES_BRAIN = ['glioma', 'meningioma', 'notumor', 'pituitary']
stats = []

for dataset_type in sets:
    set_path = os.path.join(extract_dir, dataset_type)
    if not os.path.exists(set_path):
        print(f"⚠️ Alerta: No se encontró la carpeta: {set_path}")
        continue
    for label in MIS_CLASES_BRAIN:
        path = os.path.join(set_path, label)
        if os.path.exists(path):
            count = len(os.listdir(path))
            stats.append({'label': label, 'count': count, 'dataset': dataset_type})
        else:
            print(f"❌ Carpeta de clase no encontrada: {path}")

df_stats = pd.DataFrame(stats)

if df_stats.empty:
    print("FATAL: El DataFrame está vacío. Revisa las rutas de 'extract_dir'.")
else:
    plt.figure(figsize=(12, 6))
    sns.barplot(x='label', y='count', hue='dataset', data=df_stats)
    plt.title('Distribución de Clases: Glioma, Meningioma, No Tumor y Pituitaria')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

# 5.2 Detección de archivos Corruptos
def check_images(directory):
    print("\nBuscando imágenes corruptas...")
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png')):
                path = os.path.join(root, file)
                try:
                    img = Image.open(path)
                    img.verify() 
                except (IOError, SyntaxError):
                    print(f'Archivo corrupto eliminado: {path}')
                    os.remove(path)

check_images(extract_dir)

# 5.3 Detección de Archivos Duplicados (Hashing)
def eliminar_duplicados_visuales(directorio):
    print(f"\nBuscando duplicados visuales en: {directorio}")
    hashes_vistos = {}
    duplicados_eliminados = 0
    
    for root, dirs, files in os.walk(directorio):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                path = os.path.join(root, file)
                try:
                    with Image.open(path) as img:
                        v_hash = imagehash.dhash(img)
                    if v_hash in hashes_vistos:
                        print(f"Eliminando duplicado visual: {path}")
                        os.remove(path)
                        duplicados_eliminados += 1
                    else:
                        hashes_vistos[v_hash] = path
                except Exception as e:
                    print(f"No se pudo procesar {file}: {e}")
                
    print(f"¡Limpieza terminada! Se eliminaron {duplicados_eliminados} archivos duplicados.")

eliminar_duplicados_visuales(train_dir)
eliminar_duplicados_visuales(test_dir)

# 5.4 Análisis Dimensional (Tamaños y Proporciones)
print("\nAnalizando dimensiones de las imágenes...")
sets_to_analyze = ['Training', 'Testing']
colors = {'Training': 'blue', 'Testing': 'orange'}
plt.figure(figsize=(10, 6))

widths, heights = [], [] # Para el resumen numérico global

for dataset_type in sets_to_analyze:
    w_temp, h_temp = [], []
    current_path = os.path.join(extract_dir, dataset_type)
    for root, dirs, files in os.walk(current_path):
        for file in files:
            if file.endswith('.jpg'):
                try:
                    with Image.open(os.path.join(root, file)) as im:
                        w, h = im.size
                        w_temp.append(w)
                        h_temp.append(h)
                        widths.append(w)
                        heights.append(h)
                except Exception as e:
                    pass
    plt.scatter(w_temp, h_temp, alpha=0.3, label=dataset_type, color=colors[dataset_type])

plt.xlabel('Ancho (píxeles)')
plt.ylabel('Alto (píxeles)')
plt.title('Comparación de Dimensiones: Entrenamiento vs. Test')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

total_train = sum([len(files) for r, d, files in os.walk(train_dir)])
total_test = sum([len(files) for r, d, files in os.walk(test_dir)])

print("--- RESUMEN DEL DATASET ---")
if widths and heights:
    print(f"Límite máximo de dimensiones: {max(widths)} (ancho) x {max(heights)} (alto)")
    print(f"Límite mínimo de dimensiones: {min(widths)} (ancho) x {min(heights)} (alto)")
print(f"Total de imágenes para entrenamiento: {total_train}")
print(f"Total de imágenes para prueba (Test): {total_test}")
print(f"Proporción de entrenamiento: {total_train / (total_train + total_test):.2f}")
print(f"Proporción de prueba: {total_test / (total_train + total_test):.2f}")

# 5.5 Análisis de Intensidad de Píxeles
def analizar_intensidades(extract_dir, sets=['Training', 'Testing'], sample_size=300):
    plt.figure(figsize=(12, 6))
    colores = {'Training': 'blue', 'Testing': 'orange'}
    print("\nIniciando análisis de intensidades de píxel...")

    for dataset_type in sets:
        dataset_path = os.path.join(extract_dir, dataset_type)
        all_files = []
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if file.endswith('.jpg'):
                    all_files.append(os.path.join(root, file))
        
        sampled_files = random.sample(all_files, min(sample_size, len(all_files)))
        hist_acumulado = np.zeros((256, 1))
        
        for img_path in sampled_files:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                hist = cv2.calcHist([img], [0], None, [256], [0, 256])
                hist_acumulado += hist
                
        hist_acumulado /= hist_acumulado.sum()
        plt.plot(hist_acumulado, color=colores[dataset_type], label=f'{dataset_type} (n={len(sampled_files)})', alpha=0.8)

    plt.title('Distribución de Intensidades de Píxel: Training vs. Testing')
    plt.xlabel('Intensidad (0 = Negro absoluto, 255 = Blanco absoluto)')
    plt.ylabel('Frecuencia Relativa')
    plt.xlim([0, 256])
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

analizar_intensidades(extract_dir)

# %% [markdown]
### Train y Test Data Pipelines

IMG_SIZE = (224, 224)
BATCH_SIZE = 16

# tf.data.Dataset para PyTorch nativo en Keras 3
train_ds = keras.utils.image_dataset_from_directory(
    train_dir,
    validation_split=0.2,
    class_names=MIS_CLASES_BRAIN,
    color_mode='rgb',
    subset="training",
    seed=42,
    image_size=IMG_SIZE, 
    batch_size=BATCH_SIZE
)

val_ds = keras.utils.image_dataset_from_directory(
    train_dir,
    validation_split=0.2,
    class_names=MIS_CLASES_BRAIN,
    color_mode='rgb',
    subset="validation",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

test_ds = keras.utils.image_dataset_from_directory(
    test_dir,
    class_names=MIS_CLASES_BRAIN,
    color_mode='rgb',
    shuffle=False,      
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# %% [markdown]
## Visualización de 5 muestras iniciales
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

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2)
], name="data_augmentation")

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE) 
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE) 
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE) 

# %% [markdown]
# Creación del Modelo con API Funcional (DenseNet121)

base_model = keras.applications.DenseNet121(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

# Construcción Funcional Limpia
inputs = keras.Input(shape=IMG_SIZE + (3,))
x = data_augmentation(inputs)
x = layers.Lambda(preprocess_input, name='densenet_preprocessing')(x)
x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(4, activation='softmax', name='clasificador_tumores')(x)

model = keras.Model(inputs, outputs, name="DenseNet_Brain_MRI")

# Compilación
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

model.summary() 

# %% [markdown]
# Entrenamiento

epochs = 20
early_stopping = EarlyStopping(
    monitor='val_loss', 
    patience=5, 
    restore_best_weights=True
)

log_dir = os.path.join("logs", "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,     
    write_graph=True,     
    update_freq='epoch',
    profile_batch=0       
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=[early_stopping, tensorboard_callback]
)

# %% [markdown]
# Evaluación y Reporte

loss, accuracy = model.evaluate(test_ds)
print(f"\n✅ Precisión en Test: {accuracy*100:.2f}%")

y_true = np.concatenate([y for x, y in test_ds], axis=0)
y_pred_probs = model.predict(test_ds)
y_pred = np.argmax(y_pred_probs, axis=-1)

print("\n--- REPORTE DE CLASIFICACIÓN ---")
print(classification_report(y_true, y_pred, target_names=MIS_CLASES_BRAIN))

# %% [markdown]
# ## loss vs val loss & acc vs val_acc

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Entrenamiento')
plt.plot(history.history['val_loss'], label='Validación')
plt.title('Pérdida (Loss)')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Validación')
plt.title('Precisión (Accuracy)')
plt.legend()
plt.show()

# %% [markdown]
# ## Curva ROC

def plot_roc_curve(y_true, y_pred_probs, target_names):
    n_classes = len(target_names)
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(10, 8))
    colors = ['blue', 'red', 'green', 'orange']
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC {target_names[i]} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos (FPR)')
    plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
    plt.title('Curva ROC Multiclase - Tumores Cerebrales')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.show()

plot_roc_curve(y_true, y_pred_probs, MIS_CLASES_BRAIN)

# %% [markdown]
## 6.1 Visualización de TensorBoard en Colab
'''
!npm install -g localtunnel
# Liberar puerto si estaba ocupado por ejecuciones previas
!pkill -f tensorboard 
subprocess.Popen(["tensorboard", "--logdir", "logs/fit", "--port", "6006"])

print("\n============== INSTRUCCIONES ==============")
print("1. Copia la IP numérica que aparece en la siguiente línea:")
!curl ipv4.icanhazip.com

print("\n2. Haz clic en el enlace generado por LocalTunnel y pega la IP:")
!lt --port 6006
'''