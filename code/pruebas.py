# %%

import os
# 1. CONFIGURACIÓN DEL BACKEND (Debe ir antes de importar Keras)
os.environ["KERAS_BACKEND"] = "torch"

import numpy as np
import tensorflow as tf
import keras
from keras import layers

# Configuración de hiperparámetros y constantes de hardware
BATCH_SIZE = 16
IMG_SIZE = 224

# Activamos la constante adaptativa de hardware
AUTOTUNE = tf.data.AUTOTUNE

print(f"Ecosistema Keras 3 corriendo sobre el backend de: {keras.config.backend()}\n")


# =====================================================================
# 1. SIMULACIÓN DE RUTAS DE ARCHIVOS EN DISCO Y ETIQUETAS (Train, Val, Test)
# =====================================================================
# Generamos metadatos ficticios separados para cada conjunto
rutas_train = [f"/sistema/train_imagen_{i}.png" for i in range(64)]
etiquetas_train = np.random.randint(0, 2, size=(64, 1)).astype(np.float32)

rutas_val = [f"/sistema/val_imagen_{i}.png" for i in range(16)]
etiquetas_val = np.random.randint(0, 2, size=(16, 1)).astype(np.float32)

rutas_test = [f"/sistema/test_imagen_{i}.png" for i in range(16)]
etiquetas_test = np.random.randint(0, 2, size=(16, 1)).astype(np.float32)


# =====================================================================
# 2. FUNCIÓN LÓGICA DE CARGA EN CPU
# =====================================================================
def simular_lectura_disco(ruta, etiqueta):
    """
    Simula la lectura física del almacenamiento (ej. OpenCV, PIL o PyDicom).
    Se ejecuta en la CPU de forma multihilo gracias a la infraestructura de tf.data.
    """
    def _leer():
        # Aquí simularías la apertura real del archivo basado en el parámetro 'ruta'
        return np.random.rand(IMG_SIZE, IMG_SIZE, 3).astype(np.float32)
    
    imagen_cargada = tf.py_function(_leer, [], tf.float32)
    imagen_cargada.set_shape([IMG_SIZE, IMG_SIZE, 3])
    return imagen_cargada, etiqueta


# =====================================================================
# 3. CONSTRUCCIÓN DE PIPELINES ULTRA-OPTIMIZADOS CON AUTOTUNE
# =====================================================================

# --- DATASET DE ENTRENAMIENTO ---
# Aplica el orden perfecto: Map -> Cache -> Shuffle -> Batch -> Prefetch
train_ds = tf.data.Dataset.from_tensor_slices((rutas_train, etiquetas_train))
train_ds = (
    train_ds
    .map(simular_lectura_disco, num_parallel_calls=AUTOTUNE)
    .cache()
    .shuffle(buffer_size=64)  # Ajustado al tamaño de este set para aleatoriedad perfecta
    .batch(BATCH_SIZE)
    .prefetch(buffer_size=AUTOTUNE)
)

# --- DATASET DE VALIDACIÓN ---
# Mismo flujo pero SIN SHUFFLE para evitar inestabilidades visuales en las métricas
val_ds = tf.data.Dataset.from_tensor_slices((rutas_val, etiquetas_val))
val_ds = (
    val_ds
    .map(simular_lectura_disco, num_parallel_calls=AUTOTUNE)
    .cache()
    .batch(BATCH_SIZE)
    .prefetch(buffer_size=AUTOTUNE)
)

# --- DATASET DE TESTING ---
# Mismo flujo estricto SIN SHUFFLE para no romper el mapeo de la Matriz de Confusión
test_ds = tf.data.Dataset.from_tensor_slices((rutas_test, etiquetas_test))
test_ds = (
    test_ds
    .map(simular_lectura_disco, num_parallel_calls=AUTOTUNE)
    .cache()
    .batch(BATCH_SIZE)
    .prefetch(buffer_size=AUTOTUNE)
)


# =====================================================================
# 4. RED NEURONAL DE VERIFICACIÓN (Channels Last por defecto)
# =====================================================================
modelo_rendimiento = keras.Sequential([
    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.Conv2D(16, kernel_size=3, activation="relu"),
    layers.MaxPooling2D(pool_size=2),
    layers.Flatten(),
    layers.Dense(1, activation="sigmoid")
])

modelo_rendimiento.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=keras.losses.BinaryCrossentropy(),
    metrics=[keras.metrics.BinaryAccuracy(name="accuracy")]
)

modelo_rendimiento.summary()


# =====================================================================
# 5. EJECUCIÓN DEL ENTRENAMIENTO Y EVALUACIÓN FINAL
# =====================================================================
print("\n--- Iniciando Entrenamiento Optimizado con AUTOTUNE ---")
# Pasamos train_ds y val_ds de forma directa. Las etiquetas ya van integradas.

modelo_rendimiento.fit(
    train_ds,
    validation_data=val_ds,
    epochs=3
)

print("\n--- Iniciando Evaluación Final con el Set de Testing (Orden Conservado) ---")
metricas_test = modelo_rendimiento.evaluate(test_ds)
