# %% [markdowon]
# ### Curso de CNN con PyTorch
## Creando nuestro propio Clasificador de Imágenes 👀😷

from IPython.display import display, Markdown
aprenderas="""
En este lab aprenderás:

* [Pytorch](https://pytorch.org/)
* [Torchvision](https://pytorch.org/vision/stable/index.html)
* Descargar un dataset, prepararlo, entrenarlo, realizar finetuning y guardarlo.
display(Markdown(aprenderas))

### 1) Cargar el dataset 🤓
**Tenemos 2 formas:**
**1)** Cargar el dataset arrastrando el .zip hacia el notebook.
#!unzip "labeled-chest-xray-images.zip"
**2)** Guardar el .zip en Google Drive y luego darle permisos al notebook para que pueda accederlo."""
display(Markdown(aprenderas))
# %%
# --- 1.1 Importación de Librerías ---
import os
import time
import copy
import shutil
import zipfile
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# PyTorch y Torchvision
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, models, transforms

# --- 1.2 Configuración del Dispositivo (Hardware) ---
# Optimizamos el benchmark de CuDNN para acelerar la convergencia en GPUs
cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"🚀 El modelo se ejecutará en: {device}")

# %% [markdown]
# ### Capítulo 2: Extracción y Carga del Dataset 🤓
# Conectamos con Google Drive, forzamos una extracción limpia para evitar errores de sesiones pasadas y localizamos las carpetas de entrenamiento y validación.

# %%
# --- 2.1 Conexión y Extracción ---
from google.colab import drive
drive.mount('/content/drive')

data_zip = "/content/drive/MyDrive/Data/Curso de Redes Neuronales Convolucionales/labeled-chest-xray-images.zip"
target_dir = "/content/dataset_limpio"

# Limpieza preventiva del entorno local
if os.path.exists(target_dir):
    print("🧹 Borrando directorio antiguo incompleto...")
    shutil.rmtree(target_dir)

print("📦 Extrayendo el dataset desde Google Drive...")
os.makedirs(target_dir, exist_ok=True)
with zipfile.ZipFile(data_zip, 'r') as zip_ref:
    zip_ref.extractall(target_dir)
print("✅ Extracción completa.")

# --- 2.2 Localizador Automático de Carpetas ---
# Busca dinámicamente dónde quedaron 'train' y 'val'
data_dir = target_dir
for root, dirs, files in os.walk(target_dir):
    if "train" in dirs and "val" in dirs:
        data_dir = root
        break

print(f"🎯 Ruta raíz del Dataset: {data_dir}")

# %% [markdown]
# ### Capítulo 3: Pipeline de Datos (Transformaciones y Dataloaders) 👌
# Definimos el preprocesamiento de las imágenes. El Data Augmentation se aplica solo en la fase de entrenamiento para generalizar mejor el modelo.

# %%
# --- 3.1 Data Augmentation y Normalización ---
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # Normalización basada en ImageNet
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# --- 3.2 Construcción de Datasets y Dataloaders ---
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}

BATCH_SIZE = 32
NUM_WORKERS = 2

dataloaders = {
    x: torch.utils.data.DataLoader(
        image_datasets[x], 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False
    ) for x in ['train', 'val']
}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

print(f"🚀 Dataloaders listos. Clases detectadas: {class_names}")

# %% [markdown]
# ### Capítulo 4: Exploración Visual de los Datos 🔍
# Desnormalizamos un lote de imágenes para verificar visualmente que las transformaciones se estén aplicando correctamente.

# %%
def imshow(inp, title=None):
    """Convierte un tensor normalizado de PyTorch de vuelta a una imagen visible de Matplotlib."""
    plt.figure(figsize=(14, 8))
    
    # Reordenar ejes de (C, H, W) a (H, W, C)
    inp = inp.numpy().transpose((1, 2, 0))
    
    # Desnormalizar
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    
    plt.imshow(inp)
    plt.axis('off') # Vista médica limpia
    
    if title is not None:
        if isinstance(title, list):
            n_por_fila = 8
            filas_texto = [title[i:i + n_por_fila] for i in range(0, len(title), n_por_fila)]
            titulo_formateado = "\n".join([", ".join(fila) for fila in filas_texto])
            plt.title(titulo_formateado, fontsize=10, fontweight='bold', pad=12)
        else:
            plt.title(title, fontsize=12, fontweight='bold')
            
    plt.tight_layout()
    plt.show()

# Extraer y visualizar un batch
inputs, classes = next(iter(dataloaders['train']))
out = torchvision.utils.make_grid(inputs, padding=4)
imshow(out, title=[class_names[x] for x in classes])

# %% [markdown]
# ### Capítulo 5: Arquitectura de la CNN y Finetuning 😨
# Estructuramos nuestra red neuronal de forma secuencial, extrayendo características espaciales antes de aplanarlas para la clasificación.

# %%
# --- 5.1 Definición de la Red Secuencial ---
layers = [
    # Bloque de Extracción de Características: Convolución → Activación → Agrupamiento (Pooling)
    nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3, padding=1), 
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2), 

    # Bloque de Clasificación: Aplanamiento → Capa Lineal
    nn.Flatten(),
    nn.Linear(in_features=4 * 112 * 112, out_features=2) 
]

model_ft = nn.Sequential(*layers)
model_ft = model_ft.to(device)

# %% [markdown]
# ### Capítulo 6: Bucle de Entrenamiento 💪
# Definimos la función principal que iterará sobre los *epochs*, calculando gradientes en la fase `train` y evaluando la precisión en la fase `val`.

# %%
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Habilita Dropout y BatchNorm
            else:
                model.eval()   # Congela la red para inferencia pura

            running_loss = 0.0
            running_corrects = 0

            # Iteración sobre los mini-batches
            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad() # Reinicia gradientes

                # Forward pass (Trackea el historial solo en train)
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward pass y optimización
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Guardar el mejor modelo encontrado
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Entrenamiento completado en {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Mejor Accuracy en Validación: {best_acc:4f}')

    model.load_state_dict(best_model_wts)
    return model
# %% [markdown]
# ### Capítulo 7: Función de Visualización de 6 Imágenes de Validación🔍
def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

# %% [markdown]
# ### Capítulo 8: Ejecución del Entrenamiento y Guardado 💾
# Instanciamos la función de pérdida (CrossEntropy), el optimizador (SGD) y el planificador de tasa de aprendizaje.

# %%
# --- 8.1 Configuración de Hiperparámetros ---
criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

# --- 8.2 Lanzamiento del Entrenamiento ---
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=5)

# --- 8.3 Persistencia del Modelo ---
torch.save(model_ft, "model.pth")
print("💾 Modelo guardado exitosamente como 'model.pth'")
# %% [markdown]
# ### Capítulo 9: Visualización de 6 Imágenes de Validación
visualize_model(model_ft)
# %% [markdown]
# ### Capítulo 10: Inferencia en Producción 🤙
# Simulamos cómo se utilizaría el modelo ya entrenado para diagnosticar nuevas radiografías individuales.

# %%
# --- 10.1 Carga del Modelo Entrenado ---
mi_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mi_modelo = torch.load("model.pth", map_location=mi_device, weights_only=False)
mi_modelo.eval() # Modo inferencia
mi_modelo.to(mi_device)

# --- 10.2 Función Desacoplada de Predicción ---
def predict_image(image_path, model, class_names, device):
    """
    Recibe la ruta física de una imagen, aplica el preprocesamiento exacto 
    usado en entrenamiento y retorna el diagnóstico de la red.
    """
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert("RGB")
    # unsqueeze(0) simula un batch_size de 1
    image_tensor = preprocess(image).unsqueeze(0).to(device)  

    with torch.no_grad():
        outputs = model(image_tensor)
        _, preds = torch.max(outputs, 1)

    return class_names[preds[0].item()]

# %% [markdown]
# ### Ejemplos de Inferencia
# *Nota: Modifica las rutas de prueba según tus archivos locales.*

# %%
# Ejemplo 1
path_image_1 = f"{data_dir}/val/NORMAL/NORMAL-11419-0001.jpeg"
try:
    pred_1 = predict_image(path_image_1, mi_modelo, class_names, mi_device)
    print(f"🩺 Imagen 1 | Diagnóstico predicho: {pred_1}")
except FileNotFoundError:
    print(f"⚠️ No se encontró la imagen de prueba en: {path_image_1}")

# Ejemplo 2 (Asegúrate de tener un archivo bacteria.jpeg en /content)
path_image_2 = "/content/bacteria.jpeg"
try:
    pred_2 = predict_image(path_image_2, mi_modelo, class_names, mi_device)
    print(f"🩺 Imagen 2 | Diagnóstico predicho: {pred_2}")
except FileNotFoundError:
    print(f"⚠️ No se encontró la imagen de prueba en: {path_image_2}")
Ejemplos="""
**Ejemplos de uso**
class_names = ['NORMAL', 'PNEUMONIA']
path_image = "/content/chest_xray/val/NORMAL/NORMAL-11419-0001.jpeg"

predicted_class = predict_image(path_image, mi_modelo, class_names, mi_device)
print(f"La clase predicha es: {predicted_class}")
path_image = "/content/bacteria.jpeg"

predicted_class = predict_image(path_image, mi_modelo, class_names, mi_device)
print(f"La clase predicha es: {predicted_class}")
### 7) Conclusiones

- Aprender sobre los distintos objetos y métodos que nos ofrece Pytorch / Torchvision.

- Realizar el proceso completo de clasificación de imágenes con Pytorch.

- Aprender tips sobre implementación con el uso de la GPU."""
display(Markdown(Ejemplos))
