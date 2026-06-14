# %% [markdown]
# # Pipeline de Clasificación de Imágenes y Evaluación de Modelos
# Este notebook contiene el flujo completo para el procesamiento de imágenes médicas, 
# entrenamiento de arquitecturas CNN (desde cero y Transfer Learning), evaluación exhaustiva 
# de métricas de clasificación y explicabilidad de modelos mediante Grad-CAM.

# %%
import os
import zipfile
import shutil
import time
import copy
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, models, transforms
import torchvision.transforms.functional as TF
from torchvision.models import ResNet50_Weights

from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from IPython.display import display, Markdown

# %% [markdown]
# ### 2.2) Configuración del Entorno y Hardware 👀

# %%
cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"🚀 Operando en el dispositivo: {device}")

# %% [markdown]
# ### 2.3) Data Augmentation y Normalización 📷 📸

# %%
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# %% [markdown]
# ### 2.4) Extracción de Dataset y Configuración de Dataloaders 🔍

# %%
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

# Localizador Automático de Carpetas
data_dir = target_dir
for root, dirs, files in os.walk(target_dir):
    if "train" in dirs and "val" in dirs:
        data_dir = root
        break

print(f"🎯 Ruta raíz del Dataset: {data_dir}")

# Construcción de las instancias de datos
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}

# Optimizamos con pin_memory=True si usamos CUDA para acelerar la transferencia al device
dataloaders = {
    x: torch.utils.data.DataLoader(
        image_datasets[x], 
        batch_size=4, 
        shuffle=True, 
        num_workers=2, 
        pin_memory=True if torch.cuda.is_available() else False
    ) for x in ['train', 'val']
}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
print(f"Categorías del Dataset: {class_names}")

# %% [markdown]
# ### 2.5) Funciones de Inspección Visual de Datos 🖼️

# %%
def imshow(inp, title=None):
    """Desnormaliza y muestra un tensor de imagen."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

# Obtener un batch de datos de entrenamiento para auditar de forma visual
inputs, classes = next(iter(dataloaders['train']))
out = torchvision.utils.make_grid(inputs)
imshow(out, title=[class_names[x] for x in classes])

# %% [markdown]
# ---
# ## 3) Definición de Arquitecturas Candidatas 🧠
# *Nota: Ejecuta la celda de la arquitectura que desees entrenar. Cada una sobrescribirá la variable `model_ft`.*

# %% [markdown]
# #### Opción A: CNN Personalizada Básica (CNN 1)

# %%
ARQUITECTURA_ACTIVA = 'cnn1'  # <--- ¡Aquí controlas todo!

# %% [markdown]
# #### Configuración del Modelo según la selección

# %%
if ARQUITECTURA_ACTIVA == 'cnn1':
    layers = [nn.Conv2d(3, 4, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2), nn.Flatten(), nn.Linear(4 * 112 * 112, 2)]
    model_ft = nn.Sequential(*layers).to(device)
    
elif ARQUITECTURA_ACTIVA == 'cnn2':
    layers = [nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2), nn.Flatten(), nn.Linear(32 * 56 * 56, 128), nn.ReLU(), nn.Linear(128, 2)]
    model_ft = nn.Sequential(*layers).to(device)
    
elif ARQUITECTURA_ACTIVA == 'resnet':
    model_ft = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, len(class_names))
    model_ft = model_ft.to(device)

print(f"🎯 Arquitectura cargada en memoria lista para entrenar: {ARQUITECTURA_ACTIVA.upper()}")

# %% [markdown]
# ---
# ## 4) Motores de Entrenamiento y Evaluación del Sistema ⚙️

# %%
def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, device, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    val_preds, val_labels, val_probs = [], [], []
    train_preds, train_labels, train_probs = [], [], []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}\n' + '-' * 10)

        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()

            running_loss = 0.0
            running_corrects = 0

            preds_epoch, labels_epoch, probs_epoch = [], [], []

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                # Optimización: cálculo directo de aciertos sin fugas de memoria GPU
                running_corrects += (preds == labels).sum().item()

                # Recolección segura de predicciones y probabilidades (vía Softmax)
                preds_epoch.extend(preds.detach().cpu().numpy())
                labels_epoch.extend(labels.detach().cpu().numpy())
                probs_epoch.extend(torch.softmax(outputs, dim=1).detach().cpu().numpy())

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Almacenamiento persistente de datos del mejor modelo
            if phase == 'train':
                train_preds, train_labels, train_probs = preds_epoch, labels_epoch, probs_epoch
            elif phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                val_preds, val_labels, val_probs = preds_epoch, labels_epoch, probs_epoch

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')

    model.load_state_dict(best_model_wts)
    return model, (train_preds, train_labels, train_probs), (val_preds, val_labels, val_probs)

# %%
def visualize_model(model, num_images=6):
    """Muestra de forma aleatoria predicciones individuales del set de validación."""
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure(figsize=(10, 8))

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size(0)):
                images_so_far += 1
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'Predicted: {class_names[preds[j]]} | Actual: {class_names[labels[j]]}')
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

# %%
def evaluate_model(train_data, val_data, class_names):
    """Genera Reporte estadístico completo, Matriz de Confusión y curvas ROC."""
    train_preds, train_labels, train_probs = train_data
    val_preds, val_labels, val_probs = val_data

    print("\n=============================================")
    print("🔬 REPORTE DE CLASIFICACIÓN (VALIDACIÓN)")
    print("=============================================\n")
    print(classification_report(val_labels, val_preds, target_names=class_names))

    # Matriz de Confusión Estructural (Absoluta + Relativa por clase)
    cm = confusion_matrix(val_labels, val_preds)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    labels_combined = np.empty_like(cm).astype('object')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            labels_combined[i, j] = f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)'

    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=labels_combined, fmt='', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Matriz de Confusión (Absoluto + Porcentual)')
    plt.tight_layout()
    plt.show()

    # Evaluación Discriminatoria mediante Curvas ROC
    if len(class_names) == 2:
        plt.figure(figsize=(7, 5))
        for split, labels, probs in [('Train', train_labels, train_probs), ('Validation', val_labels, val_probs)]:
            y_true = np.array(labels)
            y_scores = np.array(probs)[:, 1]

            fpr, tpr, _ = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{split} (AUC = {roc_auc:.2f})')

        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Clasificador Aleatorio')
        plt.xlabel('False Positive Rate (1 - Especificidad)')
        plt.ylabel('True Positive Rate (Sensibilidad / Recall)')
        plt.title('Curva ROC: Train vs Validation')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

# %% [markdown]
# ---
# ## 5) Ejecución del Entrenamiento 💥

# %%
criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

# Entrenamiento adaptado
model_ft, train_data, val_data = train_model(
    model_ft, criterion, optimizer_ft, exp_lr_scheduler,
    dataloaders, dataset_sizes, device, num_epochs=5
)

# %%
visualize_model(model_ft)

# %% [markdown]
# ## 6) GLOSARIO DE MÉTRICAS DE CLASIFICACIÓN
#
# - **Accuracy (Exactitud Global):** Proporción de predicciones correctas sobre el total de casos evaluados. Puede ser engañosa en datasets desbalanceados.
# - **Precisión (Valor Predictivo Positivo):** Proporción de verdaderos positivos sobre el total de alertas positivas emitidas. Mide la fiabilidad del modelo ante una alarma.
# - **Recall (Sensibilidad):** Proporción de verdaderos positivos detectados sobre el total real de casos positivos. Mide la capacidad de captura del fenómeno.
# - **F1-score:** Media armónica entre la Precisión y el Recall, penalizando severamente los desbalances entre ambas.
# - **Curva ROC y AUC:** Gráfica de Sensibilidad frente a la Tasa de Falsos Positivos ($1 - \text{Especificidad}$). El AUC resume de 0 a 1 la robustez discriminatoria global del modelo.

# %%
evaluate_model(train_data, val_data, class_names)

# %% [markdown]
# ### 5.1) Persistencia del Modelo 💾

# %%
torch.save(model_ft, "model.pth")
print("📥 Estructura y pesos guardados en 'model.pth'")

# %% [markdown]
# ---
# ## 6) Explicabilidad Visual con Grad-CAM 🧐

# %%
class GradCAM:
    """Computa mapas de activación por gradientes (Grad-CAM) para PyTorch."""
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.target_layer = target_layer
        self.activation = None
        self.gradient = None

        # Limpieza estricta de hooks previos en la capa objetivo
        if hasattr(target_layer, '_forward_hooks'):
            target_layer._forward_hooks.clear()
        if hasattr(target_layer, '_backward_hooks'):
            target_layer._backward_hooks.clear()
        if hasattr(target_layer, '_full_backward_hooks'):
            target_layer._full_backward_hooks.clear()

        def forward_hook(module, input, output):
            self.activation = output.detach()
        def backward_hook(module, grad_input, grad_output):
            self.gradient = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def __call__(self, input_tensor, class_idx=None):
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        one_hot = torch.zeros_like(output, device=output.device)
        one_hot[0, class_idx] = 1
        output.backward(gradient=one_hot)

        grads = self.gradient[0]  # [C, H, W]
        acts  = self.activation[0] # [C, H, W]

        weights = grads.mean(dim=(1, 2)) # Global Average Pooling [C]

        cam = torch.zeros(acts.shape[1:], device=acts.device)
        for w, a in zip(weights, acts):
            cam += w * a
        cam = torch.relu(cam)

        cam -= cam.min()
        cam /= (cam.max() + 1e-8)
        return cam.cpu().numpy()

# %%
def _denormalize_to_numpy(input_tensor):
    """Función auxiliar para desnormalizar imágenes de Imagenet a formato estándar RGB."""
    img = input_tensor.squeeze().cpu()
    img = TF.normalize(img,
                       [-m/s for m, s in zip([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])],
                       [1/s for s in [0.229, 0.224, 0.225]])
    img = torch.clamp(img, 0, 1)
    return (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

def show_gradcam(input_tensor, mask, title=None):
    """Superpone la máscara Grad-CAM calculada sobre el mapa RGB original."""
    img_np = _denormalize_to_numpy(input_tensor)
    
    heatmap = cv2.resize(mask, (img_np.shape[1], img_np.shape[0]))
    heatmap = (heatmap * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)

    plt.figure(figsize=(5, 5))
    plt.imshow(overlay)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()

def run_and_show_gradcam(model, target_layer, dataloader, class_names, device, num_images=5):
    """Ejecuta de forma secuencial Grad-CAM sobre un lote entero de validación."""
    model.eval()
    gradcam = GradCAM(model, target_layer)
    data_iter = iter(dataloader['val'])
    images_shown = 0 # Contador de imágenes mostradas

    plt.figure(figsize=(4 * num_images, 4))
    while images_shown < num_images:
        try:
            inputs, labels = next(data_iter)
        except StopIteration:
            break
        for idx in range(inputs.size(0)):
            if images_shown >= num_images:
                break
            inp = inputs[idx].unsqueeze(0).to(device)
            lbl = labels[idx].item()
            mask = gradcam(inp, class_idx=lbl)

            img_np = _denormalize_to_numpy(inp)
            heatmap = cv2.resize(mask, (img_np.shape[1], img_np.shape[0]))
            heatmap = (heatmap * 255).astype(np.uint8)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)

            plt.subplot(1, num_images, images_shown + 1)
            plt.imshow(overlay)
            plt.title(f"GT: {class_names[lbl]}")
            plt.axis('off')

            images_shown += 1
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ### Mapeo de Capas según Arquitectura elegida:
# - **CNN 1:** `target_layer = model_ft[0]` (Primera capa convolucional).
# - **CNN 2:** `target_layer = model_ft[3]` (Segunda capa convolucional).
# - **ResNet50:** `target_layer = model_ft.layer4[-1].conv3` (Último cuello de botella).

# %%
if ARQUITECTURA_ACTIVA == 'cnn1':
    target_layer_selected = model_ft[0] 
elif ARQUITECTURA_ACTIVA == 'cnn2':
    target_layer_selected = model_ft[3] 
elif ARQUITECTURA_ACTIVA == 'resnet':
    target_layer_selected = model_ft.layer4[-1].conv3

print(f"👁️ Capa seleccionada para Grad-CAM: {target_layer_selected}")
run_and_show_gradcam(model_ft, target_layer_selected, dataloaders, class_names, device, num_images=5)

# %% [markdown]
# #### Aplicación de Grad-CAM a un Archivo de Imagen Específico

# %%
def gradcam_from_path(model, target_layer, image_path, device, class_names, preprocess=None):
    if preprocess is None:
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    image = Image.open(image_path).convert("RGB")
    input_tensor = preprocess(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor)
        pred_idx = outputs.argmax(dim=1).item()
        pred_class = class_names[pred_idx]

    gradcam = GradCAM(model, target_layer)
    mask = gradcam(input_tensor, class_idx=pred_idx)

    return mask, pred_class, input_tensor

# Pruebas unitarias sobre archivo local
image_path = "/content/bacteria.jpeg"

if os.path.exists(image_path):
    mask, pred_class, tensor_img = gradcam_from_path(
        model=model_ft,
        target_layer=target_layer_selected,
        image_path=image_path,
        device=device,
        class_names=class_names
    )
    print(f"🕵️ Análisis Clínico Grad-CAM - Clase Predicha: {pred_class}")
    show_gradcam(tensor_img, mask, title=f"Grad-CAM → {pred_class}")
else:
    print(f"⚠️ Archivo no encontrado en {image_path} para ejecutar Grad-CAM individual.")

# %% [markdown]
# ---
# ## 7) Pipeline de Inferencia en Producción 🤙

# %%
# Aislamiento de Entorno: Carga independiente para despliegue industrial
mi_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mi_modelo = torch.load("model.pth", map_location=mi_device, weights_only=False)
mi_modelo.eval()
mi_modelo.to(mi_device)

def predict_image(image_path, model, class_names, device):
    """Pipeline cerrado de inferencia para producción."""
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        _, preds = torch.max(outputs, 1)

    return class_names[preds[0].item()]

# Ejecución de Pruebas de Producción
prod_classes = class_names
test_images = [
    "/content/chest_xray/val/NORMAL/NORMAL-1049278-0001.jpeg",
    "/content/chest_xray/val/PNEUMONIA/BACTERIA-1135262-0002.jpeg"
]

for path in test_images:
    if os.path.exists(path):
        predicted_class = predict_image(path, mi_modelo, prod_classes, mi_device)
        print(f"⚡ Inferencia de Producción para {os.path.basename(path)} -> Resultado: {predicted_class}")

# %% [markdown]
# ## 8) Conclusiones del Pipeline Realizado 🎯
# 
# - **Dominio Técnico de PyTorch:** Se consolidó la abstracción de flujos de control mediante módulos avanzados de `Torchvision` e inferencia optimizada con gestión estricta de memoria en GPU (`detach()`, `cpu()`, `pin_memory`).
# - **Auditoría Internivel de Clasificadores:** Al integrar curvas ROC de Entrenamiento y Validación de forma simultánea, el sistema expone de forma directa la presencia de sesgos o sobreajuste (*overfitting*).
# - **Explicabilidad Biomédica:** La incorporación de Grad-CAM provee al especialista humano un método transparente para auditar si las activaciones de la red se centran en estructuras patológicas reales o en artefactos técnicos del dataset.