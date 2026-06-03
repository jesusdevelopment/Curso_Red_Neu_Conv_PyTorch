# %% [markdown]
## Importación de Bibliotecas
from IPython.display import display, Markdown
import torch
import torch.nn as nn
from google.colab import drive
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Dispositivo de cómputo activo: {device}")
# %% [markdown]
## Curso de CNN con PyTorch

# %% [markdown]
## 1) 🎯 ¿Qué es el Pooling?
Pooling="""
El **pooling** es una operación que se utiliza para **reducir la dimensionalidad espacial** (alto y ancho) de una imagen o de los mapas de activación.

Esto reduce la cantidad de parámetros y computación en la red, y ayuda a controlar el sobreajuste / overfitting."""
display(Markdown(Pooling))
# %% [markdown] 
## 2) 📐 Fundamento Matemático
Fundamento=r"""
Dado un tensor de entrada \( X ∈ ℝ^{H×W} \), el pooling aplica una función (máximo, promedio, etc.) sobre regiones no superpuestas (o con cierto `stride`).

**Max Pooling:**

$$
y_{i,j} = \max_{(m,n) ∈ R_{i,j}} x_{m,n}
$$

**Average Pooling:**

$$
y_{i,j} = \frac{1}{|R_{i,j}|} \sum_{(m,n) ∈ R_{i,j}} x_{m,n}
$$

Donde \( R_{i,j} \) es la región local (como una ventana 2x2) que se "pooliza"."""
display(Markdown(Fundamento))

# %% [markdown]
## 3) Ejemplo de MaxPooling y AvgPooling en Tensores
# Simulación de un lote de datos: 1 imagen, 1 canal (escala de grises), de 64x64 píxeles
input_tensor = torch.randn(1, 1, 64, 64)

# 1. Configuración de Max Pooling (Operación estándar en capas intermedias)
max_layer = nn.MaxPool2d(kernel_size=2, stride=2)
output_max = max_layer(input_tensor)

# 2. Configuración de Average Pooling (Operación de suavizado u optimización)
avg_layer = nn.AvgPool2d(kernel_size=2, stride=2)
output_avg = avg_layer(input_tensor)

# Verificación de dimensiones resultantes en la consola del Notebook
print("Dimensión original del tensor: ", input_tensor.shape) # [1, 1, 64, 64]
print("Dimensión final tras MaxPool:  ", output_max.shape)     # [1, 1, 32, 32]
print("Dimensión final tras AvgPool:  ", output_avg.shape)     # [1, 1, 32, 32]
# %% [markdown]
## Aclaración
Aclaracion="""
Aclaración:

- Pooling es un downsampling basado en estadísticas locales,
- Resize es un escalado con interpolación."""
display(Markdown(Aclaracion))
# %% [markdown]
## 4) 🧪 Ejemplos
### 🧪 4.1) Imagen Gradiente Blanco y Negro

# Imagen gradiente (64x64)
gradient = np.tile(np.linspace(0, 1, 64), (64, 1))
img_gray = torch.tensor(gradient, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1,1,64,64)

# Poolings
pool_max = nn.MaxPool2d(kernel_size=2, stride=2)
pool_avg = nn.AvgPool2d(kernel_size=2, stride=2)

img_max = pool_max(img_gray)
img_avg = pool_avg(img_gray)

# Visualización
fig, axs = plt.subplots(1, 3, figsize=(12, 4))
axs[0].imshow(img_gray.squeeze(), cmap='gray')
axs[0].set_title("Original")

axs[1].imshow(img_max.squeeze(), cmap='gray')
axs[1].set_title("Max Pooling")

axs[2].imshow(img_avg.squeeze(), cmap='gray')
axs[2].set_title("Avg Pooling")

for ax in axs: ax.axis('off')
plt.tight_layout()
plt.show()

### 🧪 4.2) Imagen tipo letra "P"

# Crear imagen tipo "P"
img_pil = Image.new("L", (64, 64), color=0)
draw = ImageDraw.Draw(img_pil)
draw.text((10, 10), "P", fill=255)

img_tensor = torch.tensor(np.array(img_pil) / 255.0, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

img_max = pool_max(img_tensor)
img_avg = pool_avg(img_tensor)

# Visualización
fig, axs = plt.subplots(1, 3, figsize=(12, 4))
axs[0].imshow(img_tensor.squeeze(), cmap='gray')
axs[0].set_title("Letra 'P' Original")

axs[1].imshow(img_max.squeeze(), cmap='gray')
axs[1].set_title("Max Pooling")

axs[2].imshow(img_avg.squeeze(), cmap='gray')
axs[2].set_title("Avg Pooling")

for ax in axs: ax.axis('off')
plt.tight_layout()
plt.show()

### 🧪 4.3) Imagen RGB subida por el usuario

drive.mount('/content/drive')
import torchvision.transforms as transforms

# Ruta a imagen local
img_path = "/content/drive/MyDrive/Data/Curso de Redes Neuronales Convolucionales/GBM.jpeg"
img = Image.open(img_path).convert('RGB')

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()  # (3, H, W)
])

img_tensor = transform(img).unsqueeze(0)  # (1, 3, 64, 64)

pool_max_rgb = nn.MaxPool2d(2, 2)
pool_avg_rgb = nn.AvgPool2d(2, 2)

img_max = pool_max_rgb(img_tensor)
img_avg = pool_avg_rgb(img_tensor)

# Mostrar los 3 canales como imágenes
fig, axs = plt.subplots(3, 3, figsize=(12, 8))
titles = ["Original", "Max Pool", "Avg Pool"]
for i in range(3):
    axs[i][0].imshow(img_tensor[0][i].numpy(), cmap='gray')
    axs[i][1].imshow(img_max[0][i].detach().numpy(), cmap='gray')
    axs[i][2].imshow(img_avg[0][i].detach().numpy(), cmap='gray')

for j in range(3):
    for i in range(3):
        axs[j][i].axis('off')
        if j == 0:
            axs[j][i].set_title(titles[i])

plt.tight_layout()
plt.show()
# %% [markdown]
Uso="""
## 5) ✅ ¿Cuándo usar MaxPooling vs AvgPooling?
| Técnica       | Cuándo Usar                                                                 |
|---------------|------------------------------------------------------------------------------|
| Max Pooling   | Cuando querés capturar la **característica más dominante** (ej. bordes)     |
| Avg Pooling   | Cuando te interesa suavizar o hacer un resumen **más general y robusto**    |
| Sin Pooling   | En tareas sensibles al **detalle espacial** (ej. segmentación densa)         |
## 6) 💡 Tips
- Pooling ayuda a que las redes sean más **invariantes a traslaciones pequeñas**.
- **MaxPooling** funciona muy bien en problemas de visión porque retiene la información más fuerte (por ejemplo, bordes).
- En arquitecturas modernas como **ResNet**, el pooling final se hace con `AdaptiveAvgPool2d((1,1))` antes de la capa `Linear`.
- Alternativas modernas:
  - ✅ **Strided convolutions** (sin necesidad de pooling explícito)
  - ✅ **Blur pooling**: reducción menos agresiva
  - ✅ **Attention pooling**: más inteligente, adaptativo"""
display(Markdown(Uso))

# %% [markdown]
## 6) Convoluciones Avanzadas
# 6.1) Strided Convolutions
# Reemplazo directo de un bloque [Conv2d(S=1) + MaxPool2d(2,2)] por una Convolución con Salto
# Reduce las dimensiones espaciales a la mitad mientras aprende los pesos de compresión
strided_conv = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)