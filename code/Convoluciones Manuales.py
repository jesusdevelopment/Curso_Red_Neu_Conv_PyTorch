# %% [markdown]
## Curso de CNN con PyTorch
from IPython.display import display, Markdown
# %% [markdown]
## 1) 🎯 Capas Convolucionales

capas_convolucionales="""
Las **capas convolucionales** son el corazón de las CNN (Redes Neuronales Convolucionales).

Permiten extraer **características espaciales** como bordes, texturas o patrones complejos directamente desde las imágenes de entrada."""
display(Markdown(capas_convolucionales))
# %% [markdown]
## 2) 📐 Fundamento Matemático
### 2.1) 🔢 ¿Qué es una convolución?
Una_convolucion=r"""
Una **convolución** es una operación matemática que aplica un **filtro (o kernel)** sobre una imagen para producir un **mapa de características**.

**Fórmula:**
$$
O(i, j) = \sum_{m=0}^{kH-1} \sum_{n=0}^{kW-1} I(i+m, j+n) \cdot K(m,n)
$$

Donde:
- \( I \): Imagen de entrada  
- \( K \): Kernel o filtro  
- \( O \): Salida (feature map)  
- \( kH, kW \): Altura y ancho del filtro  """
display(Markdown(Una_convolucion))
# %% [markdown]
### 2.2) 🧰 Componentes importantes
Filtros_Kernel=r"""
### Filtros (Kernels)
###
**Filtros (Kernels)**
- Son pequeños tensores (e.g. 3×3, 5×5) entrenables.
- Detectan patrones específicos como bordes, líneas o texturas.

**Stride**
- Determina cuánto se desplaza el filtro al aplicarse.
- `stride=1` → máxima superposición  
- `stride=2` → reduce resolución más rápido

**Padding**
- Añade bordes de ceros para no perder información en los bordes.
- `same` padding → salida del mismo tamaño que la entrada  
- `valid` padding → sin relleno

**Canales**
- Imágenes RGB tienen 3 canales.
- Cada filtro se aplica en **todos los canales**, y se suman."""
display(Markdown(Filtros_Kernel))
# %% [markdown] 
# Todo el texto (incluyendo el título) debe ir dentro de las comillas
Patrones_espaciales = r"""
### 2.3) 🔍 ¿Para qué sirven?
- Extraer patrones espaciales
- Detectar jerarquías (de píxeles a bordes, de bordes a objetos)
- Reducir dimensionalidad con stride o pooling
"""
display(Markdown(Patrones_espaciales))
# %% [markdown] 

## 3) 🧪 Ejemplos
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from PIL import Image

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Dispositivo de cómputo activo: {device}")
# %% [markdown]
### 📷 Imagen 1: Gradiente blanco y negro

# Imagen sintética: gradiente horizontal
gradient = np.tile(np.linspace(0, 1, 64), (64, 1))
img_grad = torch.tensor(gradient, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

### 📷 Imagen 2: Letra "P"

P_img = np.zeros((64, 64))
## Parte vertical
P_img[10:50, 10:20] = 1
## Parte superior del círculo de la P
P_img[10:20, 10:40] = 1
## Parte media horizontal de la P
P_img[30:40, 10:40] = 1
## Borde derecho del "círculo"
P_img[20:30, 30:40] = 1
## Parte interior para que no parezca una B
P_img[20:30, 20:30] = 0  # vaciar la parte interior
img_P = torch.tensor(P_img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

# %% [markdown]
### 📷 Imagen 3: RGB cargada por el usuario

# Carga y preprocesamiento

# Cargar imagen
from google.colab import drive
drive.mount('/content/drive')
img_path = "/content/drive/MyDrive/Data/Curso de Redes Neuronales Convolucionales/GBM.jpeg"

try:
    # Cargamos en escala de grises para procesar un solo canal de entrada (1, 1, H, W)
    img_gbm = Image.open(img_path).convert("L").resize((64, 64))
    transform = transforms.Compose([transforms.ToTensor()])
    img_gbm_tensor = transform(img_gbm).unsqueeze(0).to(device)
    print("¡Éxito! Imagen del GBM cargada y transferida a la GPU.")
    gbm_disponible = True
except FileNotFoundError:
    print(f"⚠️ Archivo no detectado en: '{img_path}'")
    print("Nota: El script ejecutará solo las imágenes sintéticas hasta que subas el JPEG a Colab.")
    gbm_disponible = False
    
### 🧪 3.1) Aplicar un filtro de detección de bordes
# Filtro Sobel Vertical (Detecta gradientes en el eje X)
sobel_x = torch.tensor([[-1.,  0.,  1.],
                        [-2.,  0.,  2.],
                        [-1.,  0.,  1.]])

# Filtro Sobel Horizontal (Detección de Bordes Horizontales)
sobel_y = torch.tensor([[-1., -2., -1.],
                        [ 0.,  0.,  0.],
                        [ 1.,  2.,  1.]])

# Apilamos ambos operadores para procesarlos en una única operación convolucional
# Shape resultante: (Filtros_out=2, Canales_in=1, H=3, W=3)
kernels = torch.stack([sobel_y, sobel_x]).unsqueeze(1).to(device)
# %% [markdown]
# ## 3) Definición de Funciones Analíticas y Visualización

# %%
def aplicar_convolucion(img, kernel_tensor, stride=1, padding=1):
    return F.conv2d(img, kernel_tensor, stride=stride, padding=padding)

def visualizar_bancos_de_filtros(img, title):
    """Aplica el banco de filtros y grafica los mapas de características resultantes."""
    out = aplicar_convolucion(img, kernels)
    
    # Retornamos los tensores a CPU y formato NumPy para el renderizado de Matplotlib
    out_np = out.squeeze().detach().cpu().numpy()
    img_np = img.squeeze().detach().cpu().numpy()

    plt.figure(figsize=(12, 4))

    # Canal Original
    plt.subplot(1, 3, 1)
    plt.imshow(img_np, cmap='gray')
    plt.title("Entrada Original")
    plt.axis("off")

    # Respuesta al filtro horizontal
    plt.subplot(1, 3, 2)
    plt.imshow(out_np[0], cmap='gray')
    plt.title("Respuesta Sobel Y (Bordes Horiz.)")
    plt.axis("off")

    # Respuesta al filtro vertical
    plt.subplot(1, 3, 3)
    plt.imshow(out_np[1], cmap='gray')
    plt.title("Respuesta Sobel X (Bordes Vert.)")
    plt.axis("off")

    plt.suptitle(title, y=1.02, fontsize=12)
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## 4) Ejecución y Extracción de Características Espaciales

# %%
# Evaluación sobre las señales de control (Sintéticas)
visualizar_bancos_de_filtros(img_grad, "Análisis de Bordes: Gradiente Continuo")
visualizar_bancos_de_filtros(img_P, "Análisis de Bordes: Estructura Digital 'P'")

# Evaluación sobre la anatomía médica (Si el archivo JPEG fue cargado)
if gbm_disponible:
    visualizar_bancos_de_filtros(img_gbm_tensor, "Análisis de Bordes Histológicos / Anatómicos: GBM")

# %% [markdown]
# ## 5) Evaluación Dimensional: Impacto de Stride y Padding

# %%
def analizar_reduccion_espacial(img, combinaciones, title):
    """Visualiza cómo se altera el tamaño de la matriz de salida según el paso del filtro."""
    kernel_operativo = sobel_y.unsqueeze(0).unsqueeze(0).to(device)
    
    plt.figure(figsize=(4 * len(combinaciones), 4))
    
    for i, (s, p) in enumerate(combinaciones):
        res = aplicar_convolucion(img, kernel_operativo, stride=s, padding=p)
        res_np = res.squeeze().detach().cpu().numpy()
        
        plt.subplot(1, len(combinaciones), i + 1)
        plt.imshow(res_np, cmap='gray')
        # Imprimimos explícitamente el Shape resultante para verificar el submuestreo
        plt.title(f"Stride={s}, Padding={p}\nDim: {res_np.shape}")
        plt.axis('off')
        
    plt.suptitle(title, y=1.05)
    plt.tight_layout()
    plt.show()

# Combinaciones analíticas de Stride y Padding
configuraciones = [(1, 1), (2, 1), (4, 1)]

analizar_reduccion_espacial(img_P, configuraciones, "Efecto Dimensional en Objeto Sintético")

if gbm_disponible:
    analizar_reduccion_espacial(img_gbm_tensor, configuraciones, "Efecto Dimensional en Tensor GBM")# %% [markdown]
## 4) 💡 Tips
Tips=r"""
| Tema | Recomendaciones |
|------|------------------|
| Filtros | Diseñados para extraer características específicas. Los primeros detectan bordes; los últimos detectan partes de objetos. |
| Stride > 1 | Útil para reducir tamaño espacial y acelerar procesamiento. Pero se puede perder información fina. |
| Padding | Siempre usar `padding=1` si querés conservar la dimensión. Muy importante para no achicar imágenes rápidamente. |
| Canales | En RGB, cada filtro procesa todos los canales y produce 1 mapa de salida. Convierte 3 canales a N mapas. | """
display(Markdown(Tips))
