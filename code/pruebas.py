# %%

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

def evaluar_rendimiento_clinico(etiquetas_reales, logits_predichos, clases):
    """
    Calcula e imprime las métricas de clasificación estándar de la industria.
    """
    # Transformar logits a probabilidades mediante sigmoide (asumiendo caso binario)
    probabilidades = 1 / (1 + np.exp(-logits_predichos))

    # Asignar clase basándose en el umbral estándar de 0.5
    predicciones_binarias = (probabilidades >= 0.5).astype(int)

    # 1. Generar Matriz de Confusión
    matriz = confusion_matrix(etiquetas_reales, predicciones_binarias)
    vn, fp, fn, vp = matriz.ravel()

    # 2. Calcular AUC-ROC
    auc_score = roc_auc_score(etiquetas_reales, probabilidades)

    # 3. Desplegar Reporte Estructurado en Consola
    print("==================================================")
    print("🔬 REPORTE DE EVALUACIÓN CLÍNICA DEL MODELO")
    print("==================================================")
    print(f"Matriz de Confusión Estructural:")
    print(f"  [VN: {vn}]   [FP: {fp}]")
    print(f"  [FN: {fn}]   [VP: {vp}]\n")

    print("Métricas de Forma Detallada:")
    print(classification_report(etiquetas_reales, predicciones_binarias, target_names=clases))
    print(f"Área Bajo la Curva (AUC-ROC): {auc_score:.4f}")
    print("==================================================")

    return vn, fp, fn, vp, auc_score

# --- Simulación de un lote de evaluación de 10 pacientes ---
#0 = NORMAL
#1 = PNEUMONIA
targets = np.array([0, 1, 0, 0, 1, 1, 0, 1, 0, 1])
outputs = np.array([-2.1, 3.4, -0.5, -1.2, 0.1, 4.2, -3.1, -0.2, -1.1, 2.5])
evaluar_rendimiento_clinico(targets, outputs, ['NORMAL', 'PNEUMONIA'])