# =============================================================================
# EXAMEN UNIDAD 2 - MINERÍA DE DATOS
# Sistema de Análisis de Rendimiento Estudiantil
# Dataset: student-mat.csv (UCI Machine Learning Repository)
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# Configuración de estilo
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11

# =============================================================================
# 1. CARGA Y EXPLORACIÓN DE DATOS
# =============================================================================
print("=" * 70)
print("1. CARGA Y EXPLORACIÓN DE DATOS")
print("=" * 70)

df = pd.read_csv('student-mat.csv', sep=';')
print(f"\nDimensiones del dataset: {df.shape[0]} registros x {df.shape[1]} columnas")
print(f"\nColumnas disponibles:\n{list(df.columns)}")
print(f"\nPrimeras 5 filas:")
print(df.head().to_string())
print(f"\nEstadísticas descriptivas:")
print(df.describe().to_string())
print(f"\nValores nulos por columna: {df.isnull().sum().sum()} (total)")
print(f"\nTipos de datos:")
print(df.dtypes.to_string())

# =============================================================================
# 2. PREPROCESAMIENTO Y VARIABLE OBJETIVO
# =============================================================================
print("\n" + "=" * 70)
print("2. PREPROCESAMIENTO Y VARIABLE OBJETIVO")
print("=" * 70)

# Crear variable objetivo binaria: aprobado (G3 >= 10) = 1, reprobado = 0
df['aprobado'] = (df['G3'] >= 10).astype(int)
print(f"\nDistribución de la variable objetivo (aprobado):")
print(f"  Aprobado (1): {df['aprobado'].sum()} ({df['aprobado'].mean()*100:.1f}%)")
print(f"  Reprobado (0): {(df['aprobado']==0).sum()} ({(1-df['aprobado'].mean())*100:.1f}%)")

# Visualizar distribución
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Distribución de G3
axes[0].hist(df['G3'], bins=20, color='#3498db', edgecolor='white', alpha=0.8)
axes[0].axvline(x=10, color='red', linestyle='--', linewidth=2, label='Umbral aprobación (10)')
axes[0].set_xlabel('Nota Final (G3)')
axes[0].set_ylabel('Frecuencia')
axes[0].set_title('Distribución de Notas Finales (G3)')
axes[0].legend()

# Distribución aprobado/reprobado
colors = ['#e74c3c', '#2ecc71']
labels = ['Reprobado', 'Aprobado']
counts = [df['aprobado'].value_counts()[0], df['aprobado'].value_counts()[1]]
axes[1].bar(labels, counts, color=colors, edgecolor='white', width=0.5)
for i, v in enumerate(counts):
    axes[1].text(i, v + 3, str(v), ha='center', fontweight='bold', fontsize=14)
axes[1].set_title('Distribución Aprobado / Reprobado')
axes[1].set_ylabel('Cantidad de Estudiantes')

plt.tight_layout()
plt.savefig('01_distribucion_objetivo.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Gráfico guardado: 01_distribucion_objetivo.png")

# --- PREPROCESAMIENTO DE FEATURES ---
# Eliminar G1, G2, G3 para evitar DATA LEAKAGE y la variable objetivo
features_to_drop = ['G1', 'G2', 'G3', 'aprobado']

# Codificar variables categóricas binarias (yes/no)
binary_cols = ['schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic']
for col in binary_cols:
    df[col] = df[col].map({'yes': 1, 'no': 0})

# Codificar otras categóricas
from sklearn.preprocessing import LabelEncoder
cat_cols = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason', 'guardian']
le_dict = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col + '_enc'] = le.fit_transform(df[col])
    le_dict[col] = le

# Crear DataFrame de features
feature_cols = [c for c in df.columns if c not in features_to_drop and c not in cat_cols]
X = df[feature_cols].copy()
y = df['aprobado'].copy()

print(f"\nFeatures seleccionadas ({X.shape[1]}):")
print(list(X.columns))
print(f"\nShape de X: {X.shape}")
print(f"Shape de y: {y.shape}")
