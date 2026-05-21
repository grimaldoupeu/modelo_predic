# =============================================================================
# PARTE 2: PARTICIÓN DE DATOS Y BASELINE
# =============================================================================

# Continuación de examen_estudiantes.py
# Ejecutar después de la Parte 1

from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

print("\n" + "=" * 70)
print("3. PARTICIÓN DE DATOS (Train 70% / Validation 15% / Test 15%)")
print("=" * 70)

# === EXPLICACIÓN DE LA PARTICIÓN ===
print("""
📖 EXPLICACIÓN DE LA PARTICIÓN:
────────────────────────────────────────────────────────────
La partición de datos divide el dataset en 3 conjuntos:

• ENTRENAMIENTO (70%): Para que el modelo aprenda los patrones.
• VALIDACIÓN (15%): Para ajustar hiperparámetros y seleccionar 
  el mejor modelo sin tocar los datos de prueba.
• PRUEBA (15%): Evaluación final e imparcial del modelo.

Se usa stratify=y para mantener la proporción de clases en cada
conjunto, asegurando representatividad.
────────────────────────────────────────────────────────────
""")

# Paso 1: Separar Train (70%) y Temp (30%)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

# Paso 2: Separar Temp en Validation (15%) y Test (15%)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print(f"Conjunto de ENTRENAMIENTO: {X_train.shape[0]} muestras ({X_train.shape[0]/len(X)*100:.0f}%)")
print(f"  - Aprobados: {y_train.sum()} | Reprobados: {(y_train==0).sum()}")
print(f"Conjunto de VALIDACIÓN:    {X_val.shape[0]} muestras ({X_val.shape[0]/len(X)*100:.0f}%)")
print(f"  - Aprobados: {y_val.sum()} | Reprobados: {(y_val==0).sum()}")
print(f"Conjunto de PRUEBA:        {X_test.shape[0]} muestras ({X_test.shape[0]/len(X)*100:.0f}%)")
print(f"  - Aprobados: {y_test.sum()} | Reprobados: {(y_test==0).sum()}")

# === DATA LEAKAGE ===
print("""
⚠️  DATA LEAKAGE - ¿QUÉ ES Y CÓMO EVITARLO?
────────────────────────────────────────────────────────────
El Data Leakage ocurre cuando información que NO estaría disponible
en producción se filtra al modelo durante el entrenamiento.

EN ESTE PROBLEMA:
• Las columnas G1 (nota 1er periodo) y G2 (nota 2do periodo) son 
  información FUTURA respecto a G3 (nota final).
• Si las usamos como features, el modelo "haría trampa" porque en
  la realidad no tendríamos esas notas antes de predecir G3.

CÓMO LO EVITAMOS:
1. ✅ Eliminamos G1 y G2 del conjunto de features.
2. ✅ El StandardScaler se ajusta SOLO con datos de entrenamiento
   y se aplica (transform) a validación y prueba.
3. ✅ No usamos información del test para ninguna decisión.
────────────────────────────────────────────────────────────
""")

# Escalamiento (fit solo en train, transform en todos)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print("✅ Escalamiento aplicado correctamente (fit solo en train)")

# === MODELO BASELINE ===
print("\n" + "=" * 70)
print("4. MODELO BASELINE")
print("=" * 70)

print("""
📖 MODELO BASELINE:
────────────────────────────────────────────────────────────
El modelo baseline es el punto de referencia mínimo. Usamos
DummyClassifier con estrategia 'most_frequent' que siempre
predice la clase mayoritaria (Aprobado).

Todo modelo útil DEBE superar este baseline.
────────────────────────────────────────────────────────────
""")

baseline = DummyClassifier(strategy='most_frequent', random_state=42)
baseline.fit(X_train_scaled, y_train)

# Evaluar baseline en validación
y_pred_baseline_val = baseline.predict(X_val_scaled)
y_proba_baseline_val = np.full(len(y_val), y_train.mean())  # probabilidad constante

acc_baseline = accuracy_score(y_val, y_pred_baseline_val)
f1_baseline = f1_score(y_val, y_pred_baseline_val, zero_division=0)
auc_baseline = 0.5  # Por definición, modelo sin discriminación

print(f"Resultados del Baseline en VALIDACIÓN:")
print(f"  Accuracy:  {acc_baseline:.4f} ({acc_baseline*100:.1f}%)")
print(f"  F1-Score:  {f1_baseline:.4f}")
print(f"  AUC:       {auc_baseline:.4f}")
print(f"\n  Interpretación: El baseline siempre predice 'Aprobado'.")
print(f"  Logra {acc_baseline*100:.1f}% de accuracy simplemente porque")
print(f"  la mayoría de estudiantes están aprobados.")
