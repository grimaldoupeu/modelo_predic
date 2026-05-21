# =============================================================================
# PARTE 4: CLASIFICACIÓN (Árbol de Decisión + Random Forest)
# =============================================================================

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             confusion_matrix, ConfusionMatrixDisplay,
                             classification_report, roc_curve)
import time

print("\n" + "=" * 70)
print("6. CLASIFICACIÓN")
print("=" * 70)

# === 6.1 ÁRBOL DE DECISIÓN ===
print("\n🌲 6.1 Árbol de Decisión")
print("-" * 40)

print("""
📖 ÁRBOL DE DECISIÓN:
────────────────────────────────────────────────────────────
Es un modelo que divide los datos en nodos según reglas 
if-then basadas en las features. Cada nodo representa una
pregunta sobre una característica, y las hojas son las 
predicciones finales.

Ventajas: Interpretable, no requiere escalado.
Desventajas: Propenso a overfitting si no se limita la profundidad.
────────────────────────────────────────────────────────────
""")

t0 = time.time()
dt = DecisionTreeClassifier(max_depth=5, random_state=42, class_weight='balanced')
dt.fit(X_train_scaled, y_train)
time_dt = time.time() - t0

# Predicciones en validación
y_pred_dt_val = dt.predict(X_val_scaled)
y_proba_dt_val = dt.predict_proba(X_val_scaled)[:, 1]

acc_dt = accuracy_score(y_val, y_pred_dt_val)
f1_dt = f1_score(y_val, y_pred_dt_val)
auc_dt = roc_auc_score(y_val, y_proba_dt_val)

print(f"Resultados en VALIDACIÓN:")
print(f"  Accuracy:  {acc_dt:.4f} ({acc_dt*100:.1f}%)")
print(f"  F1-Score:  {f1_dt:.4f}")
print(f"  AUC:       {auc_dt:.4f}")
print(f"  Tiempo:    {time_dt:.4f}s")

# Visualizar árbol
fig, ax = plt.subplots(figsize=(20, 10))
plot_tree(dt, feature_names=X.columns.tolist(), class_names=['Reprobado', 'Aprobado'],
          filled=True, rounded=True, fontsize=8, ax=ax,
          proportion=True, impurity=False)
ax.set_title('Árbol de Decisión (max_depth=5)', fontsize=16)
plt.tight_layout()
plt.savefig('05_arbol_decision.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Gráfico guardado: 05_arbol_decision.png")

# === 6.2 RANDOM FOREST ===
print("\n🌳 6.2 Random Forest")
print("-" * 40)

print("""
📖 RANDOM FOREST:
────────────────────────────────────────────────────────────
Es un ensemble de múltiples árboles de decisión entrenados
con submuestras aleatorias (bagging). Cada árbol vota y la
predicción final es por mayoría.

Ventajas: Robusto, reduce overfitting, importancia de features.
Desventajas: Menos interpretable, más costoso computacionalmente.
────────────────────────────────────────────────────────────
""")

t0 = time.time()
rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42,
                            class_weight='balanced', n_jobs=-1)
rf.fit(X_train_scaled, y_train)
time_rf = time.time() - t0

# Predicciones en validación
y_pred_rf_val = rf.predict(X_val_scaled)
y_proba_rf_val = rf.predict_proba(X_val_scaled)[:, 1]

acc_rf = accuracy_score(y_val, y_pred_rf_val)
f1_rf = f1_score(y_val, y_pred_rf_val)
auc_rf = roc_auc_score(y_val, y_proba_rf_val)

print(f"Resultados en VALIDACIÓN:")
print(f"  Accuracy:  {acc_rf:.4f} ({acc_rf*100:.1f}%)")
print(f"  F1-Score:  {f1_rf:.4f}")
print(f"  AUC:       {auc_rf:.4f}")
print(f"  Tiempo:    {time_rf:.4f}s")

# Feature Importance
importances = pd.Series(rf.feature_importances_, index=X.columns)
importances_sorted = importances.sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(10, 8))
importances_sorted.plot(kind='barh', color='#3498db', edgecolor='white', ax=ax)
ax.set_title('Importancia de Características - Random Forest', fontsize=14)
ax.set_xlabel('Importancia')
plt.tight_layout()
plt.savefig('06_feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Gráfico guardado: 06_feature_importance.png")

# === EXPLICACIÓN DE MÉTRICAS ===
print("""
📖 MÉTRICAS DE EVALUACIÓN:
────────────────────────────────────────────────────────────
• ACCURACY: Proporción de predicciones correctas sobre el total.
  Fórmula: (TP + TN) / (TP + TN + FP + FN)
  Limitación: Puede ser engañosa con clases desbalanceadas.

• F1-SCORE: Media armónica de Precisión y Recall.
  Fórmula: 2 × (Precision × Recall) / (Precision + Recall)
  Útil cuando las clases están desbalanceadas.

• AUC (Area Under the Curve): Mide la capacidad del modelo
  para distinguir entre clases. 
  AUC = 0.5: Sin discriminación (aleatorio)
  AUC = 1.0: Discriminación perfecta
────────────────────────────────────────────────────────────
""")
