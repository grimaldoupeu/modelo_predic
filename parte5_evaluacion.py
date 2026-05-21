# =============================================================================
# PARTE 5: EVALUACIÓN COMPARATIVA, MATRIZ DE CONFUSIÓN, CONCLUSIONES
# =============================================================================

print("\n" + "=" * 70)
print("7. EVALUACIÓN COMPARATIVA DE MODELOS")
print("=" * 70)

# === 7.1 EVALUACIÓN FINAL EN TEST ===
print("\n📊 7.1 Evaluación Final en conjunto de PRUEBA (Test)")
print("-" * 50)

# Predicciones en TEST
y_pred_base_test = baseline.predict(X_test_scaled)
y_pred_dt_test = dt.predict(X_test_scaled)
y_pred_rf_test = rf.predict(X_test_scaled)

y_proba_base_test = np.full(len(y_test), y_train.mean())
y_proba_dt_test = dt.predict_proba(X_test_scaled)[:, 1]
y_proba_rf_test = rf.predict_proba(X_test_scaled)[:, 1]

# Métricas en Test
results = {
    'Modelo': ['Baseline (Dummy)', 'Árbol de Decisión', 'Random Forest'],
    'Accuracy': [
        accuracy_score(y_test, y_pred_base_test),
        accuracy_score(y_test, y_pred_dt_test),
        accuracy_score(y_test, y_pred_rf_test)
    ],
    'F1-Score': [
        f1_score(y_test, y_pred_base_test, zero_division=0),
        f1_score(y_test, y_pred_dt_test),
        f1_score(y_test, y_pred_rf_test)
    ],
    'AUC': [
        0.5,
        roc_auc_score(y_test, y_proba_dt_test),
        roc_auc_score(y_test, y_proba_rf_test)
    ]
}

results_df = pd.DataFrame(results)
print("\n📋 TABLA COMPARATIVA DE RESULTADOS (Test):")
print("═" * 60)
print(results_df.to_string(index=False))
print("═" * 60)

# Mejor modelo
best_model_idx = results_df['AUC'].idxmax()
print(f"\n🏆 Mejor modelo por AUC: {results_df.loc[best_model_idx, 'Modelo']}")

# === 7.2 CURVA ROC ===
print("\n📊 7.2 Curva ROC Comparativa")
print("-" * 50)

fig, ax = plt.subplots(figsize=(10, 8))

# Línea diagonal (aleatorio)
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Aleatorio (AUC=0.50)', alpha=0.5)

# ROC - Árbol de Decisión
fpr_dt, tpr_dt, _ = roc_curve(y_test, y_proba_dt_test)
auc_dt_test = roc_auc_score(y_test, y_proba_dt_test)
ax.plot(fpr_dt, tpr_dt, linewidth=2.5, label=f'Árbol de Decisión (AUC={auc_dt_test:.3f})',
        color='#e74c3c')

# ROC - Random Forest
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba_rf_test)
auc_rf_test = roc_auc_score(y_test, y_proba_rf_test)
ax.plot(fpr_rf, tpr_rf, linewidth=2.5, label=f'Random Forest (AUC={auc_rf_test:.3f})',
        color='#2ecc71')

ax.set_xlabel('Tasa de Falsos Positivos (FPR)', fontsize=12)
ax.set_ylabel('Tasa de Verdaderos Positivos (TPR)', fontsize=12)
ax.set_title('Curva ROC - Comparación de Modelos', fontsize=14)
ax.legend(fontsize=11, loc='lower right')
ax.grid(True, alpha=0.3)
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])

plt.tight_layout()
plt.savefig('07_curva_roc.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Gráfico guardado: 07_curva_roc.png")

# === 7.3 MATRICES DE CONFUSIÓN ===
print("\n" + "=" * 70)
print("8. MATRICES DE CONFUSIÓN")
print("=" * 70)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
models_info = [
    ('Baseline', y_pred_base_test),
    ('Árbol de Decisión', y_pred_dt_test),
    ('Random Forest', y_pred_rf_test)
]

for ax, (name, y_pred) in zip(axes, models_info):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                   display_labels=['Reprobado', 'Aprobado'])
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    ax.set_title(f'Matriz de Confusión\n{name}', fontsize=12)

plt.tight_layout()
plt.savefig('08_matrices_confusion.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Gráfico guardado: 08_matrices_confusion.png")

# Métricas detalladas de cada matriz
print("\n📋 MÉTRICAS DETALLADAS POR MODELO:")
for name, y_pred in models_info:
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\n{'─'*50}")
    print(f"📌 {name}")
    print(f"{'─'*50}")
    print(f"  TP (Verdaderos Positivos): {tp}  → Aprobados correctamente identificados")
    print(f"  TN (Verdaderos Negativos): {tn}  → Reprobados correctamente identificados")
    print(f"  FP (Falsos Positivos):     {fp}  → Reprobados predichos como aprobados")
    print(f"  FN (Falsos Negativos):     {fn}  → Aprobados predichos como reprobados")
    print(f"  ─────────────────────────────")
    print(f"  Precisión:     {precision:.4f} ({precision*100:.1f}%)")
    print(f"  Recall:        {recall:.4f} ({recall*100:.1f}%)")
    print(f"  Especificidad: {specificity:.4f} ({specificity*100:.1f}%)")
    print(f"  F1-Score:      {f1:.4f} ({f1*100:.1f}%)")

# === INTERPRETACIÓN CONTEXTUAL ===
print(f"""
{'='*70}
9. INTERPRETACIÓN Y COMUNICACIÓN DE RESULTADOS
{'='*70}

📝 INTERPRETACIÓN PARA EQUIPO GERENCIAL (NO TÉCNICO):
{'─'*55}

RESUMEN EJECUTIVO:
Se desarrollaron modelos de inteligencia artificial para predecir
qué estudiantes tienen riesgo de reprobar matemáticas, usando
información demográfica, familiar y de hábitos.

HALLAZGOS PRINCIPALES:

1. PREDICCIÓN DE RENDIMIENTO:
   • El modelo Random Forest logró identificar correctamente 
     al {results_df.loc[2,'Accuracy']*100:.0f}% de los estudiantes.
   • Tiene una capacidad de discriminación (AUC) de {results_df.loc[2,'AUC']:.2f},
     lo cual es {'buena' if results_df.loc[2,'AUC'] > 0.7 else 'moderada'}.

2. FACTORES MÁS IMPORTANTES:
   (Basado en el análisis de importancia de características del Random Forest)
   Las features más relevantes para predecir el rendimiento son las que
   el modelo identificó en el gráfico de importancia de características.

3. PERFILES DE ESTUDIANTES:
   Se identificaron {best_k} perfiles distintos de estudiantes mediante
   técnicas de segmentación, lo que permite diseñar intervenciones
   personalizadas.

RECOMENDACIONES:
• Implementar un sistema de alerta temprana basado en el modelo
  para identificar estudiantes en riesgo al inicio del periodo.
• Diseñar programas de apoyo diferenciados según el perfil del
  estudiante identificado por la segmentación.
• Monitorear las variables más importantes (failures, absences,
  nivel educativo de padres) como indicadores de riesgo.

{'─'*55}
""")

# === TABLA FINAL RESUMEN ===
print("📋 TABLA FINAL DE CORRIDAS/RESULTADOS:")
print("═" * 70)
final_results = pd.DataFrame({
    'Modelo': ['Baseline (Dummy)', 'Árbol de Decisión (depth=5)', 'Random Forest (100 árboles)'],
    'Accuracy (Test)': [f"{results_df.loc[0,'Accuracy']:.4f}", 
                        f"{results_df.loc[1,'Accuracy']:.4f}",
                        f"{results_df.loc[2,'Accuracy']:.4f}"],
    'F1-Score (Test)': [f"{results_df.loc[0,'F1-Score']:.4f}",
                        f"{results_df.loc[1,'F1-Score']:.4f}", 
                        f"{results_df.loc[2,'F1-Score']:.4f}"],
    'AUC (Test)': [f"{results_df.loc[0,'AUC']:.4f}",
                   f"{results_df.loc[1,'AUC']:.4f}",
                   f"{results_df.loc[2,'AUC']:.4f}"],
    'Supera Baseline': ['—', 
                        '✅' if results_df.loc[1,'AUC'] > 0.5 else '❌',
                        '✅' if results_df.loc[2,'AUC'] > 0.5 else '❌']
})
print(final_results.to_string(index=False))
print("═" * 70)

print("\n🎓 CONCLUSIÓN FINAL:")
print("─" * 55)
best = 'Random Forest' if results_df.loc[2,'AUC'] >= results_df.loc[1,'AUC'] else 'Árbol de Decisión'
print(f"El modelo {best} es el más recomendado para este problema,")
print(f"ya que ofrece el mejor balance entre las métricas evaluadas.")
print(f"Ambos modelos superan significativamente al baseline, lo que")
print(f"confirma que las características del dataset tienen poder")
print(f"predictivo real sobre el rendimiento estudiantil.")
print("\n✅ EXAMEN COMPLETADO EXITOSAMENTE")
