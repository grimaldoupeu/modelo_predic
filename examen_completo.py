# =============================================================================
# EXAMEN UNIDAD 2 - MINERÍA DE DATOS (PIPELINE MEJORADO MULTI-MATERIA)
# Sistema de Análisis y Predicción de Rendimiento Estudiantil (MAT y POR)
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             confusion_matrix, ConfusionMatrixDisplay,
                             classification_report, roc_curve, silhouette_score)
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
warnings.filterwarnings('ignore')

# Configuración estética para wow effect en gráficos
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("muted")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

# Crear carpeta para guardar gráficos si no existe
os.makedirs('static/charts', exist_ok=True)

print("=" * 80)
print("INICIANDO PIPELINE MULTI-MATERIA DE MINERÍA DE DATOS - EXAMEN U2")
print("=" * 80)

# Carga de datos
mat = pd.read_csv('student-mat.csv', sep=';')
por = pd.read_csv('student-por.csv', sep=';')

print(f"Dataset Matemáticas (MAT): {mat.shape[0]} estudiantes, {mat.shape[1]} columnas")
print(f"Dataset Portugués (POR):  {por.shape[0]} estudiantes, {por.shape[1]} columnas")

# Detección de estudiantes comunes (Leakage alert)
id_cols = ['school','sex','age','address','famsize','Pstatus','Medu','Fedu','Mjob','Fjob','reason','guardian']
shared_students = pd.merge(mat, por, on=id_cols, suffixes=('_mat', '_por'))
num_shared = len(shared_students)

print("\n" + "!" * 80)
print(f"ALERTA CRÍTICA DE DATA LEAKAGE: SE DETECTARON {num_shared} ESTUDIANTES EN COMÚN")
print("!" * 80)
print(f"- Los datasets de Matemáticas y Portugués comparten exactamente {num_shared} estudiantes.")
print("- Esto representa un riesgo muy grave de DATA LEAKAGE si se fusionaran de manera ingenua.")
print("- SOLUCIÓN ADOPTADA:")
print("  Para evitar fuga de información por perfiles cruzados y reportar predicciones limpias,")
print("  entrenaremos un modelo predictivo EXCLUSIVO para Matemáticas (MAT) y otro EXCLUSIVO")
print("  para Portugués (POR) de forma 100% independiente y aislada.")
print("!" * 80 + "\n")

# Función modular para procesar cada materia de forma aislada
def procesar_materia(df_raw, nombre_materia, prefix):
    print("=" * 80)
    print(f"PROCESANDO MATERIA: {nombre_materia} ({prefix.upper()})")
    print("=" * 80)
    
    # 1. Manejo de Notas G3 = 0 (Deserción Escolar)
    g0_students = df_raw[df_raw['G3'] == 0]
    print(f"- Se detectaron {len(g0_students)} estudiantes con nota final G3 = 0 en {nombre_materia}.")
    print(f"- Estudiantes que tenían nota inicial G1 > 0: {(g0_students['G1'] > 0).sum()} de {len(g0_students)}")
    
    # Filtrar para clasificador activo (estudiantes que terminaron el curso)
    df_active = df_raw[df_raw['G3'] > 0].copy()
    print(f"-> Estudiantes activos para modelado: {df_active.shape[0]} (excluidos {len(g0_students)} abandonos)")
    
    # Definir umbral de aprobación ordinario (nota >= 10 sobre 20)
    df_active['aprobado'] = (df_active['G3'] >= 10).astype(int)
    class_counts = df_active['aprobado'].value_counts()
    print(f"Distribución de clases (Aprobado/Reprobado activo):")
    print(f"  Aprobado (1):  {class_counts.get(1, 0)} ({class_counts.get(1, 0)/len(df_active)*100:.1f}%)")
    print(f"  Reprobado (0): {class_counts.get(0, 0)} ({class_counts.get(0, 0)/len(df_active)*100:.1f}%)")
    
    # Guardar gráfica 1: Distribución
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(df_raw['G3'], bins=20, color='#4A90E2', edgecolor='white', alpha=0.8)
    axes[0].axvline(x=10, color='#D0021B', linestyle='--', linewidth=2, label='Umbral Aprobación (10)')
    axes[0].set_xlabel('Nota Final (G3)')
    axes[0].set_ylabel('Cantidad de Estudiantes')
    axes[0].set_title(f'Distribución de Notas Finales ({nombre_materia})', fontsize=12, fontweight='bold')
    axes[0].legend()
    
    colors = ['#F5A623', '#2ECC71']
    axes[1].bar(['Reprobado Activo', 'Aprobado Activo'], [class_counts.get(0, 0), class_counts.get(1, 0)], color=colors, edgecolor='white', width=0.4)
    for i, v in enumerate([class_counts.get(0, 0), class_counts.get(1, 0)]):
        axes[1].text(i, v + (v * 0.02) + 1, f"{v}\n({v/len(df_active)*100:.1f}%)", ha='center', fontweight='bold')
    axes[1].set_title(f'Estudiantes Activos en {nombre_materia}', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Estudiantes')
    plt.tight_layout()
    plt.savefig(f'static/charts/01_distribucion_{prefix}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Preprocesamiento de variables categóricas y booleanas
    binary_cols = ['schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic']
    for col in binary_cols:
        df_active[col] = df_active[col].map({'yes': 1, 'no': 0})
        
    cat_cols = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason', 'guardian']
    for col in cat_cols:
        le = LabelEncoder()
        df_active[col + '_enc'] = le.fit_transform(df_active[col])
        
    # Eliminar notas G1 y G2 para evitar Fuga de Información del Futuro
    features_to_drop = ['G1', 'G2', 'G3', 'aprobado']
    feature_cols = [c for c in df_active.columns if c not in features_to_drop and c not in cat_cols]
    
    X = df_active[feature_cols].copy()
    y = df_active['aprobado'].copy()
    
    # 2. PARTICIÓN DE DATOS (70% Train, 15% Val, 15% Test)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)
    
    # Escalamiento limpio (fit únicamente en train)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Entrenar Baseline (Dummy)
    baseline = DummyClassifier(strategy='most_frequent', random_state=42)
    baseline.fit(X_train_scaled, y_train)
    y_pred_base = baseline.predict(X_val_scaled)
    acc_base = accuracy_score(y_val, y_pred_base)
    f1_base = f1_score(y_val, y_pred_base, zero_division=0)
    
    print(f"\n1. PARTICIÓN Y BASELINE ({prefix.upper()}):")
    print(f"  - Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    print(f"  - Accuracy Baseline (Val): {acc_base:.4f}")
    print(f"  - F1 Baseline (Val):       {f1_base:.4f}")
    
    # 3. SEGMENTACIÓN (CLUSTERING)
    num_features = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures',
                    'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences']
    X_cluster = df_active[num_features].copy()
    X_cluster_scaled = StandardScaler().fit_transform(X_cluster)
    
    inertias = []
    silhouettes = []
    k_range = range(2, 9)
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        lbls = km.fit_predict(X_cluster_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_cluster_scaled, lbls))
        
    best_k = list(k_range)[np.argmax(silhouettes)]
    print(f"\n2. CLUSTERING ({prefix.upper()}):")
    print(f"  - K Óptimo sugerido por Silhouette: K={best_k} (Score: {max(silhouettes):.4f})")
    
    # Graficar codo y silhouette
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(k_range, inertias, 'o-', color='#34495E', linewidth=2)
    axes[0].set_title(f'Método del Codo ({nombre_materia})', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Número de Clústeres (K)')
    axes[0].set_ylabel('Inercia')
    
    axes[1].plot(k_range, silhouettes, 's-', color='#E74C3C', linewidth=2)
    axes[1].axvline(x=best_k, color='#2ECC71', linestyle='--', label=f'K óptimo ({best_k})')
    axes[1].set_title(f'Índice Silhouette ({nombre_materia})', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Número de Clústeres (K)')
    axes[1].set_ylabel('Score Silhouette')
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(f'static/charts/02_codo_silhouette_{prefix}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Dendrograma
    Z = linkage(X_cluster_scaled, method='ward')
    plt.figure(figsize=(12, 5))
    dendrogram(Z, truncate_mode='lastp', p=30, show_contracted=True)
    plt.title(f'Dendrograma de Clustering Jerárquico - {nombre_materia}', fontsize=14, fontweight='bold')
    plt.xlabel('Índice Estudiantil o Subgrupo')
    plt.ylabel('Distancia Ward')
    plt.tight_layout()
    plt.savefig(f'static/charts/03_dendrograma_{prefix}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Asignar clústeres finales
    km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    df_active['cluster_kmeans'] = km_final.fit_predict(X_cluster_scaled)
    
    cluster_profiles = df_active.groupby('cluster_kmeans')[num_features + ['aprobado']].mean()
    
    # Mapa de calor de perfiles normalizados
    profile_norm = (cluster_profiles - cluster_profiles.mean()) / cluster_profiles.std()
    plt.figure(figsize=(10, 6))
    sns.heatmap(profile_norm.T, annot=True, fmt=".2f", cmap="RdYlBu_r", center=0, linewidths=0.5)
    plt.title(f'Mapa de Calor de Perfiles de Estudiantes ({nombre_materia})', fontsize=13, fontweight='bold')
    plt.xlabel('Clúster')
    plt.ylabel('Variable de Perfil')
    plt.tight_layout()
    plt.savefig(f'static/charts/04_perfiles_{prefix}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4. CLASIFICACIÓN
    # Árbol de Decisión con class_weight='balanced'
    dt = DecisionTreeClassifier(max_depth=4, class_weight='balanced', random_state=42)
    dt.fit(X_train_scaled, y_train)
    
    # Random Forest con class_weight='balanced'
    rf = RandomForestClassifier(n_estimators=150, max_depth=6, class_weight='balanced', random_state=42, n_jobs=-1)
    rf.fit(X_train_scaled, y_train)
    
    # Graficar Árbol
    plt.figure(figsize=(20, 10))
    plot_tree(dt, feature_names=feature_cols, class_names=['Reprobado', 'Aprobado'], filled=True, rounded=True, fontsize=9)
    plt.title(f'Árbol de Decisión con Pesos Balanceados - {nombre_materia}', fontsize=16, fontweight='bold')
    plt.savefig(f'static/charts/05_arbol_decision_{prefix}.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    # Importancia de features de Random Forest
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances[indices[:12]], y=[feature_cols[i] for i in indices[:12]], palette="viridis")
    plt.title(f'Top 12 Variables Más Influyentes - {nombre_materia}', fontsize=13, fontweight='bold')
    plt.xlabel('Importancia de Gini')
    plt.tight_layout()
    plt.savefig(f'static/charts/06_importancia_features_{prefix}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 5. EVALUACIÓN Y MATRICES DE CONFUSIÓN EN TEST
    y_pred_base_test = baseline.predict(X_test_scaled)
    y_pred_dt_test = dt.predict(X_test_scaled)
    y_pred_rf_test = rf.predict(X_test_scaled)
    
    y_prob_base_test = np.full(len(y_test), y_train.mean())
    y_prob_dt_test = dt.predict_proba(X_test_scaled)[:, 1]
    y_prob_rf_test = rf.predict_proba(X_test_scaled)[:, 1]
    
    metrics = []
    for name, y_pred, y_prob in [
        ('Baseline (Dummy)', y_pred_base_test, y_prob_base_test),
        ('Árbol de Decisión', y_pred_dt_test, y_prob_dt_test),
        ('Random Forest', y_pred_rf_test, y_prob_rf_test)
    ]:
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        accuracy = accuracy_score(y_test, y_pred)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        
        metrics.append({
            'Modelo': name,
            'Accuracy': accuracy,
            'Precisión': precision,
            'Recall (Sensibilidad)': recall,
            'F1-Score': f1,
            'AUC-ROC': auc,
            'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn
        })
        
    metrics_df = pd.DataFrame(metrics)
    print(f"\n3. TABLA COMPARATIVA ({prefix.upper()}):")
    print(metrics_df[['Modelo', 'Accuracy', 'F1-Score', 'AUC-ROC', 'Recall (Sensibilidad)']].to_string(index=False))
    
    # Graficar Curva ROC
    plt.figure(figsize=(8, 7))
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Clasificador Aleatorio (AUC = 0.50)')
    for name, y_prob, color in [
        ('Árbol de Decisión', y_prob_dt_test, '#E74C3C'),
        ('Random Forest', y_prob_rf_test, '#2ECC71')
    ]:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        plt.plot(fpr, tpr, color=color, lw=2.5, label=f'{name} (AUC = {auc:.3f})')
        
    plt.xlabel('Tasa de Falsos Positivos (FPR)')
    plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
    plt.title(f'Comparación de Curvas ROC ({nombre_materia})', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'static/charts/07_curva_roc_{prefix}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Matrices de confusión
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    models_test_info = [
        ('Baseline', y_pred_base_test),
        ('Árbol de Decisión', y_pred_dt_test),
        ('Random Forest', y_pred_rf_test)
    ]
    for ax, (name, y_pred) in zip(axes, models_test_info):
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Reprobado', 'Aprobado'])
        disp.plot(ax=ax, cmap='Blues', values_format='d')
        ax.grid(False)
        ax.set_title(f'Matriz de Confusión\n{name}', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'static/charts/08_matrices_confusion_{prefix}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Guardar pickles específicos
    joblib.dump(rf, f'student_rf_model_{prefix}.pkl')
    joblib.dump(scaler, f'student_scaler_{prefix}.pkl')
    print(f"Modelos de {nombre_materia} exportados como student_rf_model_{prefix}.pkl y student_scaler_{prefix}.pkl")
    
    return feature_cols, cluster_profiles

# Ejecutar para Matemáticas
feature_cols_mat, profiles_mat = procesar_materia(mat, "Matemáticas", "mat")

# Ejecutar para Portugués
feature_cols_por, profiles_por = procesar_materia(por, "Portugués", "por")

# Guardar lista común de variables de entrada
joblib.dump(feature_cols_mat, 'student_feature_cols.pkl')
print("\n" + "=" * 80)
print("PIPELINE MULTI-MATERIA COMPLETADO CON EXITO.")
print("Todos los graficos guardados en la carpeta static/charts.")
print("=" * 80)
