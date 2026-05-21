# =============================================================================
# PARTE 3: SEGMENTACIÓN (CLUSTERING)
# =============================================================================

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, silhouette_samples
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.cm as cm

print("\n" + "=" * 70)
print("5. SEGMENTACIÓN - CLUSTERING")
print("=" * 70)

# Seleccionar features numéricas para clustering
num_features = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures',
                'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences']
X_cluster = df[num_features].copy()
X_cluster_scaled = StandardScaler().fit_transform(X_cluster)

# === 5.1 K-MEANS CON MÉTODO DEL CODO ===
print("\n📊 5.1 K-Means Clustering")
print("-" * 40)

inertias = []
silhouettes = []
K_range = range(2, 11)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_cluster_scaled)
    inertias.append(km.inertia_)
    sil = silhouette_score(X_cluster_scaled, labels)
    silhouettes.append(sil)
    print(f"  K={k}: Inercia={km.inertia_:.1f}, Silhouette={sil:.4f}")

# Gráficos: Codo y Silhouette
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
axes[0].set_xlabel('Número de Clústeres (K)')
axes[0].set_ylabel('Inercia')
axes[0].set_title('Método del Codo - K-Means')
axes[0].grid(True, alpha=0.3)

axes[1].plot(K_range, silhouettes, 'rs-', linewidth=2, markersize=8)
axes[1].set_xlabel('Número de Clústeres (K)')
axes[1].set_ylabel('Silhouette Score')
axes[1].set_title('Índice Silhouette vs K')
axes[1].grid(True, alpha=0.3)

best_k = list(K_range)[np.argmax(silhouettes)]
axes[1].axvline(x=best_k, color='green', linestyle='--', label=f'Mejor K={best_k}')
axes[1].legend()

plt.tight_layout()
plt.savefig('02_codo_silhouette.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"\n✅ K óptimo según Silhouette: {best_k}")

# Aplicar K-Means con K óptimo
kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df['cluster_kmeans'] = kmeans_final.fit_predict(X_cluster_scaled)

print(f"\nDistribución de clústeres K-Means (K={best_k}):")
print(df['cluster_kmeans'].value_counts().sort_index().to_string())

# === SILHOUETTE SCORE - EXPLICACIÓN ===
print(f"""
📖 ÍNDICE SILHOUETTE:
────────────────────────────────────────────────────────────
El índice Silhouette mide qué tan bien cada punto pertenece a 
su clúster comparado con los clústeres vecinos.

• Rango: [-1, 1]
• Cercano a 1: El punto está bien asignado a su clúster
• Cercano a 0: El punto está en la frontera entre clústeres
• Cercano a -1: El punto probablemente está mal asignado

Silhouette promedio obtenido: {silhouette_score(X_cluster_scaled, df['cluster_kmeans']):.4f}
────────────────────────────────────────────────────────────
""")

# === 5.2 CLUSTERING JERÁRQUICO ===
print("📊 5.2 Clustering Jerárquico")
print("-" * 40)

# Dendrograma
Z = linkage(X_cluster_scaled, method='ward')

fig, ax = plt.subplots(figsize=(14, 6))
dendrogram(Z, truncate_mode='lastp', p=30, leaf_rotation=90,
           leaf_font_size=10, show_contracted=True, ax=ax)
ax.set_title('Dendrograma - Clustering Jerárquico (Ward)', fontsize=14)
ax.set_xlabel('Muestras')
ax.set_ylabel('Distancia')
ax.axhline(y=50, color='red', linestyle='--', label='Corte sugerido')
ax.legend()
plt.tight_layout()
plt.savefig('03_dendrograma.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Gráfico guardado: 03_dendrograma.png")

# Aplicar clustering jerárquico con mismo K
hc = AgglomerativeClustering(n_clusters=best_k, linkage='ward')
df['cluster_jerarquico'] = hc.fit_predict(X_cluster_scaled)

sil_hc = silhouette_score(X_cluster_scaled, df['cluster_jerarquico'])
sil_km = silhouette_score(X_cluster_scaled, df['cluster_kmeans'])

print(f"\nComparación de Silhouette Scores:")
print(f"  K-Means:              {sil_km:.4f}")
print(f"  Clustering Jerárquico: {sil_hc:.4f}")

# === 5.3 INTERPRETACIÓN DE PERFILES ===
print("\n📊 5.3 Interpretación de Perfiles de Estudiantes")
print("-" * 40)

profile = df.groupby('cluster_kmeans')[num_features + ['aprobado']].mean()
print("\nPerfiles promedio por clúster:")
print(profile.round(2).to_string())

# Heatmap de perfiles
fig, ax = plt.subplots(figsize=(14, 5))
profile_norm = (profile - profile.mean()) / profile.std()
sns.heatmap(profile_norm.T, annot=True, fmt='.2f', cmap='RdYlGn',
            center=0, ax=ax, linewidths=0.5)
ax.set_title('Perfiles de Estudiantes por Clúster (valores normalizados)', fontsize=13)
ax.set_xlabel('Clúster')
ax.set_ylabel('Características')
plt.tight_layout()
plt.savefig('04_perfiles_cluster.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Gráfico guardado: 04_perfiles_cluster.png")

# Interpretación automática
print("\n📝 INTERPRETACIÓN DE CLÚSTERES:")
print("─" * 50)
for c in range(best_k):
    p = profile.loc[c]
    print(f"\n🔹 Clúster {c} ({(df['cluster_kmeans']==c).sum()} estudiantes):")
    if p['studytime'] > profile['studytime'].mean():
        print(f"   • Mayor tiempo de estudio ({p['studytime']:.1f})")
    if p['failures'] > profile['failures'].mean():
        print(f"   • Más fracasos previos ({p['failures']:.1f})")
    if p['goout'] > profile['goout'].mean():
        print(f"   • Mayor vida social ({p['goout']:.1f})")
    if p['Dalc'] > profile['Dalc'].mean() or p['Walc'] > profile['Walc'].mean():
        print(f"   • Mayor consumo de alcohol (D:{p['Dalc']:.1f}, W:{p['Walc']:.1f})")
    if p['absences'] > profile['absences'].mean():
        print(f"   • Más ausencias ({p['absences']:.1f})")
    print(f"   • Tasa de aprobación: {p['aprobado']*100:.1f}%")
