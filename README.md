# 📊 Sistema de Análisis de Rendimiento Estudiantil - Minería de Datos U2

Un sistema inteligente de análisis y predicción de rendimiento académico basado en minería de datos, machine learning y técnicas avanzadas de clustering y clasificación.

![Rendimiento U2](https://img.shields.io/badge/Sistema-Activo-brightgreen) ![Python](https://img.shields.io/badge/Python-3.8+-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-Backend-orange) ![ML](https://img.shields.io/badge/Machine%20Learning-Clasificación%20%26%20Clustering-blueviolet)

## 🎯 Descripción General

Este proyecto analiza el rendimiento de estudiantes en dos materias (Matemáticas y Portugués) utilizando características socioeconómicas y hábitos personales. Implementa un pipeline completo de minería de datos que incluye:

- **Análisis Exploratorio de Datos (EDA)**
- **Segmentación de Estudiantes** mediante clustering (K-Means, Clustering Jerárquico)
- **Modelos Predictivos** para clasificación de rendimiento
- **Dashboard Interactivo** para visualización de métricas y diagnósticos

### ✨ Características Principales

#### 1. **Dashboard de Monitoreo General**
- Visualización de métricas académicas por materia
- Monitoreo de tasa de aprobación y deserción
- Análisis de estudiantes activos

![Dashboard Portugués](docs/dashboard-portugues.png)

#### 2. **Modelos de Clasificación**
Evaluación comparativa de algoritmos con métricas detalladas:
- **Baseline (Dummy Classifier)**: Accuracy 74.07%, AUC 0.5000
- **Árbol de Decisión**: Accuracy 68.52%, AUC 0.5509
- **Random Forest (150 árboles)**: Accuracy 75.93%, AUC 0.7607 ✅

El modelo **Random Forest** se seleccionó como el mejor performer con superior capacidad discriminativa.

![Modelos de Clasificación](docs/modelos-clasificacion.png)

#### 3. **Simulador Predictor**
Herramienta interactiva para calcular el diagnóstico predictivo de estudiantes individuales:
- Ingreso de características socioeconómicas
- Predicción de riesgo de reprobación
- Probabilidades ajustadas según clase

![Simulador Predictor](docs/simulador-predictor.png)

#### 4. **Análisis de Estrategia de Examen**
- **Preprocesamiento Riguroso**: Eliminación de notas intermedias G1 y G2 (fuga de datos)
- **Balanceo de Clases**: Penalización de pesos para mitigar desbalance
- **Validación Robusta**: Test set (15% de datos) nunca visto por el modelo

## 📁 Estructura del Proyecto

```
├── app.py                          # API FastAPI principal
├── examen_completo.py             # Pipeline completo de minería de datos
├── parte2_particion_baseline.py   # Partición de datos y modelos baseline
├── parte3_segmentacion.py         # Análisis de clustering
├── parte4_clasificacion.py        # Modelos de clasificación
├── parte5_evaluacion.py           # Evaluación y comparación de modelos
│
├── student-mat.csv                # Dataset Matemáticas (395 estudiantes)
├── student-por.csv                # Dataset Portugués (649 estudiantes)
│
├── student_rf_model_mat.pkl       # Modelo Random Forest (Matemáticas)
├── student_rf_model_por.pkl       # Modelo Random Forest (Portugués)
├── student_scaler_mat.pkl         # Escalador StandardScaler (MAT)
├── student_scaler_por.pkl         # Escalador StandardScaler (POR)
├── student_feature_cols.pkl       # Columnas de características
│
├── index.html                     # Frontend interactivo
├── static/
│   └── charts/                    # Gráficos generados
│
├── examen_estudiantes.ipynb       # Notebook Jupyter completo
└── generate_notebook.py           # Generador de notebooks

```

## 🚀 Instalación

### Requisitos
- Python 3.8 o superior
- pip (gestor de paquetes)

### Pasos de Instalación

1. **Clonar el repositorio**
```bash
git clone <tu-repo-url>
cd examen_U2
```

2. **Crear entorno virtual (recomendado)**
```bash
python -m venv venv

# En Windows
venv\Scripts\activate

# En macOS/Linux
source venv/bin/activate
```

3. **Instalar dependencias**
```bash
pip install -r requirements.txt
```

### Dependencias Principales
```
fastapi==0.104.1
uvicorn==0.24.0
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.1
matplotlib==3.7.2
seaborn==0.12.2
joblib==1.3.1
scipy==1.11.2
```

## 📊 Uso

### Ejecutar el Pipeline Completo
```bash
python examen_completo.py
```
Genera:
- Análisis exploratorio
- Gráficos de segmentación
- Modelos entrenados (.pkl)
- Reporte de evaluación

### Ejecutar el Dashboard/API
```bash
python app.py
```
Acceder a: `http://localhost:8000`

Endpoints disponibles:
- `GET /` - Dashboard principal
- `GET /api/metrics/{subject}` - Métricas (mat/por)
- `POST /api/predict` - Predicción individual

### Ejecutar Componentes Específicos
```bash
# Solo análisis de partición y baseline
python parte2_particion_baseline.py

# Solo análisis de segmentación
python parte3_segmentacion.py

# Solo clasificación
python parte4_clasificacion.py

# Solo evaluación
python parte5_evaluacion.py
```

### Notebooks Jupyter
```bash
jupyter notebook examen_estudiantes.ipynb
```

## 📈 Datos

### Conjuntos de Datos
- **student-mat.csv**: 395 estudiantes, 33 variables (Matemáticas)
- **student-por.csv**: 649 estudiantes, 33 variables (Portugués)

### Características Incluidas
- **Demográficas**: edad, sexo, dirección, tamaño de familia
- **Familiares**: educación de padres, ocupación de padres
- **Socioeconómicas**: escuela, tipo de vivienda, razón de inscripción
- **Académicas**: período de estudio, notas previas
- **Personales**: consumo de alcohol, salidas con amigos, tiempo de estudio
- **Variable Objetivo**: G3 (nota final, 0-20)

## 🤖 Modelos Implementados

### Clustering
- **K-Means**: Segmentación no supervisada de estudiantes
- **Clustering Jerárquico**: Análisis dendrograma de similitud

### Clasificación
1. **DummyClassifier** - Baseline comparativo
2. **DecisionTreeClassifier** - Interpretabilidad
3. **RandomForestClassifier** (150 árboles) - Mejor performance ⭐

### Métricas de Evaluación
- Accuracy
- F1-Score
- ROC-AUC
- Matrices de confusión
- Curvas ROC

## 🔍 Hallazgos Clave

### Segmentación
- Identificación de **K clusters óptimos** mediante análisis de silueta
- Caracterización de grupos de estudiantes con patrones de comportamiento distintos

### Predicción
- **Random Forest supera baseline en discriminación** (AUC 0.7607 vs 0.5000)
- **Balanceo de clases mejora recall** en estudiantes con riesgo de reprobación
- **Eliminación de leakage** asegura predicciones válidas para próximas cohortes

## 📋 Decisiones Estratégicas

1. **Preprocesamiento Riguroso**: Eliminación de G1/G2 para evitar información del futuro
2. **Penalización de Pesos**: Manejo del desbalance de clases (mayoría aprobada)
3. **Validación Robusta**: Test set de 15% nunca visto en entrenamiento
4. **Multi-Materia**: Modelos separados para Matemáticas y Portugués
5. **API REST**: Deployment listo para producción

## 🛠️ Tecnologías

| Tecnología | Propósito |
|-----------|----------|
| **Python** | Lenguaje principal |
| **FastAPI** | Framework API web |
| **Scikit-learn** | Machine Learning |
| **Pandas** | Manipulación de datos |
| **Matplotlib/Seaborn** | Visualización |
| **Joblib** | Serialización de modelos |
| **HTML/CSS** | Frontend interactivo |

## 📝 Licencia

Este proyecto es de uso educativo.

## 👨‍💻 Autor

Proyecto de Examen - Minería de Datos Unidad 2

## 📞 Contacto

Para preguntas o sugerencias sobre el proyecto, crea un issue en el repositorio.

---

**Status**: ✅ Sistema Activo | **Última actualización**: 2024 | **Versión**: 1.0
