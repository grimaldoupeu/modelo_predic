# 🔧 Guía de Desarrollo

## Arquitectura del Proyecto

### Flujo de Datos

```
CSV Data (student-mat.csv, student-por.csv)
    ↓
examen_completo.py (Pipeline Principal)
    ├─ parte2_particion_baseline.py (Partición de datos 80/20, modelos baseline)
    ├─ parte3_segmentacion.py (K-Means, Clustering Jerárquico)
    ├─ parte4_clasificacion.py (Decision Tree, Random Forest)
    └─ parte5_evaluacion.py (Métricas y comparación)
    ↓
Modelos entrenados (*.pkl)
    ↓
app.py (API FastAPI)
    ↓
index.html (Dashboard Frontend)
```

## Descripción de Archivos

### Archivo Principal

**`examen_completo.py`** - Pipeline completo automatizado
- Carga de datos
- Validación y limpieza
- Partición entrenamiento/test (80/20)
- Entrenamiento de modelos
- Evaluación comparativa
- Generación de gráficos

### Componentes Modularizados

**`parte2_particion_baseline.py`**
- División de datos
- Escalamiento (StandardScaler)
- Modelos baseline (DummyClassifier)
- Almacenamiento de modelos

**`parte3_segmentacion.py`**
- K-Means clustering
- Clustering jerárquico
- Análisis de silueta
- Dendrogramas

**`parte4_clasificacion.py`**
- Entrenamiento de clasificadores
- Optimización de parámetros
- Matrices de confusión
- Curvas ROC

**`parte5_evaluacion.py`**
- Comparación de modelos
- Reportes de desempeño
- Feature importance
- Recomendaciones

### API y Frontend

**`app.py`**
- Endpoints FastAPI
- Carga de modelos
- Predicciones en tiempo real
- Cálculo de métricas

**`index.html`**
- Dashboard interactivo
- Formulario de predicción
- Visualización de gráficos

## Workflow de Ejecución

### Para Desarrollo
```bash
# 1. Entrenar modelos
python examen_completo.py

# 2. Ejecutar API
python app.py

# 3. Abrir en navegador
# http://localhost:8000
```

### Para Análisis Exploratorio
```bash
jupyter notebook examen_estudiantes.ipynb
```

## Variables de Característica

### Features Utilizados (después de encoding)
```python
numerical_features = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 
                      'failures', 'famrel', 'freetime', 'goout', 
                      'Dalc', 'Walc', 'health', 'absences']

categorical_features = ['school', 'sex', 'address', 'famsize', 'Pstatus', 
                        'Mjob', 'Fjob', 'reason', 'guardian', 'schoolsup', 
                        'famsup', 'paid', 'activities', 'nursery', 'higher', 
                        'internet', 'romantic']
```

### Variable Objetivo
- **G3** (nota final): 0-20
- **Clasificación**: 0-10 (Reprobado) vs 10+ (Aprobado)

## Manejo de Datos

### Limpieza
- Detección de valores faltantes
- Eliminación de duplicados
- Validación de rangos

### Balance de Clases
- Problema: ~85% aprobados, ~15% reprobados
- Solución: `class_weight='balanced'` en Random Forest
- Efecto: Mejor recall en clase minoritaria

### Detección de Leakage
- G1, G2 son notas de períodos anteriores
- Eliminar para evitar información del futuro
- Usar solo características socioeconómicas y demográficas

## Parámetros de Modelos

### Random Forest (Mejor Modelo)
```python
RandomForestClassifier(
    n_estimators=150,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
```

### K-Means Clustering
```python
KMeans(
    n_clusters=3,  # Óptimo según silueta
    random_state=42,
    n_init=10
)
```

## Métricas de Desempeño

### Train Set (85%)
- Accuracy: ~90%
- AUC-ROC: ~0.95

### Test Set (15%)
- Accuracy: 75.93%
- AUC-ROC: 0.7607
- F1-Score: 0.8506
- Recall: 0.9250

## Estructura de Predicción

### Input (Formulario)
```json
{
  "school": "GP",
  "sex": "M",
  "age": 18,
  "address": "U",
  "famsize": "LE3",
  "Pstatus": "A",
  "Medu": 4,
  "Fedu": 4,
  "Mjob": "teacher",
  "Fjob": "teacher",
  "reason": "course",
  "guardian": "mother",
  "traveltime": 2,
  "studytime": 2,
  "failures": 0,
  "schoolsup": "yes",
  "famsup": "no",
  "paid": "no",
  "activities": "yes",
  "nursery": "yes",
  "higher": "yes",
  "internet": "yes",
  "romantic": "no",
  "freetime": 3,
  "goout": 4,
  "Dalc": 1,
  "Walc": 1,
  "health": 5,
  "absences": 0
}
```

### Output (Predicción)
```json
{
  "prediction": 1,
  "prediction_text": "APROBADO",
  "probability_pass": 0.8940,
  "probability_fail": 0.1060,
  "risk_level": "Bajo"
}
```

## Testing

### Recomendaciones para Tests
```python
# test_models.py
def test_model_loading():
    assert models['mat'] is not None
    assert models['por'] is not None

def test_prediction():
    # Crear datos de prueba
    # Verificar predicción
    pass

def test_metrics_consistency():
    # Validar que métricas estén en rango [0,1]
    pass
```

## Despliegue a Producción

### Requerimientos
1. Servidor Linux/Windows con Python 3.8+
2. Dependencias instaladas: `pip install -r requirements.txt`
3. Modelos entrenados en `./` (archivos .pkl)

### Ejecución
```bash
# Desarrollo
python app.py

# Producción (con Gunicorn)
pip install gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker app:app --bind 0.0.0.0:8000
```

### Docker (Opcional)
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Troubleshooting

### Error: ModuleNotFoundError
```bash
pip install -r requirements.txt
```

### Error: Modelo no encontrado
- Verificar que los archivos `.pkl` existan
- Ejecutar `python examen_completo.py` para entrenar

### Error de Puerto en uso
```bash
# Cambiar puerto en app.py
uvicorn.run(app, host="0.0.0.0", port=8001)
```

## Optimizaciones Futuras

- [ ] Incluir validación cruzada estratificada (k-fold)
- [ ] Añadir feature selection automático
- [ ] Implementar ensembles con múltiples modelos
- [ ] Crear tests unitarios
- [ ] Documentar endpoints Swagger
- [ ] Añadir autenticación a la API
- [ ] Cache de predicciones
- [ ] Monitoreo de performance en producción

