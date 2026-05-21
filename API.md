# 🔌 Documentación de API

## Visión General

API REST construida con **FastAPI** para predicción en tiempo real del rendimiento estudiantil.

**Base URL**: `http://localhost:8000`

## Endpoints Disponibles

### 1. Dashboard Principal

```http
GET /
```

**Descripción**: Retorna la página HTML del dashboard interactivo

**Response**:
- Content-Type: `text/html`
- Status: 200

**Ejemplo**:
```bash
curl http://localhost:8000/
```

---

### 2. Obtener Métricas de Materia

```http
GET /api/metrics/{subject}
```

**Descripción**: Obtiene métricas consolidadas de una materia específica

**Parámetros**:
- `subject` (path): `mat` (Matemáticas) o `por` (Portugués)

**Response**:
```json
{
  "subject": "mat",
  "total_students": 395,
  "active_students": 369,
  "desertion_count": 26,
  "pass_rate": 74.5,
  "fail_students": 101,
  "pass_students": 294
}
```

**Ejemplos**:
```bash
# Matemáticas
curl http://localhost:8000/api/metrics/mat

# Portugués
curl http://localhost:8000/api/metrics/por
```

**Status**:
- 200 OK: Métricas obtenidas
- 400 Bad Request: Materia inválida

---

### 3. Predicción Individual

```http
POST /api/predict
```

**Descripción**: Realiza una predicción de rendimiento para un estudiante

**Content-Type**: `application/json`

**Request Body**:
```json
{
  "subject": "mat",
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

**Response**:
```json
{
  "prediction": 1,
  "prediction_text": "APROBADO",
  "probability_pass": 0.8940,
  "probability_fail": 0.1060,
  "risk_level": "Bajo"
}
```

**Campos de Response**:
- `prediction`: 1 = Aprobará, 0 = Reprobará
- `prediction_text`: Interpretación en texto
- `probability_pass`: Probabilidad de aprobación (0.0-1.0)
- `probability_fail`: Probabilidad de reprobación (0.0-1.0)
- `risk_level`: Categoría de riesgo (Bajo, Medio, Alto)

**Ejemplo cURL**:
```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "subject": "mat",
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
  }'
```

**Ejemplo Python**:
```python
import requests
import json

url = "http://localhost:8000/api/predict"
data = {
    "subject": "mat",
    "school": "GP",
    "sex": "M",
    "age": 18,
    # ... resto de campos
}

response = requests.post(url, json=data)
result = response.json()
print(f"Predicción: {result['prediction_text']}")
print(f"Probabilidad de aprobación: {result['probability_pass']*100:.1f}%")
```

**Ejemplo JavaScript**:
```javascript
const data = {
    subject: "mat",
    school: "GP",
    sex: "M",
    age: 18,
    // ... resto de campos
};

fetch('http://localhost:8000/api/predict', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
    },
    body: JSON.stringify(data)
})
.then(response => response.json())
.then(result => {
    console.log(`Predicción: ${result.prediction_text}`);
    console.log(`Probabilidad: ${(result.probability_pass * 100).toFixed(1)}%`);
});
```

**Status**:
- 200 OK: Predicción realizada correctamente
- 400 Bad Request: Datos inválidos o incompletos
- 500 Internal Server Error: Error en el servidor

---

## Modelos de Datos

### Subject
```
"mat"  - Matemáticas
"por"  - Portugués
```

### Valores Válidos para Features Categóricas

**school**:
- `"GP"` - Gabriel Pereira
- `"MS"` - Mousinho da Silveira

**sex**:
- `"M"` - Masculino
- `"F"` - Femenino

**address**:
- `"U"` - Urbano
- `"R"` - Rural

**famsize**:
- `"LE3"` - Familia pequeña (≤3)
- `"GT3"` - Familia grande (>3)

**Pstatus**:
- `"A"` - Vivir apartados
- `"T"` - Vivir juntos

**Mjob / Fjob** (Ocupación de madre/padre):
- `"teacher"` - Profesor
- `"health"` - Sector salud
- `"services"` - Servicios
- `"at_home"` - En casa
- `"other"` - Otro

**reason** (Razón de elección de escuela):
- `"course"` - Elección del curso
- `"other"` - Otra razón
- `"home"` - Cercanía al hogar
- `"reputation"` - Reputación

**guardian**:
- `"mother"` - Madre
- `"father"` - Padre
- `"other"` - Otro

**yes/no fields**:
- `"yes"` o `"no"` para: `schoolsup`, `famsup`, `paid`, `activities`, `nursery`, `higher`, `internet`, `romantic`

### Valores Válidos para Features Numéricos

**age**: 15-22
**Medu/Fedu** (Educación madre/padre): 0-4
- 0 = sin educación
- 1 = educación primaria
- 2 = educación 5-9 años
- 3 = educación secundaria
- 4 = educación superior

**traveltime**: 1-4 (minutos)
**studytime**: 1-4 (horas semanales)
**failures**: 0-4 (reprobaciones previas)
**freetime**: 1-5 (escala 1-5)
**goout**: 1-5 (escala 1-5)
**Dalc**: 1-5 (consumo alcohol entre semana)
**Walc**: 1-5 (consumo alcohol fin de semana)
**health**: 1-5 (estado de salud)
**absences**: 0-93 (días de ausencia)

---

## Códigos de Estado HTTP

| Status | Descripción | Cuándo |
|--------|-------------|--------|
| 200 | OK | Solicitud exitosa |
| 400 | Bad Request | Datos inválidos o faltantes |
| 404 | Not Found | Endpoint no existe |
| 422 | Unprocessable Entity | Validación de datos fallida |
| 500 | Internal Server Error | Error del servidor |

---

## Rate Limiting

Por defecto: **Sin límite** (ajustar en producción)

---

## Autenticación

Actualmente: **Sin autenticación** (considerar implementar en producción)

---

## CORS (Cross-Origin Resource Sharing)

Configuración actual: **Permitir todos los orígenes**

Ajustar para producción:
```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://tudominio.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## Ejemplos Completos

### Ejemplo 1: Predicción Exitosa

**Request**:
```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "subject": "mat",
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
    "studytime": 3,
    "failures": 0,
    "schoolsup": "yes",
    "famsup": "yes",
    "paid": "no",
    "activities": "yes",
    "nursery": "yes",
    "higher": "yes",
    "internet": "yes",
    "romantic": "no",
    "freetime": 4,
    "goout": 3,
    "Dalc": 1,
    "Walc": 1,
    "health": 5,
    "absences": 2
  }'
```

**Response**:
```json
{
  "prediction": 1,
  "prediction_text": "APROBADO",
  "probability_pass": 0.9240,
  "probability_fail": 0.0760,
  "risk_level": "Bajo"
}
```

### Ejemplo 2: Predicción de Riesgo

**Cambios en request**:
```json
{
  ...
  "studytime": 1,
  "failures": 2,
  "schoolsup": "no",
  "famsup": "no",
  "Walc": 5,
  "absences": 15,
  ...
}
```

**Response**:
```json
{
  "prediction": 0,
  "prediction_text": "REPROBACIÓN DETECTADA",
  "probability_pass": 0.2840,
  "probability_fail": 0.7160,
  "risk_level": "Alto"
}
```

---

## Testing de API

### Con Postman
1. Importar colección de endpoints
2. Configurar variables de ambiente
3. Ejecutar tests

### Con Python requests
```python
import requests

base_url = "http://localhost:8000"

# Test 1: Métricas
response = requests.get(f"{base_url}/api/metrics/mat")
assert response.status_code == 200
print(response.json())

# Test 2: Predicción
data = {...}
response = requests.post(f"{base_url}/api/predict", json=data)
assert response.status_code == 200
print(response.json())
```

---

## Troubleshooting

### Error: Connection refused
- Verificar que el servidor está ejecutándose: `python app.py`
- Verificar el puerto: `http://localhost:8000`

### Error: Invalid field
- Revisar nombres de campos exactos
- Revisar valores válidos para campos categóricos

### Error: Internal server error (500)
- Revisar logs del servidor
- Verificar que los archivos `.pkl` existen
- Ejecutar: `python examen_completo.py`

---

**Última actualización**: 2024
**Versión API**: 1.0
**Status**: Activa ✅
