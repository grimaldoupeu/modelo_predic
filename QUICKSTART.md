# ⚡ Quick Start - Guía de Inicio Rápido

## 1️⃣ Instalación (5 minutos)

```bash
# Clonar repositorio
git clone <tu-repositorio-url>
cd examen_U2

# Crear entorno virtual
python -m venv venv

# Activar entorno
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

## 2️⃣ Opción A: Ejecutar Todo (Pipeline Completo)

```bash
python examen_completo.py
```

**Genera:**
- 📊 Análisis exploratorio
- 🔍 Gráficos de segmentación
- 🤖 Modelos entrenados (*.pkl)
- 📈 Reportes de evaluación

**Tiempo:** ~2 minutos

## 3️⃣ Opción B: Usar el Dashboard Interactivo

```bash
python app.py
```

**Luego abre:** `http://localhost:8000`

**Funcionalidades:**
- 📊 Dashboard de métricas
- 🎯 Simulador de predicción
- 📈 Gráficos comparativos

## 4️⃣ Opción C: Análisis en Notebook

```bash
jupyter notebook examen_estudiantes.ipynb
```

**Permite:**
- Explorar datos paso a paso
- Visualizar gráficos
- Modificar análisis

## 📋 Estructura Básica

```
examen_U2/
├── app.py                    ← API Dashboard
├── examen_completo.py        ← Pipeline completo
├── student-mat.csv           ← Datos Matemáticas
├── student-por.csv           ← Datos Portugués
├── requirements.txt          ← Dependencias
└── README.md                 ← Este archivo
```

## 🎯 Lo Más Importante

### Para ejecutar modelos:
```bash
python examen_completo.py
```

### Para ver resultados en web:
```bash
python app.py
# Abre: http://localhost:8000
```

### Para explorar datos:
```bash
jupyter notebook examen_estudiantes.ipynb
```

## ✅ Verificar que todo funciona

```bash
# Test 1: Verificar Python
python --version

# Test 2: Verificar dependencias
pip list | grep -E "fastapi|pandas|scikit"

# Test 3: Ejecutar pipeline
python examen_completo.py
```

## 🚨 Errores Comunes

| Error | Solución |
|-------|----------|
| `ModuleNotFoundError` | `pip install -r requirements.txt` |
| `Port 8000 already in use` | Cambiar puerto en app.py |
| `No module named 'venv'` | `python -m venv venv` |

## 📚 Próximos Pasos

1. ✅ Clonar repositorio
2. ✅ Instalar dependencias
3. ✅ Ejecutar `python examen_completo.py`
4. ✅ Ejecutar `python app.py`
5. ✅ Abrir `http://localhost:8000`

---

**¡Ya estás listo!** Para más detalles, consulta [README.md](README.md) y [DEVELOPMENT.md](DEVELOPMENT.md)
