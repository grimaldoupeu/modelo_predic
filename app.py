import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = FastAPI(title="Sistema de Analisis de Rendimiento Estudiantil")

static_dir = os.path.join(BASE_DIR, 'static')
os.makedirs(os.path.join(static_dir, 'charts'), exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

models = {}
scalers = {}
feature_cols = None

for prefix in ['mat', 'por']:
    try:
        models[prefix] = joblib.load(os.path.join(BASE_DIR, f'student_rf_model_{prefix}.pkl'))
        scalers[prefix] = joblib.load(os.path.join(BASE_DIR, f'student_scaler_{prefix}.pkl'))
        print(f"Modelo {prefix.upper()} cargado correctamente.")
    except Exception as e:
        print(f"Error al cargar modelo {prefix}: {e}")

try:
    feature_cols = joblib.load(os.path.join(BASE_DIR, 'student_feature_cols.pkl'))
    print("Feature columns cargadas correctamente.")
except Exception as e:
    print(f"Error al cargar feature_cols: {e}")

MAT_CSV = os.path.join(BASE_DIR, 'student-mat.csv')
POR_CSV = os.path.join(BASE_DIR, 'student-por.csv')

def compute_metrics(csv_path):
    df = pd.read_csv(csv_path, sep=';')
    total = len(df)
    desertion = int((df['G3'] == 0).sum())
    active = total - desertion
    df_active = df[df['G3'] > 0].copy()
    aprobados = int((df_active['G3'] >= 10).sum())
    tasa_aprobacion = round((aprobados / active) * 100, 1) if active > 0 else 0.0
    return {
        "total": total,
        "active": active,
        "desertion": desertion,
        "aprobados": aprobados,
        "tasa_aprobacion": tasa_aprobacion
    }

metrics_cache = {}
if os.path.exists(MAT_CSV):
    metrics_cache['mat'] = compute_metrics(MAT_CSV)
    m = metrics_cache['mat']
    print(f"Métricas MAT: {m['total']} total, {m['active']} activos, {m['desertion']} deserción, {m['tasa_aprobacion']}% aprobación")
if os.path.exists(POR_CSV):
    metrics_cache['por'] = compute_metrics(POR_CSV)
    m = metrics_cache['por']
    print(f"Métricas POR: {m['total']} total, {m['active']} activos, {m['desertion']} deserción, {m['tasa_aprobacion']}% aprobación")

class StudentData(BaseModel):
    materia: str = "mat"
    age: float = 16.0
    Medu: float = 2.0
    Fedu: float = 2.0
    traveltime: float = 1.0
    studytime: float = 2.0
    failures: float = 0.0
    schoolsup: float = 0.0
    famsup: float = 1.0
    paid: float = 0.0
    activities: float = 1.0
    nursery: float = 1.0
    higher: float = 1.0
    internet: float = 1.0
    romantic: float = 0.0
    famrel: float = 4.0
    freetime: float = 3.0
    goout: float = 3.0
    Dalc: float = 1.0
    Walc: float = 2.0
    health: float = 3.0
    absences: float = 4.0
    school_enc: float = 0.0
    sex_enc: float = 0.0
    address_enc: float = 1.0
    famsize_enc: float = 0.0
    Pstatus_enc: float = 1.0
    Mjob_enc: float = 2.0
    Fjob_enc: float = 2.0
    reason_enc: float = 1.0
    guardian_enc: float = 1.0

MATERIA_NOMBRES = {'mat': 'Matemáticas', 'por': 'Portugués'}

@app.get("/", response_class=HTMLResponse)
def get_dashboard():
    index_path = os.path.join(BASE_DIR, "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.get("/api/metrics/{materia}")
def get_metrics(materia: str):
    if materia not in metrics_cache:
        raise HTTPException(status_code=404, detail=f"Materia '{materia}' no encontrada. Usa 'mat' o 'por'.")
    data = metrics_cache[materia].copy()
    data['materia_nombre'] = MATERIA_NOMBRES.get(materia, materia)
    return data

@app.post("/predict")
def predict_student(data: StudentData):
    materia = data.materia
    if materia not in models:
        raise HTTPException(status_code=400, detail=f"Modelo para materia '{materia}' no disponible.")
    if feature_cols is None:
        raise HTTPException(status_code=500, detail="Feature columns no cargadas.")
    rf = models[materia]
    scaler = scalers[materia]
    try:
        student_dict = data.model_dump()
        student_dict.pop('materia')
        df_input = pd.DataFrame([student_dict])
        df_input = df_input[feature_cols]
        X_scaled = scaler.transform(df_input)
        prediction = int(rf.predict(X_scaled)[0])
        probabilities = rf.predict_proba(X_scaled)[0]
        prob_aprobado = float(probabilities[1])
        prob_reprobado = float(probabilities[0])
        materia_nombre = MATERIA_NOMBRES.get(materia, materia)
        return {
            "prediction": prediction,
            "prediction_label": "Aprobado" if prediction == 1 else "Reprobado",
            "prob_aprobado": prob_aprobado,
            "prob_reprobado": prob_reprobado,
            "materia": materia_nombre,
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("Levantando servidor local en http://127.0.0.1:8000 ...")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
