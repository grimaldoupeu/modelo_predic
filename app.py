import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = FastAPI(title="Sistema de Analisis de Rendimiento Estudiantil")

# Montar carpeta static para servir graficos
static_dir = os.path.join(BASE_DIR, 'static')
os.makedirs(os.path.join(static_dir, 'charts'), exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Cargar modelos y transformaciones con rutas absolutas
try:
    rf = joblib.load(os.path.join(BASE_DIR, 'student_rf_model.pkl'))
    scaler = joblib.load(os.path.join(BASE_DIR, 'student_scaler.pkl'))
    feature_cols = joblib.load(os.path.join(BASE_DIR, 'student_feature_cols.pkl'))
    print("Modelos cargados correctamente en la API.")
except Exception as e:
    print(f"Error al cargar los modelos: {e}. Por favor ejecuta primero 'examen_completo.py'")


# Definir la estructura de datos que recibe la API para prediccion
class StudentData(BaseModel):
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

@app.get("/", response_class=HTMLResponse)
def get_dashboard():
    index_path = os.path.join(BASE_DIR, "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.post("/predict")
def predict_student(data: StudentData):
    try:
        # Convertir datos a DataFrame ordenado por las columnas de features correctas
        student_dict = data.dict()
        df_input = pd.DataFrame([student_dict])
        
        # Seleccionar y ordenar las columnas exactamente como en el entrenamiento
        df_input = df_input[feature_cols]
        
        # Escalar datos
        X_scaled = scaler.transform(df_input)
        
        # Predecir clase y probabilidad
        prediction = int(rf.predict(X_scaled)[0])
        probabilities = rf.predict_proba(X_scaled)[0]
        prob_aprobado = float(probabilities[1])
        prob_reprobado = float(probabilities[0])
        
        return {
            "prediction": prediction,
            "prediction_label": "Aprobado" if prediction == 1 else "Reprobado",
            "prob_aprobado": prob_aprobado,
            "prob_reprobado": prob_reprobado,
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("Levantando servidor local en http://127.0.0.1:8000 ...")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
