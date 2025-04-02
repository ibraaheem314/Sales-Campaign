from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import logging
from datetime import datetime

# Configuration
logger = logging.getLogger("uvicorn")
model = joblib.load("models/production_model.pkl")

app = FastAPI(title="Sales Conversion API")

class PredictionInput(BaseModel):
    call_time: int
    duration: float
    region: str
    product: str
    script_version: str
    day_of_week: str

@app.post("/predict")
async def predict(input: PredictionInput):
    try:
        features = pd.DataFrame([input.dict()])
        features['peak_hour'] = features['call_time'].between(14, 16)
        features['duration_min'] = features['duration'] / 60
        
        proba = model.predict_proba(features)[0][1]
        return {
            "conversion_probability": float(proba),
            "recommended_action": "prioritize" if proba > 0.3 else "standard"
        }
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_version": "1.2.0"
    }