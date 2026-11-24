import os

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from .schemas import (
    HeartDiseaseInput,
    HealthResponse,
    InfoResponse,
    PredictionResponse,
)

# --- App setup ---
app = FastAPI(
    title="Heart Disease Prediction API",
    description=(
        "A FastAPI service that predicts the presence of heart disease "
        "using a machine learning classifier trained on the UCI Heart Disease dataset."
    ),
    version="1.0.0",
)

# --- Load model at startup ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "heart_model.joblib")

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    model = None
    print(f"WARNING: Could not load model from {MODEL_PATH}: {e}")

FEATURE_LIST = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]


# --- Endpoints ---

@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """Check if the service is running and the model is loaded."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return HealthResponse(status="healthy")


@app.get("/info", response_model=InfoResponse, tags=["Info"])
def model_info():
    """Return information about the loaded model."""
    model_type = type(model).__name__ if model else "Unknown"
    return InfoResponse(
        model_type=model_type,
        feature_list=FEATURE_LIST,
        dataset="UCI Heart Disease Dataset",
        description=(
            "Binary classifier predicting the presence or absence of heart disease "
            "based on 13 clinical features."
        ),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(input_data: HeartDiseaseInput):
    """Predict whether a patient has heart disease."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    features = pd.DataFrame([{
        "age": input_data.age,
        "sex": input_data.sex,
        "cp": input_data.cp,
        "trestbps": input_data.trestbps,
        "chol": input_data.chol,
        "fbs": input_data.fbs,
        "restecg": input_data.restecg,
        "thalach": input_data.thalach,
        "exang": input_data.exang,
        "oldpeak": input_data.oldpeak,
        "slope": input_data.slope,
        "ca": input_data.ca,
        "thal": input_data.thal,
    }])

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]

    return PredictionResponse(
        heart_disease=bool(prediction),
        confidence=float(max(probability)),
    )
