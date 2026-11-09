
from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI
from app.schemas import PassengerInput, PredictionResponse

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "champion_model.joblib"

model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="Titanic ML Inference API",
    description="Predict Titanic passenger survival",
    version="1.0.0"
)

@app.get("/")
def home():
    return {
        "message": "Titanic ML Inference API is running",
        "status": "healthy"
    }

@app.post("/predict", response_model=PredictionResponse)
def predict_passenger(passenger: PassengerInput):
    input_data = pd.DataFrame([passenger.model_dump()])

    prediction = int(model.predict(input_data)[0])
    probabilities = model.predict_proba(input_data)[0]

    probability_map = {
        int(label): float(probability)
        for label, probability in zip(model.classes_, probabilities)
    }

    return {
        "prediction": prediction,
        "prediction_label": "Survived" if prediction == 1 else "Did not survive",
        "survival_probability": probability_map.get(1, 0.0),
        "non_survival_probability": probability_map.get(0, 0.0)
    }
