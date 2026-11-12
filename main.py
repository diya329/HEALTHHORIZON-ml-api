
import os
import joblib
import pandas as pd

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Load trained models
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

classifier = joblib.load(
    os.path.join(MODEL_DIR, "healthshield_classifier.pkl")
)

regressor = joblib.load(
    os.path.join(MODEL_DIR, "healthshield_regressor.pkl")
)

scaler = joblib.load(
    os.path.join(MODEL_DIR, "healthshield_scaler.pkl")
)

label_encoder = joblib.load(
    os.path.join(MODEL_DIR, "healthshield_label_encoder.pkl")
)

features = [
    "temperature",
    "rainfall",
    "humidity",
    "water_supply_interruptions",
    "mosquito_activity_index",
    "fever_cases",
    "diarrhoea_cases",
    "vomiting_cases",
    "population_density",
    "historical_avg_fever"
]

app = FastAPI(
    title="HEALTHSHIELD ML API",
    description="Neighbourhood Health Early-Warning Network",
    version="1.0"
)

# Temporary demo configuration.
# Restrict this to your actual dashboard origin before production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)


class NeighbourhoodData(BaseModel):
    neighbourhood_id: str = "NH_NEW"
    temperature: float
    rainfall: float
    humidity: float
    water_supply_interruptions: int
    mosquito_activity_index: float
    fever_cases: int
    diarrhoea_cases: int
    vomiting_cases: int
    population_density: float
    historical_avg_fever: float


priority_map = {
    "Low": "P4 - Routine Monitoring",
    "Moderate": "P3 - Advisory Issued",
    "High": "P2 - Mobile Health Team Dispatch",
    "Critical": "P1 - Immediate Intervention"
}

action_map = {
    "Low": "Continue standard municipal surveillance.",
    "Moderate": "Test local water chlorination levels and distribute hygiene information.",
    "High": "Investigate water quality and drainage; arrange public-health assessment.",
    "Critical": "Urgent public-health investigation and emergency response assessment."
}


@app.get("/")
def home():
    return {
        "message": "HEALTHSHIELD ML API is running!",
        "status": "online"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "models_loaded": True
    }


@app.post("/predict")
def predict(data: NeighbourhoodData):

    input_data = data.model_dump()

    input_df = pd.DataFrame(
        [[input_data[f] for f in features]],
        columns=features
    )

    scaled_data = scaler.transform(input_df)

    prediction = classifier.predict(scaled_data)[0]
    category = label_encoder.inverse_transform(
        [int(prediction)]
    )[0]

    score = float(regressor.predict(scaled_data)[0])
    score = max(0.0, min(100.0, score))

    return {
        "neighbourhood_id": data.neighbourhood_id,
        "predicted_risk_category": str(category),
        "risk_score": round(score, 2),
        "investigation_priority": priority_map[str(category)],
        "recommended_action": action_map[str(category)],
        "prediction_source": "trained_ml_models",
        "data_type": "synthetic_demo"
    }
