from fastapi import FastAPI, Response
import joblib
import pandas as pd

from api.schemas import SurveyInput

app = FastAPI(
    title="Mental Health Prediction API",
    description="Predict whether a person is likely to seek mental health treatment.",
    version="1.0",
)

# Load trained model (Pipeline: preprocessor + classifier)
MODEL_PATH = "models/model.pkl"
model = joblib.load(MODEL_PATH)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.head("/health")
def health_head():
    return Response(status_code=200)

@app.post("/predict")
def predict(input_data: SurveyInput):

    # Convert Pydantic model → dict → DataFrame
    input_dict = input_data.dict()
    df = pd.DataFrame([input_dict])

    # Run model
    pred = model.predict(df)[0]
    proba = model.predict_proba(df)[0, 1]

    # Format response
    return {
        "prediction": int(pred),
        "probability_yes": float(proba),
        "model": "logistic-regression + preprocessing pipeline"
    }
