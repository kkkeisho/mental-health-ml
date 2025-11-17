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


@app.get("/")
def root():
    # Test expects this message to contain the phrase
    # "Mental Health Prediction API"
    return {"message": "Mental Health Prediction API is running"}


@app.head("/")
def root_head():
    # Render requires HEAD / to return 200 for its health check
    return Response(status_code=200)


@app.post("/predict")
def predict(input_data: SurveyInput):

    # Convert Pydantic model → dictionary
    # Prefer model_dump() for Pydantic v2; fall back to dict() if v1
    try:
        input_dict = input_data.model_dump()
    except AttributeError:
        input_dict = input_data.dict()

    df = pd.DataFrame([input_dict])

    # Run the model
    pred = model.predict(df)[0]
    proba = model.predict_proba(df)[0, 1]

    # Format the response
    return {
        "prediction": int(pred),
        "probability_yes": float(proba),
        "model": "logistic-regression + preprocessing pipeline",
    }