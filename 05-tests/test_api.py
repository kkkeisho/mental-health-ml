# tests/test_api.py
from fastapi.testclient import TestClient
from api.app import app

client = TestClient(app)


def test_root():
    resp = client.get("/")
    assert resp.status_code == 200
    assert "Mental Health Prediction API" in resp.json()["message"]


def test_predict():
    payload = {
        "Age": 30,
        "Gender": "male",
        "Country": "Japan",
        "self_employed": "No",
        "family_history": "No",
        "work_interfere": "Sometimes",
        "no_employees": "6-25",
        "remote_work": "Yes",
        "tech_company": "Yes",
        "benefits": "Yes",
        "care_options": "Not sure",
        "wellness_program": "No",
        "seek_help": "Yes",
        "anonymity": "Yes",
        "leave": "Somewhat easy",
        "mental_health_consequence": "No",
        "phys_health_consequence": "No",
        "coworkers": "Some of them",
        "supervisor": "Yes",
        "mental_health_interview": "No",
        "phys_health_interview": "Yes",
        "mental_vs_physical": "Don't know",
        "obs_consequence": "No"
        }

    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert "prediction" in body
    assert "probability_yes" in body
