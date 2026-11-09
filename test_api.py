
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

VALID_PASSENGER = {
    "Pclass": 3,
    "Sex": "female",
    "Age": 25,
    "SibSp": 0,
    "Parch": 0,
    "Fare": 7.25,
    "Embarked": "S"
}

def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_prediction():
    response = client.post("/predict", json=VALID_PASSENGER)
    assert response.status_code == 200

def test_prediction_schema():
    response = client.post("/predict", json=VALID_PASSENGER)
    data = response.json()
    assert "prediction" in data
    assert "prediction_label" in data
    assert "survival_probability" in data
    assert "non_survival_probability" in data

def test_prediction_value():
    response = client.post("/predict", json=VALID_PASSENGER)
    assert response.json()["prediction"] in [0, 1]

def test_probability_range():
    response = client.post("/predict", json=VALID_PASSENGER)
    data = response.json()
    assert 0 <= data["survival_probability"] <= 1
    assert 0 <= data["non_survival_probability"] <= 1

def test_probability_sum():
    response = client.post("/predict", json=VALID_PASSENGER)
    data = response.json()
    total = data["survival_probability"] + data["non_survival_probability"]
    assert abs(total - 1.0) < 0.0001

def test_invalid_passenger_class():
    passenger = VALID_PASSENGER.copy()
    passenger["Pclass"] = 10
    response = client.post("/predict", json=passenger)
    assert response.status_code == 422

def test_missing_field():
    passenger = VALID_PASSENGER.copy()
    del passenger["Age"]
    response = client.post("/predict", json=passenger)
    assert response.status_code == 422

def test_invalid_age():
    passenger = VALID_PASSENGER.copy()
    passenger["Age"] = -5
    response = client.post("/predict", json=passenger)
    assert response.status_code == 422
