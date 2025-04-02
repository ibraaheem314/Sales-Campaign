import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_predict_endpoint():
    test_data = {
        "call_time": 15,
        "duration": 350,
        "region": "East",
        "product": "SaaS",
        "script_version": "B",
        "day_of_week": "Wednesday"
    }
    
    response = client.post("/predict", json=test_data)
    assert response.status_code == 200
    assert 0 <= response.json()['conversion_probability'] <= 1

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()['status'] == 'healthy'