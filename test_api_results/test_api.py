import requests 
from fastapi.testclient import TestClient
from ..app import app

client = TestClient(app=app)

def test_prediction():
    response = client.post(
        "/predict",
        json=payload
    )

    assert response.status_code == 200