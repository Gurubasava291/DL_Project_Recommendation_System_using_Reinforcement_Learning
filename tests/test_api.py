import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_read_index():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]

def test_recommend_dqn():
    response = client.post("/recommend", json={"state": [1, 2, 3], "k": 5, "model_type": "dqn"})
    assert response.status_code == 200
    data = response.json()
    assert "recommendations" in data
    assert len(data["recommendations"]) == 5
    assert data["model"] == "Deep Q-Network"

def test_recommend_invalid_state():
    # State length should be exactly 3
    response = client.post("/recommend", json={"state": [1, 2], "k": 5, "model_type": "dqn"})
    assert response.status_code == 400

def test_metrics():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "latest_run" in response.json()
