from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_health_reports_no_model():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_ready": False}


def test_prediction_fails_closed_without_model():
    response = client.post(
        "/api/v1/predict",
        json={"compound": "caffeine", "input_type": "name", "endpoint": "general"},
    )
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "MODEL_NOT_CONFIGURED"


def test_blank_compound_is_rejected():
    response = client.post(
        "/api/v1/predict",
        json={"compound": "   ", "input_type": "name", "endpoint": "general"},
    )
    assert response.status_code == 422


def test_unknown_endpoint_is_rejected():
    response = client.post(
        "/api/v1/predict",
        json={"compound": "caffeine", "input_type": "name", "endpoint": "made_up"},
    )
    assert response.status_code == 422
