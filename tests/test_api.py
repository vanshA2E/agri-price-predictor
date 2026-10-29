from fastapi.testclient import TestClient

from app.api import app

client = TestClient(app)


def test_predict_valid_request() -> None:
    response = client.post("/predict", json={"commodity": "Onion", "state": "Rajasthan"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["commodity"] == "Onion"
    assert payload["state"] == "Rajasthan"
    assert payload["forecast_price"] >= 0
    assert payload["target_week"]


def test_predict_unknown_commodity_returns_404() -> None:
    response = client.post("/predict", json={"commodity": "Unknown crop", "state": "Rajasthan"})
    assert response.status_code == 404
