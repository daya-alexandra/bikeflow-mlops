from fastapi.testclient import TestClient

from bikeflow.api.main import app

client = TestClient(app)

VALID_REQUEST = {
    "prediction_time": "2026-07-15T08:00:00+09:00",
    "temperature_c": 24.5,
    "humidity_pct": 61.0,
    "wind_speed_m_s": 1.8,
    "visibility_10m": 1800,
    "dew_point_c": 16.4,
    "solar_radiation_mj_m2": 1.2,
    "rainfall_mm": 0.0,
    "snowfall_cm": 0.0,
    "holiday": False,
    "functioning_day": True,
}


def test_predict_returns_stub_prediction() -> None:
    response = client.post("/predict", json=VALID_REQUEST)

    assert response.status_code == 200
    assert response.json() == {
        "prediction_time": VALID_REQUEST["prediction_time"],
        "predicted_rentals": 42.0,
        "model_version": "stub-v0",
    }


def test_predict_rejects_invalid_humidity() -> None:
    invalid_request = VALID_REQUEST | {"humidity_pct": 101}

    response = client.post("/predict", json=invalid_request)

    assert response.status_code == 422
