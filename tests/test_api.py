import pytest
from fastapi.testclient import TestClient

from app.api import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_valid_predict_request():
    r = client.post("/predict", json={
        "request_text": "water purifier leaking from the bottom",
        "product_family": "Water Purifier",
        "warranty_status": "in_warranty",
        "channel": "whatsapp",
    })
    assert r.status_code == 200
    body = r.json()
    assert body["predicted_team"]
    assert "confidence" in body
    assert "confidence_band" in body
    assert "review_required" in body
    assert isinstance(body["reasons"], list) and len(body["reasons"]) > 0


def test_missing_request_text_returns_422():
    r = client.post("/predict", json={"product_family": "Water Purifier"})
    assert r.status_code == 422


def test_empty_request_text_returns_422():
    r = client.post("/predict", json={"request_text": ""})
    assert r.status_code == 422


def test_predict_with_only_required_field():
    r = client.post("/predict", json={"request_text": "fan making noise"})
    assert r.status_code == 200
    assert r.json()["predicted_team"]


def test_response_schema_has_similar_cases():
    r = client.post("/predict", json={"request_text": "purifier leaking water"})
    body = r.json()
    assert "similar_cases" in body
    for case in body["similar_cases"]:
        assert set(case.keys()) >= {"request_text", "team", "similarity"}
