from app import app

client = app.test_client()
GOOD = {"amount": 25.0, "hour": 14, "is_foreign": 0, "tx_last_24h": 1, "account_age_days": 1500}
RISKY = {"amount": 950.0, "hour": 3, "is_foreign": 1, "tx_last_24h": 9, "account_age_days": 12}


def test_health_is_ok():
    r = client.get("/health")
    assert r.status_code == 200 and r.get_json()["status"] == "ok"


def test_predict_returns_probability_and_version():
    r = client.post("/predict", json=GOOD)
    body = r.get_json()
    assert r.status_code == 200
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert isinstance(body["is_fraud"], bool)
    assert body["model_version"]


def test_risky_scores_higher_than_routine():
    low = client.post("/predict", json=GOOD).get_json()["fraud_probability"]
    high = client.post("/predict", json=RISKY).get_json()["fraud_probability"]
    assert high > low


def test_missing_field_is_422():
    r = client.post("/predict", json={"amount": 10})
    assert r.status_code == 422 and r.get_json()["error"] == "validation_failed"


def test_impossible_values_are_422():
    r = client.post("/predict", json={**GOOD, "amount": -5, "hour": 25})
    assert r.status_code == 422
    fields = {d["field"] for d in r.get_json()["details"]}
    assert {"amount", "hour"} <= fields


def test_unknown_field_is_rejected():
    r = client.post("/predict", json={**GOOD, "card_number": "4111"})
    assert r.status_code == 422


def test_non_json_body_is_400():
    r = client.post("/predict", data="hello")
    assert r.status_code == 400


def test_batch_returns_one_result_per_row_in_order():
    r = client.post("/predict_batch", json={"transactions": [GOOD, RISKY]})
    preds = r.get_json()["predictions"]
    assert r.status_code == 200 and len(preds) == 2
    assert preds[1]["fraud_probability"] > preds[0]["fraud_probability"]


def test_batch_too_large_is_413():
    r = client.post("/predict_batch", json={"transactions": [GOOD] * 1001})
    assert r.status_code == 413


def test_metrics_endpoint_exposes_counters():
    client.post("/predict", json=GOOD)
    text = client.get("/metrics").get_data(as_text=True)
    assert "api_requests_total" in text and "api_request_latency_ms_bucket" in text
