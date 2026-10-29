"""Fraud-scoring service: loads a saved model once and serves it over HTTP."""
import collections
import json
import logging
import os
import threading
import time
import uuid

import joblib
import pandas as pd
from flask import Flask, g, jsonify, request
from pydantic import BaseModel, ConfigDict, Field, ValidationError

MODEL_PATH = os.getenv("MODEL_PATH", "model/model.joblib")
META_PATH = os.getenv("META_PATH", "model/metadata.json")
MAX_BATCH = 2000

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("fraud-api")

# 1. Load the model ONCE at start-up.
model = joblib.load(MODEL_PATH)
with open(META_PATH) as f:
    meta = json.load(f)
FEATURES = meta["features"]
THRESHOLD = float(os.getenv("THRESHOLD", meta.get("threshold", 0.5)))


# 2. The API contract: what a valid request looks like.
class Transaction(BaseModel):
    model_config = ConfigDict(extra="forbid")          # unknown fields are rejected
    amount: float = Field(gt=0, lt=1_000_000)
    hour: int = Field(ge=0, le=23)
    is_foreign: int = Field(ge=0, le=1)
    tx_last_24h: int = Field(ge=0, le=1000)
    account_age_days: int = Field(ge=0, le=36500)


app = Flask(__name__)

# In-process metrics (a real system would use the prometheus_client library).
_lock = threading.Lock()
BUCKETS_MS = [5, 10, 25, 50, 100, 250, 500, 1000]
REQS = collections.Counter()        # (path, status) -> count
LAT = collections.Counter()         # bucket upper bound -> count of requests <= bound
LAT_SUM = [0.0, 0]                  # [sum of latency ms, request count]
PREDS = collections.Counter()       # "flagged" / "not_flagged" -> rows scored


def score(rows):
    """rows: list of dicts -> list of probabilities (same order)."""
    frame = pd.DataFrame(rows, columns=FEATURES)
    return model.predict_proba(frame)[:, 1]


def result(prob):
    return {"fraud_probability": round(float(prob), 4),
            "is_fraud": bool(prob >= THRESHOLD),
            "threshold": THRESHOLD,
            "model_version": meta["version"]}


def validation_error(exc):
    details = [{"field": ".".join(str(p) for p in e["loc"]), "message": e["msg"]} for e in exc.errors()]
    return jsonify(error="validation_failed", details=details), 422


@app.before_request
def start_timer():
    g.t0 = time.perf_counter()
    g.request_id = request.headers.get("X-Request-ID", uuid.uuid4().hex[:12])


# 4. One structured log line per request.
@app.after_request
def log_request(resp):
    ms = round((time.perf_counter() - g.t0) * 1000, 2)
    log.info(json.dumps({"request_id": g.request_id, "method": request.method, "path": request.path,
                         "status": resp.status_code, "latency_ms": ms, "model_version": meta["version"]}))
    resp.headers["X-Request-ID"] = g.request_id
    if request.path != "/metrics":
        with _lock:
            REQS[(request.path, resp.status_code)] += 1
            LAT_SUM[0] += ms
            LAT_SUM[1] += 1
            for b in BUCKETS_MS:
                if ms <= b:
                    LAT[b] += 1
    return resp


@app.get("/health")
def health():
    return jsonify(status="ok", model_version=meta["version"])


@app.get("/metadata")
def metadata():
    return jsonify(meta)


@app.get("/metrics")
def metrics():
    """Prometheus text format: counters, a latency histogram and prediction counts."""
    with _lock:
        out = ["# TYPE api_requests_total counter"]
        for (path, status), n in sorted(REQS.items()):
            out.append(f'api_requests_total{{path="{path}",status="{status}"}} {n}')
        out.append("# TYPE api_request_latency_ms histogram")
        for b in BUCKETS_MS:
            out.append(f'api_request_latency_ms_bucket{{le="{b}"}} {LAT[b]}')
        out.append(f'api_request_latency_ms_bucket{{le="+Inf"}} {LAT_SUM[1]}')
        out.append(f"api_request_latency_ms_sum {round(LAT_SUM[0], 3)}")
        out.append(f"api_request_latency_ms_count {LAT_SUM[1]}")
        out.append("# TYPE api_predictions_total counter")
        for k, n in sorted(PREDS.items()):
            out.append(f'api_predictions_total{{outcome="{k}"}} {n}')
        out.append(f'api_model_info{{version="{meta["version"]}"}} 1')
    return "\n".join(out) + "\n", 200, {"Content-Type": "text/plain; version=0.0.4"}


@app.post("/predict")
def predict():
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify(error="bad_request", message="Body must be a JSON object"), 400
    try:
        tx = Transaction(**body)
    except ValidationError as exc:
        return validation_error(exc)
    prob = score([tx.model_dump()])[0]
    with _lock:
        PREDS["flagged" if prob >= THRESHOLD else "not_flagged"] += 1
    return jsonify(request_id=g.request_id, **result(prob))


@app.post("/predict_batch")
def predict_batch():
    body = request.get_json(silent=True)
    if not isinstance(body, dict) or not isinstance(body.get("transactions"), list):
        return jsonify(error="bad_request", message='Body must be {"transactions": [...]}'), 400
    items = body["transactions"]
    if len(items) == 0:
        return jsonify(error="bad_request", message="transactions is empty"), 400
    if len(items) > MAX_BATCH:
        return jsonify(error="too_large", message=f"At most {MAX_BATCH} transactions per call"), 413
    try:
        rows = [Transaction(**item).model_dump() for item in items]
    except (ValidationError, TypeError) as exc:
        if isinstance(exc, ValidationError):
            return validation_error(exc)
        return jsonify(error="bad_request", message="Each item must be a JSON object"), 400
    probs = score(rows)
    with _lock:
        for p in probs:
            PREDS["flagged" if p >= THRESHOLD else "not_flagged"] += 1
    return jsonify(request_id=g.request_id, count=len(rows), predictions=[result(p) for p in probs])


# 5. Do not leak internals.
@app.errorhandler(Exception)
def unexpected(exc):
    log.exception("unhandled error")
    return jsonify(error="internal_error", message="Something went wrong"), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
