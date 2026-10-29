"""Smoke test for a running fraud-api. Usage: python smoke_test.py [base_url]"""
import json
import sys
import urllib.error
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
GOOD = {"amount": 25.0, "hour": 14, "is_foreign": 0, "tx_last_24h": 1, "account_age_days": 1500}
RISKY = {"amount": 950.0, "hour": 3, "is_foreign": 1, "tx_last_24h": 9, "account_age_days": 12}


def call(method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


checks = []
code, body = call("GET", "/health")
checks.append(("health returns 200", code == 200 and body.get("status") == "ok"))
code, low = call("POST", "/predict", GOOD)
checks.append(("routine transaction scored", code == 200 and 0 <= low["fraud_probability"] <= 1))
code, high = call("POST", "/predict", RISKY)
checks.append(("risky scores higher than routine", code == 200 and high["fraud_probability"] > low["fraud_probability"]))
code, _ = call("POST", "/predict", {"amount": -1})
checks.append(("invalid request rejected with 422", code == 422))
code, batch = call("POST", "/predict_batch", {"transactions": [GOOD, RISKY]})
checks.append(("batch returns 2 results", code == 200 and batch.get("count") == 2))

for name, ok in checks:
    print(("PASS  " if ok else "FAIL  ") + name)
print("model version:", body.get("model_version"))
sys.exit(0 if all(ok for _, ok in checks) else 1)
