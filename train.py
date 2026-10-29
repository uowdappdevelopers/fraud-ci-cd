"""Train the fraud model and write the artifacts that CI checks and the image ships.

    python train.py --data data/transactions.csv --out model

Writes  model/model.joblib   the fitted pipeline (scaler + logistic regression)
        model/metadata.json  what the service reports at /metadata
        metrics.json         scores that the quality gate compares with baseline/metrics.json
Training is deterministic: the same data and code always give the same numbers.
"""
import argparse
import hashlib
import json
import os
import platform

import joblib
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["amount", "hour", "is_foreign", "tx_last_24h", "account_age_days"]
TARGET = "is_fraud"
SEED = 42


def split(df):
    """60% train, 20% validation, 20% test, stratified on the label."""
    X, y = df[FEATURES], df[TARGET]
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.4, random_state=SEED, stratify=y)
    X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=SEED, stratify=y_tmp)
    return (X_tr, y_tr), (X_va, y_va), (X_te, y_te)


def build_model(C=1.0):
    return Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", C=C, max_iter=1000)),
    ])


def scores(model, X, y):
    pred = model.predict(X)
    return {
        "precision": round(float(precision_score(y, pred, zero_division=0)), 3),
        "recall": round(float(recall_score(y, pred, zero_division=0)), 3),
        "f1": round(float(f1_score(y, pred, zero_division=0)), 3),
        "n": int(len(y)),
        "n_fraud": int(y.sum()),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/transactions.csv")
    ap.add_argument("--out", default="model")
    ap.add_argument("--version", default="1.0.0")
    ap.add_argument("--metrics", default="metrics.json")
    ap.add_argument("--C", type=float, default=1.0, help="inverse regularisation strength")
    args = ap.parse_args(argv)

    df = pd.read_csv(args.data)
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = split(df)
    model = build_model(args.C).fit(X_tr, y_tr)

    foreign = X_te["is_foreign"] == 1
    metrics = {
        "validation": scores(model, X_va, y_va),
        "test": scores(model, X_te, y_te),
        "slices": {
            "foreign": scores(model, X_te[foreign], y_te[foreign]),
            "domestic": scores(model, X_te[~foreign], y_te[~foreign]),
        },
        "rows": int(len(df)),
        "fraud_rate": round(float(df[TARGET].mean()), 4),
        "data_md5": hashlib.md5(open(args.data, "rb").read()).hexdigest(),
    }
    metadata = {
        "model_name": "fraud-detector",
        "version": args.version,
        "features": FEATURES,
        "threshold": 0.5,
        "trained_on_md5": metrics["data_md5"],
        "git_sha": os.getenv("GITHUB_SHA", "local")[:7],
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "validation": {k: metrics["validation"][k] for k in ("precision", "recall", "f1")},
        "test": {k: metrics["test"][k] for k in ("precision", "recall", "f1")},
    }

    os.makedirs(args.out, exist_ok=True)
    joblib.dump(model, os.path.join(args.out, "model.joblib"))
    with open(os.path.join(args.out, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    with open(args.metrics, "w") as f:
        json.dump(metrics, f, indent=2)

    t = metrics["test"]
    print(f"trained on {len(X_tr)} rows | test precision {t['precision']} recall {t['recall']} f1 {t['f1']}")
    print(f"wrote {args.out}/model.joblib, {args.out}/metadata.json, {args.metrics}")


if __name__ == "__main__":
    main()
