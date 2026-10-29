import json

import joblib
import pandas as pd

import train

DATA = "data/transactions.csv"


def test_training_is_deterministic(tmp_path):
    train.main(["--data", DATA, "--out", str(tmp_path / "a"), "--metrics", str(tmp_path / "a.json")])
    train.main(["--data", DATA, "--out", str(tmp_path / "b"), "--metrics", str(tmp_path / "b.json")])
    a = json.load(open(tmp_path / "a.json"))
    b = json.load(open(tmp_path / "b.json"))
    assert a == b          # same data and code must always give the same scores


def test_artifacts_are_written_and_the_model_loads(tmp_path):
    train.main(["--data", DATA, "--out", str(tmp_path / "m"), "--metrics", str(tmp_path / "m.json")])
    model = joblib.load(tmp_path / "m" / "model.joblib")
    meta = json.load(open(tmp_path / "m" / "metadata.json"))
    assert meta["features"] == train.FEATURES
    probs = model.predict_proba(pd.read_csv(DATA)[train.FEATURES].head(5))[:, 1]
    assert ((probs >= 0) & (probs <= 1)).all()


def test_splits_do_not_overlap_and_have_expected_sizes():
    df = pd.read_csv(DATA)
    (Xtr, _), (Xva, _), (Xte, _) = train.split(df)
    assert (len(Xtr), len(Xva), len(Xte)) == (1800, 600, 600)
    assert not (set(Xtr.index) & set(Xte.index))
    assert not (set(Xva.index) & set(Xte.index))


def test_model_is_clearly_better_than_chance(tmp_path):
    train.main(["--data", DATA, "--out", str(tmp_path / "m"), "--metrics", str(tmp_path / "m.json")])
    m = json.load(open(tmp_path / "m.json"))
    assert m["test"]["recall"] > 0.5 and m["test"]["precision"] > m["fraud_rate"]
