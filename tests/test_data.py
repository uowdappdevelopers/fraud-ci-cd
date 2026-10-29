import numpy as np
import pandas as pd
import pytest

from ci import validate_data


def good_frame(n=1200):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "amount": np.round(rng.lognormal(4, 1, n), 2) + rng.random(n) * 1e-3,
        "hour": rng.integers(0, 24, n),
        "is_foreign": rng.integers(0, 2, n),
        "tx_last_24h": rng.poisson(2, n),
        "account_age_days": rng.integers(5, 3000, n),
        "is_fraud": (rng.random(n) < 0.06).astype(int),
    })
    return df


def failed_names(df):
    return {name for name, ok, _ in validate_data.run_checks(df) if not ok}


def test_a_clean_frame_passes_every_check():
    assert failed_names(good_frame()) == set()


@pytest.mark.parametrize("mutate, expected", [
    (lambda d: d.assign(amount=-1.0), "amount is positive"),
    (lambda d: d.assign(hour=25), "hour is 0-23"),
    (lambda d: d.assign(is_foreign=2), "is_foreign is 0 or 1"),
    (lambda d: d.assign(is_fraud=0), "fraud rate in expected range"),
    (lambda d: d.head(100), "enough rows"),
    (lambda d: d.drop(columns=["amount"]), "columns match the contract"),
    (lambda d: pd.concat([d, d.head(200)]), "few duplicate rows"),
])
def test_each_kind_of_bad_data_is_caught(mutate, expected):
    assert expected in failed_names(mutate(good_frame()))


def test_missing_values_are_caught():
    df = good_frame()
    df.loc[0, "amount"] = np.nan
    assert "no missing values" in failed_names(df)
