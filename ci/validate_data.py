"""Data checks that run BEFORE training. Exit code 1 if any check fails.

    python ci/validate_data.py data/transactions.csv
"""
import os
import sys

import pandas as pd

COLUMNS = ["amount", "hour", "is_foreign", "tx_last_24h", "account_age_days", "is_fraud"]
MIN_ROWS = 1000
FRAUD_RATE_RANGE = (0.01, 0.15)     # the real rate is about 5.7%
MAX_DUPLICATE_SHARE = 0.05


def run_checks(df):
    """Return a list of (name, passed, detail)."""
    results = []

    def check(name, passed, detail=""):
        results.append((name, bool(passed), detail))

    check("columns match the contract", list(df.columns) == COLUMNS, f"found {list(df.columns)}")
    if list(df.columns) != COLUMNS:
        return results                      # the checks below need the right columns
    check("enough rows", len(df) >= MIN_ROWS, f"{len(df)} rows (minimum {MIN_ROWS})")
    check("no missing values", df.isna().sum().sum() == 0, f"{int(df.isna().sum().sum())} missing")
    check("amount is positive", (df["amount"] > 0).all(), f"min {df['amount'].min()}")
    check("hour is 0-23", df["hour"].between(0, 23).all(), f"range {df['hour'].min()}-{df['hour'].max()}")
    check("is_foreign is 0 or 1", df["is_foreign"].isin([0, 1]).all())
    check("tx_last_24h is not negative", (df["tx_last_24h"] >= 0).all(), f"min {df['tx_last_24h'].min()}")
    check("account_age_days is not negative", (df["account_age_days"] >= 0).all(), f"min {df['account_age_days'].min()}")
    check("label is 0 or 1", df["is_fraud"].isin([0, 1]).all())
    rate = df["is_fraud"].mean()
    check("fraud rate in expected range", FRAUD_RATE_RANGE[0] <= rate <= FRAUD_RATE_RANGE[1],
          f"{rate:.3f} (expected {FRAUD_RATE_RANGE[0]}-{FRAUD_RATE_RANGE[1]})")
    dup = df.duplicated().mean()
    check("few duplicate rows", dup <= MAX_DUPLICATE_SHARE, f"{dup:.3f} (maximum {MAX_DUPLICATE_SHARE})")
    return results


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    path = argv[0] if argv else "data/transactions.csv"
    df = pd.read_csv(path)
    results = run_checks(df)
    width = max(len(n) for n, _, _ in results)
    for name, ok, detail in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name.ljust(width)}  {detail}")
    failed = [n for n, ok, _ in results if not ok]
    summary = os.getenv("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as f:
            f.write("### Data validation\n\n| Check | Result | Detail |\n|---|---|---|\n")
            for name, ok, detail in results:
                f.write(f"| {name} | {'pass' if ok else '**FAIL**'} | {detail} |\n")
    if failed:
        print(f"\n{len(failed)} data check(s) failed: the model was NOT trained.")
        return 1
    print(f"\nall {len(results)} data checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
