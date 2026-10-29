"""Model quality gate: decide whether the freshly trained model may be shipped.

    python ci/quality_gate.py                      # reads metrics.json and baseline/metrics.json

A model is blocked when it is below an absolute floor, or clearly worse than the approved
baseline. Exit code 1 blocks the pipeline. Rules are plain numbers in this file, so changing
a rule is a reviewed code change, not a quiet decision.
"""
import json
import os
import sys

MIN_TEST_RECALL = 0.60          # a fraud model that misses 4 in 10 frauds is not shippable
MIN_TEST_PRECISION = 0.10       # below this the review team drowns in false alarms
F1_TOLERANCE = 0.03             # allowed drop in F1 against the approved baseline
MIN_SLICE_RECALL = {"foreign": 0.50, "domestic": 0.40}   # no group may be left behind
MIN_SLICE_FRAUD_ROWS = 10       # below this many frauds a slice score is too noisy to enforce


def evaluate(candidate, baseline):
    """Return a list of (rule, passed, observed, required)."""
    t = candidate["test"]
    rules = []
    rules.append(("test recall >= floor", t["recall"] >= MIN_TEST_RECALL, t["recall"], f">= {MIN_TEST_RECALL}"))
    rules.append(("test precision >= floor", t["precision"] >= MIN_TEST_PRECISION, t["precision"], f">= {MIN_TEST_PRECISION}"))
    need_f1 = round(baseline["test"]["f1"] - F1_TOLERANCE, 3)
    rules.append(("test F1 not below baseline - tolerance", t["f1"] >= need_f1, t["f1"], f">= {need_f1}"))
    for name, floor in MIN_SLICE_RECALL.items():
        s = candidate["slices"][name]
        if s["n_fraud"] < MIN_SLICE_FRAUD_ROWS:
            rules.append((f"{name} slice recall", True, s["recall"], f"skipped, only {s['n_fraud']} frauds"))
        else:
            rules.append((f"{name} slice recall >= floor", s["recall"] >= floor, s["recall"], f">= {floor}"))
    return rules


def main(argv=None):
    with open("metrics.json") as f:
        candidate = json.load(f)
    with open("baseline/metrics.json") as f:
        baseline = json.load(f)
    rules = evaluate(candidate, baseline)

    width = max(len(r[0]) for r in rules)
    print(f"{'rule'.ljust(width)}  observed  required")
    for rule, ok, observed, required in rules:
        print(f"{'PASS' if ok else 'FAIL'}  {rule.ljust(width)}  {observed:<8}  {required}")
    d = round(candidate["test"]["f1"] - baseline["test"]["f1"], 3)
    print(f"\nF1 on the test set: candidate {candidate['test']['f1']}, baseline {baseline['test']['f1']} (change {d:+})")

    summary = os.getenv("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as f:
            f.write("### Model quality gate\n\n| Rule | Result | Observed | Required |\n|---|---|---|---|\n")
            for rule, ok, observed, required in rules:
                f.write(f"| {rule} | {'pass' if ok else '**FAIL**'} | {observed} | {required} |\n")

    if all(ok for _, ok, _, _ in rules):
        print("quality gate PASSED: this model may be built into an image")
        return 0
    print("quality gate FAILED: this model must not be shipped")
    return 1


if __name__ == "__main__":
    sys.exit(main())
