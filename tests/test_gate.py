import copy

from ci import quality_gate

BASELINE = {"test": {"precision": 0.139, "recall": 0.676, "f1": 0.231}}
GOOD = {
    "test": {"precision": 0.139, "recall": 0.676, "f1": 0.231},
    "slices": {"foreign": {"recall": 0.857, "n_fraud": 14}, "domestic": {"recall": 0.55, "n_fraud": 20}},
}


def failed(candidate):
    return [r for r, ok, _, _ in quality_gate.evaluate(candidate, BASELINE) if not ok]


def test_the_baseline_itself_passes():
    assert failed(GOOD) == []


def test_low_recall_is_blocked():
    bad = copy.deepcopy(GOOD)
    bad["test"]["recall"] = 0.40
    assert "test recall >= floor" in failed(bad)


def test_f1_drop_beyond_tolerance_is_blocked():
    bad = copy.deepcopy(GOOD)
    bad["test"]["f1"] = 0.15
    assert "test F1 not below baseline - tolerance" in failed(bad)


def test_a_small_f1_drop_within_tolerance_is_allowed():
    ok = copy.deepcopy(GOOD)
    ok["test"]["f1"] = 0.21
    assert failed(ok) == []


def test_a_group_left_behind_is_blocked():
    bad = copy.deepcopy(GOOD)
    bad["slices"]["domestic"]["recall"] = 0.20
    assert "domestic slice recall >= floor" in failed(bad)


def test_noisy_slices_with_few_frauds_are_not_enforced():
    ok = copy.deepcopy(GOOD)
    ok["slices"]["foreign"] = {"recall": 0.0, "n_fraud": 3}
    assert failed(ok) == []
