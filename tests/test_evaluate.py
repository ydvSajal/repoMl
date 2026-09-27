from src.evaluate import f05_entity, f05_macro, blocking_recall


def test_f05_entity_pdf_example():
    """PDF challenge example: pred {S2-00047, S2-00193, S3-00812} against truth {S2-00047, S3-00812} -> 0.714."""
    pred = {"S2-00047", "S2-00193", "S3-00812"}
    truth = {"S2-00047", "S3-00812"}
    score = f05_entity(pred, truth)
    assert abs(score - 0.714) <= 0.001


def test_empty_both_singleton():
    """A singleton predicted empty scores 1.0."""
    assert f05_entity(set(), set()) == 1.0
    assert f05_entity([], []) == 1.0


def test_singleton_predicted_non_empty():
    """A singleton predicted non-empty scores 0.0."""
    assert f05_entity({"X"}, set()) == 0.0
    assert f05_entity(["X"], []) == 0.0


def test_empty_prediction_non_empty_truth():
    """Non-empty truth with empty prediction scores 0.0."""
    assert f05_entity(set(), {"S2-00047"}) == 0.0
    assert f05_entity([], ["S2-00047"]) == 0.0


def test_perfect_match():
    assert f05_entity({"a", "b"}, {"a", "b"}) == 1.0


def test_f05_macro_averages_over_truth():
    truth = {
        "s1": {"a", "b"},
        "s2": set(),  # singleton
        "s3": {"c"},
    }
    # Missing s3 from pred -> treated as empty -> scores 0.0
    # s1 matches perfectly -> 1.0
    # s2 singleton empty -> 1.0
    pred = {
        "s1": {"a", "b"},
        "s2": set(),
    }
    # Macro avg: (1.0 + 1.0 + 0.0) / 3 = 2/3
    assert abs(f05_macro(pred, truth) - (2.0 / 3.0)) < 1e-6


def test_blocking_recall():
    candidates = {
        "s1": ["a", "b", "c"],
        "s2": [],
    }
    truth = {
        "s1": ["a", "b"],
        "s2": ["d"],
    }
    res = blocking_recall(candidates, truth)
    assert isinstance(res, dict)
    assert "pair_recall" in res
    assert "f05_ceiling" in res
    assert "avg_cands" in res
    assert "pct_s1_zero_cands" in res
    # 2 true pairs recovered out of 3 -> 2/3
    assert abs(res["pair_recall"] - (2.0 / 3.0)) < 1e-6
    assert res["pct_s1_zero_cands"] == 0.5