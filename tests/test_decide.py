import pandas as pd
from src.decide import decide, tune


def test_threshold():
    preds = {
        "s1": [
            ("s2_a", 0.90),
            ("s2_b", 0.60),
            ("s2_c", 0.30),
        ]
    }

    result = decide(preds, threshold=0.5, min_top=0.0, one_to_one=False)
    assert result["s1"] == ["s2_a", "s2_b"]


def test_min_top():
    preds = {
        "s1": [
            ("s2_a", 0.90),
            ("s2_b", 0.60),
        ]
    }

    result = decide(preds, threshold=0.5, min_top=0.95, one_to_one=False)
    assert result["s1"] == []


def test_empty_candidates():
    preds = {"s1": []}
    result = decide(preds, threshold=0.5, min_top=1.0)
    assert result["s1"] == []


def test_one_to_one_across_entities():
    """If the same cand is predicted for two S1s, only the higher-prob S1 keeps it."""
    preds = pd.DataFrame([
        {"s1_id": "s1_A", "cand_id": "s2_shared", "prob": 0.95},
        {"s1_id": "s1_B", "cand_id": "s2_shared", "prob": 0.80},
    ])

    result = decide(preds, threshold=0.5, min_top=0.0, one_to_one=True)
    assert result["s1_A"] == ["s2_shared"]
    assert result.get("s1_B", []) == []


def test_tune():
    preds = {
        "s1": [("cand1", 0.85), ("cand2", 0.40)],
        "s2": [("cand3", 0.95)],
    }
    truth = {
        "s1": {"cand1"},
        "s2": {"cand3"},
    }
    params = tune(preds, truth, threshold_grid=[0.5, 0.8], min_top_grid=[0.0, 0.5])
    assert "threshold" in params
    assert "min_top" in params
    assert params["val_f05"] == 1.0