"""Decision layer and threshold tuning (owner: Lavanya, task L6). Placeholder from S0."""
import pandas as pd


def decide(preds: pd.DataFrame, threshold: float, min_top: float, one_to_one: bool) -> dict[str, list[str]]:
    raise NotImplementedError("L6 (Lavanya)")


def tune(preds_val: pd.DataFrame, truth_val: dict[str, set[str]]) -> dict:
    raise NotImplementedError("L6 (Lavanya)")
