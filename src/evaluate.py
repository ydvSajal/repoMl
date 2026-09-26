"""Macro F0.5 scorer and blocking recall (owner: Lavanya, task L3). Placeholder from S0."""
import pandas as pd


def f05_entity(pred: set[str], truth: set[str]) -> float:
    raise NotImplementedError("L3 (Lavanya)")


def f05_macro(pred: dict[str, set[str]], truth: dict[str, set[str]]) -> float:
    raise NotImplementedError("L3 (Lavanya)")


def blocking_recall(cands: pd.DataFrame, truth: dict[str, set[str]]) -> dict[str, float]:
    raise NotImplementedError("L3 (Lavanya)")
