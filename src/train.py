"""LightGBM training and prediction (owner: Vidushi, tasks V4/V5). Placeholder from S0."""
import numpy as np
import pandas as pd


def make_labels(pairs: pd.DataFrame, truth: dict[str, set[str]]) -> pd.Series:
    raise NotImplementedError("V4 (Vidushi)")


def train_model(X: pd.DataFrame, y: pd.Series) -> "lgb.Booster":
    raise NotImplementedError("V4 (Vidushi)")


def predict(model: "lgb.Booster", X: pd.DataFrame) -> np.ndarray:
    raise NotImplementedError("V4 (Vidushi)")


def save_model(model, path: str) -> None:
    raise NotImplementedError("V4 (Vidushi)")


def load_model(path: str) -> "lgb.Booster":
    raise NotImplementedError("V4 (Vidushi)")
