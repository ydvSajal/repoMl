"""Pairwise and group features (owner: Vidushi, tasks V1-V3). Placeholder from S0."""
import pandas as pd

FEATURE_COLUMNS: list[str] = []


class FeatureContext:
    """IDF tables + fitted vectorizers. Vidushi defines the contents."""


def fit_feature_context(all_norm: pd.DataFrame) -> FeatureContext:
    raise NotImplementedError("V1 (Vidushi)")


def build_features(pairs: pd.DataFrame, s1_norm: pd.DataFrame, pool_norm: pd.DataFrame,
                   ctx: FeatureContext) -> pd.DataFrame:
    raise NotImplementedError("V1 (Vidushi)")


def add_group_features(feats: pd.DataFrame) -> pd.DataFrame:
    raise NotImplementedError("V3 (Vidushi)")
