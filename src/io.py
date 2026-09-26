"""TSV/parquet IO (owner: Sajal, task S1). Placeholder from S0."""
from typing import Literal

import pandas as pd


def read_tsv(path: str) -> pd.DataFrame:
    raise NotImplementedError("S1 (Sajal)")


def load_sources(split: Literal["train", "test"]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raise NotImplementedError("S1 (Sajal)")


def load_ground_truth() -> dict[str, set[str]]:
    raise NotImplementedError("S1 (Sajal)")


def write_id_lists(mapping: dict[str, list[str]], s1_ids: list[str], path: str,
                   id_col: Literal["matched_entity_ids", "candidate_entity_ids"]) -> None:
    raise NotImplementedError("S1 (Sajal)")


def read_parquet(path: str) -> pd.DataFrame:
    raise NotImplementedError("S1 (Sajal)")


def write_parquet(df: pd.DataFrame, path: str) -> None:
    raise NotImplementedError("S1 (Sajal)")
