"""TSV/parquet IO (owner: Sajal). Every TSV in the project is read through here."""
import os
from typing import Literal

import pandas as pd

from src.config import PATHS


def read_tsv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def load_sources(split: Literal["train", "test"]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = os.path.join(PATHS["dataset"], split)
    s1, s2, s3 = (read_tsv(os.path.join(base, f"{split}_source{i}.tsv")) for i in (1, 2, 3))
    return s1, s2, s3


def load_ground_truth() -> dict[str, set[str]]:
    gt = read_tsv(os.path.join(PATHS["dataset"], "train", "train_ground_truth.tsv"))
    return {s1: {i.strip() for i in ids.split(",") if i.strip()}
            for s1, ids in zip(gt["source1_entity_id"], gt["matched_entity_ids"])}


def write_id_lists(mapping: dict[str, list[str]], s1_ids: list[str], path: str,
                   id_col: Literal["matched_entity_ids", "candidate_entity_ids"]) -> None:
    assert len(set(s1_ids)) == len(s1_ids), "duplicate S1 ids"
    rows = [",".join(dict.fromkeys(mapping.get(s1, []))) for s1 in s1_ids]  # dedupe, keep order
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df = pd.DataFrame({"source1_entity_id": s1_ids, id_col: rows})
    df.to_csv(path, sep="\t", index=False, lineterminator="\n")  # LF even on Windows


def read_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


def write_parquet(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_parquet(path, index=False)


if __name__ == "__main__":
    for split in ("train", "test"):
        for i, df in enumerate(load_sources(split), 1):
            print(f"{split}_source{i}: {len(df)} rows", df["country"].value_counts().to_dict())
    gt = load_ground_truth()
    print(f"ground_truth: {len(gt)} S1, {sum(1 for v in gt.values() if not v)} singletons")
