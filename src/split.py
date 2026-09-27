"""Train/val split by S1 (owner: Lavanya, task L2)."""
import json
from pathlib import Path
from typing import Union

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import SEED


def make_splits(
    truth: Union[dict[str, set[str]], pd.DataFrame],
    s1: pd.DataFrame,
    val_fraction: float = 0.2,
    seed: int = SEED,
) -> dict[str, list[str]]:
    """Split train data 80/20 by S1 ID stratified by country x is_singleton."""
    if isinstance(truth, dict):
        df_truth = pd.DataFrame([
            {
                "source1_entity_id": s1_id,
                "is_singleton": len(matched) == 0,
            }
            for s1_id, matched in truth.items()
        ])
    else:
        df_truth = truth.copy()
        matched_str = df_truth["matched_entity_ids"].fillna("").astype(str)
        df_truth["is_singleton"] = matched_str.str.strip() == ""

    s1_countries = s1[["entity_id", "country"]].drop_duplicates("entity_id")
    merged = df_truth.merge(
        s1_countries,
        left_on="source1_entity_id",
        right_on="entity_id",
        how="left",
    )

    merged["country"] = merged["country"].fillna("UNKNOWN").astype(str)
    merged["stratum"] = merged["country"] + "_" + merged["is_singleton"].astype(str)

    train_ids, val_ids = train_test_split(
        merged["source1_entity_id"],
        test_size=val_fraction,
        random_state=seed,
        stratify=merged["stratum"],
    )

    train_list = train_ids.tolist()
    val_list = val_ids.tolist()

    result = {
        "train": train_list,
        "val": val_list,
        "seed": seed,
        "val_fraction": val_fraction,
        # Backward compatibility aliases
        "train_source1_ids": train_list,
        "val_source1_ids": val_list,
    }

    out_path = Path("data/splits.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f)

    return result


if __name__ == "__main__":
    from src.io import load_sources, load_ground_truth

    s1, _, _ = load_sources("train")
    gt = load_ground_truth()
    splits = make_splits(gt, s1)
    print(f"Splits created: {len(splits['train'])} train, {len(splits['val'])} val")
