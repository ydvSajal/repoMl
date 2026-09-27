"""Decision layer and threshold tuning (owner: Lavanya, task L6)."""
from itertools import product
from typing import Union
import pandas as pd

from src.config import THRESHOLD_GRID, MIN_TOP_GRID
from src.evaluate import f05_macro


def decide(
    preds: Union[pd.DataFrame, dict[str, list[tuple[str, float]]]],
    threshold: float = 0.5,
    min_top: float = 0.0,
    one_to_one: bool = True,
) -> dict[str, list[str]]:
    """Convert model probabilities into final match decisions per TRD §5.6.
    
    1. If one_to_one is True, for each cand_id keep only the row with highest prob.
    2. Per S1: if max(prob) < min_top, return empty list for that S1.
       Otherwise keep every candidate with prob >= threshold, ordered by prob desc.
    3. Returns mapping {s1_id: [matched_cand_ids]}.
    """
    if isinstance(preds, dict):
        rows = []
        for s1_id, cands in preds.items():
            for item in cands:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    rows.append({"s1_id": str(s1_id), "cand_id": str(item[0]), "prob": float(item[1])})
        df = pd.DataFrame(rows, columns=["s1_id", "cand_id", "prob"]) if rows else pd.DataFrame(columns=["s1_id", "cand_id", "prob"])
    else:
        df = preds.copy()
        if "source1_entity_id" in df.columns:
            df["s1_id"] = df["source1_entity_id"]
        if "candidate_entity_id" in df.columns:
            df["cand_id"] = df["candidate_entity_id"]
        df["s1_id"] = df["s1_id"].astype(str)
        df["cand_id"] = df["cand_id"].astype(str)
        df["prob"] = df["prob"].astype(float)

    if df.empty:
        all_s1 = set(preds.keys()) if isinstance(preds, dict) else set(df.get("s1_id", []))
        return {s1: [] for s1 in all_s1}

    # Sort by prob descending
    df = df.sort_values(by="prob", ascending=False)

    # 1. One-to-one deduplication across S1: keep highest prob per candidate
    if one_to_one:
        df = df.drop_duplicates(subset=["cand_id"], keep="first")

    result: dict[str, list[str]] = {}
    for s1_id, group in df.groupby("s1_id"):
        max_p = group["prob"].max()
        if min_top > 0 and max_p < min_top:
            result[s1_id] = []
        else:
            passing = group[group["prob"] >= threshold]
            result[s1_id] = passing["cand_id"].tolist()

    # Ensure all S1 entities present in original preds are preserved
    if isinstance(preds, dict):
        for s1_id in preds:
            if s1_id not in result:
                result[s1_id] = []

    return result


def tune(
    preds_val: Union[pd.DataFrame, dict[str, list[tuple[str, float]]]],
    truth_val: dict[str, Union[set[str], list[str]]],
    threshold_grid: list[float] = None,
    min_top_grid: list[float] = None,
    one_to_one: bool = True,
) -> dict:
    """Grid-search threshold x min_top to maximize macro F0.5 over all val S1 IDs."""
    th_grid = threshold_grid or THRESHOLD_GRID
    mt_grid = min_top_grid or MIN_TOP_GRID

    best_score = -1.0
    best_params = {
        "threshold": 0.5,
        "min_top": 0.0,
        "one_to_one": one_to_one,
        "val_f05": 0.0,
    }

    for th, mt in product(th_grid, mt_grid):
        decisions = decide(preds_val, threshold=th, min_top=mt, one_to_one=one_to_one)
        score = f05_macro(decisions, truth_val)
        if score > best_score:
            best_score = score
            best_params = {
                "threshold": float(th),
                "min_top": float(mt),
                "one_to_one": bool(one_to_one),
                "val_f05": float(round(best_score, 4)),
            }

    return best_params
