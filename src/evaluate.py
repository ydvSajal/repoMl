"""Scorer and evaluation utilities (owner: Lavanya, task L3)."""
from typing import Union
import pandas as pd


def f05_entity(pred: Union[set[str], list[str]], truth: Union[set[str], list[str]]) -> float:
    """Calculate F0.5 for a single S1 entity.
    
    Formula per TRD §5.7:
    P = |pred ∩ truth| / |pred| (0 if pred empty)
    R = |pred ∩ truth| / |truth| (0 if truth empty)
    if truth empty: score = 1.0 if pred empty else 0.0
    elif P + R == 0: score = 0.0
    else: score = 1.25 * P * R / (0.25 * P + R)
    """
    p_set = set(pred)
    t_set = set(truth)

    if len(t_set) == 0:
        return 1.0 if len(p_set) == 0 else 0.0

    if len(p_set) == 0:
        return 0.0

    tp = len(p_set & t_set)
    if tp == 0:
        return 0.0

    precision = tp / len(p_set)
    recall = tp / len(t_set)

    denominator = 0.25 * precision + recall
    if denominator == 0:
        return 0.0

    return (1.25 * precision * recall) / denominator


def f05_macro(
    pred: dict[str, Union[set[str], list[str]]],
    truth: dict[str, Union[set[str], list[str]]],
) -> float:
    """Averages F0.5 over all keys of truth, treating missing pred as empty."""
    if not truth:
        return 0.0

    scores = [
        f05_entity(pred.get(s1_id, []), t_matches)
        for s1_id, t_matches in truth.items()
    ]
    return sum(scores) / len(scores)


def blocking_recall(
    cands: Union[pd.DataFrame, dict[str, Union[set[str], list[str]]]],
    truth: dict[str, Union[set[str], list[str]]],
) -> dict[str, float]:
    """Calculate blocking quality metrics per TRD §3.4 and §5.7.
    
    Returns:
        pair_recall: true pairs in candidates ÷ all true pairs
        f05_ceiling: macro F0.5 if pred were truth ∩ candidates
        avg_cands: mean candidates per S1
        pct_s1_zero_cands: % of S1 entities with zero candidates
    """
    if isinstance(cands, pd.DataFrame):
        s1_col = "s1_id" if "s1_id" in cands.columns else "source1_entity_id"
        cand_col = "cand_id" if "cand_id" in cands.columns else "candidate_entity_id"
        cands_dict: dict[str, set[str]] = {}
        for s1, group in cands.groupby(s1_col)[cand_col]:
            cands_dict[str(s1)] = set(group.astype(str))
    else:
        cands_dict = {str(k): set(v) for k, v in cands.items()}

    total_true_pairs = 0
    recovered_true_pairs = 0
    zero_cands_count = 0
    total_cands = 0
    ceiling_preds: dict[str, set[str]] = {}

    for s1_id, t_matches in truth.items():
        t_set = set(t_matches)
        c_set = cands_dict.get(s1_id, set())

        total_true_pairs += len(t_set)
        recovered_true_pairs += len(t_set & c_set)
        total_cands += len(c_set)

        if len(c_set) == 0:
            zero_cands_count += 1

        ceiling_preds[s1_id] = t_set & c_set

    num_entities = len(truth)
    pair_rec = (recovered_true_pairs / total_true_pairs) if total_true_pairs > 0 else 1.0
    f05_ceil = f05_macro(ceiling_preds, truth)
    avg_cands = (total_cands / num_entities) if num_entities > 0 else 0.0
    pct_zero = (zero_cands_count / num_entities) if num_entities > 0 else 0.0

    return {
        "pair_recall": pair_rec,
        "f05_ceiling": f05_ceil,
        "avg_cands": avg_cands,
        "pct_s1_zero_cands": pct_zero,
    }


def evaluation_metrics(cands, truth):
    """Alias for blocking_recall for compatibility."""
    return blocking_recall(cands, truth)