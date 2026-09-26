"""Pipeline entry point (owner: Sajal, tasks S2/S5).

python -m src.run --split {val,test} [--stage {normalize,block,features,train,decide,all}] [--baseline] [--force]
"""
import argparse
import os

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from src.config import BASELINE_MATCH_COS, BASELINE_TOPK, BLOCK_CHAR_NGRAM, BLOCK_MIN_COUNTRY_POOL, PATHS
from src.io import load_sources, write_id_lists


def run_baseline() -> None:
    """Safety submission: top-k S2+S3 by name char TF-IDF cosine per test S1, match if cosine >= threshold."""
    s1, s2, s3 = load_sources("test")
    pool = pd.concat([s2, s3], ignore_index=True)
    names = lambda df: df["business_name"].str.lower().str.strip()
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=BLOCK_CHAR_NGRAM).fit(pd.concat([names(s1), names(pool)]))

    cands, matches = {}, {}
    for country, q in s1.groupby("country"):
        p = pool[pool["country"] == country]
        if len(p) < BLOCK_MIN_COUNTRY_POOL:
            p = pool
        # brute cosine on sparse rows; sklearn chunks the distance matrix so memory stays bounded
        nn = NearestNeighbors(n_neighbors=min(BASELINE_TOPK, len(p)), metric="cosine").fit(vec.transform(names(p)))
        dist, idx = nn.kneighbors(vec.transform(names(q)))
        ids = p["entity_id"].to_numpy()
        for s1_id, d, i in zip(q["entity_id"], dist, idx):
            cands[s1_id] = list(ids[i])
            matches[s1_id] = list(ids[i[d <= 1 - BASELINE_MATCH_COS]])

    s1_ids = list(s1["entity_id"])
    write_id_lists(matches, s1_ids, os.path.join(PATHS["output"], "matching_results.tsv"), "matched_entity_ids")
    write_id_lists(cands, s1_ids, os.path.join(PATHS["output"], "candidate_pairs.tsv"), "candidate_entity_ids")
    print(f"baseline: {len(s1_ids)} S1, {sum(map(len, matches.values()))} matches, "
          f"{sum(map(len, cands.values()))} candidates -> {PATHS['output']}/")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["val", "test"], required=True)
    p.add_argument("--stage", choices=["normalize", "block", "features", "train", "decide", "all"], default="all")
    p.add_argument("--baseline", action="store_true")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    if args.baseline:
        if args.split != "test":
            p.error("--baseline only supports --split test (val scoring needs evaluate.py)")
        return run_baseline()
    raise NotImplementedError("S5 (Sajal)")


if __name__ == "__main__":
    main()
