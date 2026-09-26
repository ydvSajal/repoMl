# 04 — Technical Requirements Document (TRD)

**Goal:** `python -m src.run --split test` turns the 3 test TSVs into `output/matching_results.tsv` and `output/candidate_pairs.tsv`, using the stages normalize → block → features → LightGBM → decide. Every stage is tuned against one shared macro F0.5 scorer on a held-out validation split.

AI agents must follow §3 exactly. Anything not specified here is the owner's choice.

## 1. Stack

| Need | Choice | License note |
|---|---|---|
| Language | Python 3.10–3.12 | – |
| Dataframes | pandas + pyarrow (parquet) | BSD / Apache 2.0 |
| Vectors | scikit-learn `TfidfVectorizer` + scipy sparse | BSD |
| String similarity | rapidfuzz | MIT |
| Matcher model | LightGBM | **MIT (final model)** |
| Accent stripping | stdlib `unicodedata` (NFKD) | No extra dependency |
| Tests | pytest | MIT |
| Optional embeddings (P1) | sentence-transformers + a small multilingual MIT or Apache 2.0 model | Check the model card licence before use |

## 2. Repo layout

See `AGENTS.md` §5 for the full tree and §2 for ownership.

## 3. Data contracts (must follow exactly)

### 3.1 Raw inputs (provided)

| File | Columns |
|---|---|
| `dataset/{train,test}/{split}_source{1,2,3}.tsv` | `entity_id, business_name, business_address, country` |
| `dataset/train/train_ground_truth.tsv` | `source1_entity_id, matched_entity_ids` (comma-separated, may be empty) |

### 3.2 Intermediate files (`data/`, gitignored)

Every column is a string unless noted. Missing values are always the empty string `""`, never NaN.

| File | Owner | Columns |
|---|---|---|
| `data/splits.json` | Lavanya | `{"train": [s1_id, ...], "val": [s1_id, ...], "seed": 42}` |
| `data/norm_{train,test}_s{1,2,3}.parquet` | Sajal runs Lavanya's `normalize_df` | `entity_id, country, name_clean, name_nosuffix, addr_clean, postcode, house_no` |
| `data/cands_{val,trainfit,test}.parquet` | Sajal | `s1_id, cand_id, blockers` (`blockers` = pipe-joined blocker names, e.g. `name\|postcode`) |
| `data/feats_{val,trainfit,test}.parquet` | Vidushi | `s1_id, cand_id, f_*` (float32 features) |
| `data/preds_{val,test}.parquet` | Vidushi | `s1_id, cand_id, prob` (float) |
| `data/decision_params.json` | Lavanya | `{"threshold": float, "min_top": float, "one_to_one": bool, "val_f05": float}` |
| `models/lgbm.txt` | Vidushi | LightGBM model file |

The split names mean:
- `trainfit` is the candidates for the **train-split S1 IDs**, with the pool being all train S2+S3. They are used to fit the model.
- `val` is the candidates for the **val-split S1 IDs**, with the same pool. They are used for scoring and tuning.
- `test` is the candidates for all test S1 IDs, with the pool being all test S2+S3.

### 3.3 Final outputs (`output/`)

| File | Columns | Rules |
|---|---|---|
| `matching_results.tsv` | `source1_entity_id, matched_entity_ids` | One row per test S1; comma-joined IDs, no spaces, no quotes; empty allowed; no duplicates |
| `candidate_pairs.tsv` | `source1_entity_id, candidate_entity_ids` | Same rules; must be a superset of the matches |

### 3.4 Function signatures (do not change without a PR approved by Sajal)

```python
# src/config.py  (Sajal; each owner adds keys in their own section)
SEED: int = 42
PATHS: dict[str, str]  # dataset, data, models, output

# src/io.py  (Sajal)
def read_tsv(path: str) -> pd.DataFrame
def load_sources(split: Literal["train", "test"]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
def load_ground_truth() -> dict[str, set[str]]           # every train S1 id is a key; value may be empty set
def write_id_lists(mapping: dict[str, list[str]], s1_ids: list[str], path: str,
                   id_col: Literal["matched_entity_ids", "candidate_entity_ids"]) -> None
def read_parquet(path: str) -> pd.DataFrame
def write_parquet(df: pd.DataFrame, path: str) -> None

# src/normalize.py  (Lavanya)
def normalize_text(s: str) -> str                        # lowercase, strip accents, & -> and, punctuation -> space, collapse spaces
def normalize_name(s: str) -> tuple[str, str]            # (name_clean, name_nosuffix)
def normalize_address(s: str) -> str
def extract_postcode(raw_address: str) -> str            # "" if none
def extract_house_no(raw_address: str) -> str            # "" if none
def normalize_df(df: pd.DataFrame) -> pd.DataFrame       # raw source df -> contract columns of 3.2

# src/split.py  (Lavanya)
def make_splits(truth: dict[str, set[str]], s1: pd.DataFrame, val_fraction: float, seed: int) -> dict[str, list[str]]

# src/evaluate.py  (Lavanya)
def f05_entity(pred: set[str], truth: set[str]) -> float
def f05_macro(pred: dict[str, set[str]], truth: dict[str, set[str]]) -> float   # averages over truth keys; missing pred = empty
def blocking_recall(cands: pd.DataFrame, truth: dict[str, set[str]]) -> dict[str, float]
    # keys: pair_recall, f05_ceiling, avg_cands, pct_s1_zero_cands

# src/blocking.py  (Sajal)
def generate_candidates(s1_norm: pd.DataFrame, pool_norm: pd.DataFrame) -> pd.DataFrame  # s1_id, cand_id, blockers

# src/features.py  (Vidushi)
FEATURE_COLUMNS: list[str]
def fit_feature_context(all_norm: pd.DataFrame) -> "FeatureContext"   # IDF tables + fitted vectorizers
def build_features(pairs: pd.DataFrame, s1_norm: pd.DataFrame, pool_norm: pd.DataFrame,
                   ctx: "FeatureContext") -> pd.DataFrame            # s1_id, cand_id, f_* (no group feats)
def add_group_features(feats: pd.DataFrame) -> pd.DataFrame          # adds f_grp_* columns

# src/train.py  (Vidushi)
def make_labels(pairs: pd.DataFrame, truth: dict[str, set[str]]) -> pd.Series   # int 0/1
def train_model(X: pd.DataFrame, y: pd.Series) -> "lgb.Booster"
def predict(model: "lgb.Booster", X: pd.DataFrame) -> np.ndarray
def save_model(model, path: str) -> None
def load_model(path: str) -> "lgb.Booster"

# src/decide.py  (Lavanya)
def decide(preds: pd.DataFrame, threshold: float, min_top: float, one_to_one: bool) -> dict[str, list[str]]
def tune(preds_val: pd.DataFrame, truth_val: dict[str, set[str]]) -> dict   # returns decision_params

# src/run.py  (Sajal)
# CLI: python -m src.run --split {val,test} [--stage {normalize,block,features,train,decide,all}] [--baseline] [--force]

# src/package.py  (Sajal)
# CLI: python -m src.package --team <team_name>
```

## 4. Configuration (`src/config.py`)

Tunable numbers live here and nowhere else.

```python
SEED = 42
PATHS = {"dataset": "dataset", "data": "data", "models": "models", "output": "output"}

# --- Sajal: blocking ---
BLOCK_NAME_TOPK = 20
BLOCK_ADDR_TOPK = 10
BLOCK_CHAR_NGRAM = (2, 4)          # analyzer="char_wb"
BLOCK_RARE_TOKEN_MAX_COUNT = 30    # a token is "rare" if it appears in <= this many pool records
BLOCK_CHUNK_SIZE = 2000            # S1 rows per sparse matmul chunk
BLOCK_MIN_COUNTRY_POOL = 50        # below this, search the whole pool instead of the same country

# --- Vidushi: model ---
WARMUP_NEG_PER_POS = 5
LGBM_PARAMS = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 63,
               "min_data_in_leaf": 20, "feature_fraction": 0.9, "bagging_fraction": 0.9,
               "bagging_freq": 1, "seed": 42, "verbose": -1}
LGBM_ROUNDS = 400

# --- Lavanya: eval / decide ---
VAL_FRACTION = 0.2
THRESHOLD_GRID = [round(0.30 + 0.05 * i, 2) for i in range(14)]   # 0.30 .. 0.95
MIN_TOP_GRID = [0.0, 0.4, 0.5, 0.6, 0.7]
```

## 5. Algorithms

### 5.1 Normalization (Lavanya)

1. `normalize_text`: lowercase, then Unicode NFKD with combining marks removed (é → e), then `&` → ` and `, then every non-alphanumeric character → space, then collapse repeated whitespace.
2. **Legal suffixes** are removed only for `name_nosuffix`, and only as whole tokens at the end or anywhere. The list includes: `pvt private ltd limited llp llc inc incorporated corp corporation co company plc gmbh sarl sas sa eurl sasu sci`. Keep the list in a module-level constant.
3. **Address abbreviations** are expanded to one canonical form, for example `rd→road st→street ave→avenue blvd→boulevard av→avenue bd→boulevard nr→near opp→opposite mg→mahatma gandhi` (the last one is optional, so test it). Keep these in a dict constant.
4. **Postcode:** a regex on the raw address. A 6-digit Indian PIN may appear as `560 001`, so join those. A 5-digit code covers US and France, and ZIP+4 is trimmed to 5 digits. Take the last match.
5. **House number:** the first token that starts with a digit and is not the postcode.
6. **Never branch on country for normalization.** All rules are applied to every record.

### 5.2 Blocking (Sajal)

For each country present in S1:
1. The pool is the S2+S3 records with the same country. If the pool is smaller than `BLOCK_MIN_COUNTRY_POOL`, or the country is unseen, use the whole pool.
2. **Name blocker:** fit a char TF-IDF (`char_wb`, `BLOCK_CHAR_NGRAM`) on `name_nosuffix` over S1 ∪ pool. Compute cosine as a chunked sparse matmul, `S1_chunk @ pool.T`, and keep the top `BLOCK_NAME_TOPK` per row with `np.argpartition`.
3. **Address blocker:** the same method on `addr_clean`, keeping the top `BLOCK_ADDR_TOPK`.
4. **Postcode + rare-token blocker:** a pair qualifies if the postcodes are equal and the two names share at least one rare token (a token appearing in ≤ `BLOCK_RARE_TOKEN_MAX_COUNT` pool records).
5. Take the union of all blockers, and record which blockers found each pair in the `blockers` column.
6. **Every S1 ID must appear in the output,** even with zero candidates. Downstream code handles this through the `s1_ids` list.

### 5.3 Pairwise features (Vidushi): `f_` prefix, float32

| Group | Features |
|---|---|
| Name | `f_name_jw` (Jaro-Winkler on name_nosuffix), `f_name_tsr` (token_sort_ratio/100), `f_name_tset` (token_set_ratio/100), `f_name_tfidf` (char TF-IDF cosine), `f_name_exact_nosuffix` (0/1), `f_name_idf_overlap` (IDF-weighted Jaccard of tokens) |
| Address | `f_addr_jw`, `f_addr_tsr`, `f_addr_tfidf`, `f_addr_idf_overlap` |
| Structured | `f_postcode_match` (1 equal, 0 different, −1 either missing), `f_house_match` (same encoding) |
| Length | `f_name_len_diff`, `f_addr_len_diff` (absolute char difference) |
| Source | `f_is_s3` (0/1 from the `cand_id` prefix) |

Do **not** include a raw country feature, because France is unseen. If needed, use `f_same_country` (0/1).

### 5.4 Group features (Vidushi): `f_grp_`

These are computed per candidate set, using `f_name_tfidf` as the base score:
- `f_grp_rank_in_s1`: the rank of the candidate among its S1's candidates (1 = best)
- `f_grp_gap_to_top`: the top score in its S1 minus this score
- `f_grp_n_s1_competing`: how many S1 IDs have this `cand_id` as a candidate
- `f_grp_rank_in_cand`: the rank of this S1 among all S1s competing for this cand (1 = best)

### 5.5 Training (Vidushi)

- **Labels:** 1 if `cand_id ∈ truth[s1_id]`, else 0.
- **Validation run:** fit on `feats_trainfit` and predict `feats_val`, using `LGBM_ROUNDS` and a fixed seed.
- **Test run:** refit on the train and val S1s together (all train candidates, same rounds), then predict `feats_test`.
- **Warm-up (before the real candidates exist):** GT positives plus `WARMUP_NEG_PER_POS` random negatives from the same country, used only to develop the feature code.

### 5.6 Decision layer (Lavanya)

1. **One-to-one assignment,** only if EDA confirms each S2/S3 ID maps to at most one S1: for each `cand_id`, keep only the row with the highest `prob`.
2. **Per S1:** if `max(prob) < min_top`, return an empty list. Otherwise keep every candidate with `prob ≥ threshold`.
3. **Tuning:** grid-search `THRESHOLD_GRID × MIN_TOP_GRID` on val, maximizing `f05_macro`. Save the best params to `data/decision_params.json`.

### 5.7 Scorer (Lavanya)

```
P = |pred ∩ truth| / |pred|   (0 if pred empty)
R = |pred ∩ truth| / |truth|  (0 if truth empty)
if truth empty: score = 1.0 if pred empty else 0.0
elif P + R == 0: score = 0.0
else: score = 1.25·P·R / (0.25·P + R)
```

The macro score is the average over **all S1 IDs in the evaluated split**. Test: the PDF example must give 0.714.

## 6. `run.py` stages (Sajal)

| Stage | Reads | Writes |
|---|---|---|
| normalize | raw TSVs | `data/norm_*` |
| block | `norm_*`, `splits.json` | `data/cands_{trainfit,val}` (val run) or `cands_test` (test run) |
| features | `norm_*`, `cands_*` | `data/feats_*` |
| train | `feats_*`, ground truth | `models/lgbm.txt`, `data/preds_*` |
| decide | `preds_*`, `decision_params.json` (test) or tune (val) | val: prints F0.5; test: `output/*.tsv` |

Each stage skips work if its outputs exist, unless `--force` is passed. `--baseline` skips features and training and matches on `f_name_tfidf ≥ 0.8` as the safety submission.

## 7. Testing

| Test file | Must cover |
|---|---|
| `tests/test_io.py` | Empty match list survives a round trip; IDs stay strings; the validator-compatible format |
| `tests/test_normalize.py` | Suffix removal, `&`/`and`, accents (`Société Générale SARL`), PIN `560 001`, ZIP+4 |
| `tests/test_evaluate.py` | The PDF example = 0.714; singleton empty = 1.0; singleton non-empty = 0.0 |
| `tests/test_blocking.py` | Every S1 in the output; no S1 IDs in candidates; exact duplicate names found |
| `tests/test_features.py` | No NaN; all `FEATURE_COLUMNS` present; identical records give maximum similarity |
| `tests/test_decide.py` | One-to-one keeps the best S1 only; `min_top` produces an empty list |

## 8. Performance budget

- A full val run should finish in 20 minutes or less on a 16 GB laptop.
- If it runs out of memory, lower `BLOCK_CHUNK_SIZE` and never use dense matrices.
- Cache every stage output as parquet.
