"""Tunable numbers live here and nowhere else (TRD §4).

Owned by Sajal. Each owner may add keys in their own section only.
"""

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
