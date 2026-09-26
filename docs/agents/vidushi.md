# Agent Instructions: Vidushi (AI/ML, Model Owner)

You are the coding agent working for **Vidushi**. Read `AGENTS.md` and `docs/04-TRD.md` first. Do the steps below **in order, one at a time**. Each step ends with a PR, and you must not start the next step until your human confirms the PR is open.

## Your context

- **Role:** model owner. You turn candidate pairs into match probabilities.
- **Files you own (edit only these):** `src/features.py`, `src/train.py`, `tests/test_features.py`, `reports/model_report.md`, and the Vidushi section of `src/config.py` (add keys only).
- **Files you must not edit:** `io.py`, `blocking.py`, `run.py` (Sajal), and `normalize.py`, `split.py`, `evaluate.py`, `decide.py` (Lavanya).
- **Inputs you use:**
  - `src/io.py` (from H1)
  - `data/norm_*.parquet`, or `normalize_df` output (from H4)
  - `data/cands_{trainfit,val,test}.parquet` (from H6)
  - `data/splits.json` and `evaluate.f05_macro` (from H2)
- **Outputs others need from you:**
  - `data/preds_val.parquet` → Lavanya, by H8
  - `data/preds_test.parquet` + `models/lgbm.txt` → Sajal, by H9
- **Contract reminder:** feature columns start with `f_` and are float32 with no NaNs, and the prediction columns are exactly `s1_id, cand_id, prob`.

## Step V1: base features on warm-up pairs (H0.5 → H2)

**Branch:** `vidushi/V1-base-features`

You don't need to wait for blocking. Build and test the feature code on warm-up pairs.

**Do:**
1. Load the train sources and ground truth with `src/io.py`. Until `normalize.py` is merged, build temporary normalized columns inside your test or dev script (not in `features.py`) using lowercase + strip.
2. Build **warm-up pairs:**
   - every ground-truth positive `(s1_id, cand_id)`
   - plus `WARMUP_NEG_PER_POS` random negatives per positive, drawn from S2+S3 records with the same country (seed = `SEED`)
3. In `src/features.py`, implement:
   - `fit_feature_context(all_norm)`: fit one char TF-IDF (`char_wb`, (2,4)) on names and one on addresses over all records, plus token document-frequency tables for IDF.
   - `build_features(pairs, s1_norm, pool_norm, ctx)`: join the pairs to both sides' normalized columns and compute the name features from TRD §5.3 first (`f_name_jw`, `f_name_tsr`, `f_name_tset`, `f_name_tfidf`, `f_name_exact_nosuffix`).
     - Use `rapidfuzz.distance.JaroWinkler.similarity` and `rapidfuzz.fuzz.token_sort_ratio` / `token_set_ratio` (divide by 100).
     - Compute TF-IDF cosine row-wise as the sum of the elementwise product of L2-normalized sparse rows. This is vectorized, so don't loop in Python over matrices.
   - `FEATURE_COLUMNS`: the list of all `f_` columns, in a fixed order.
4. `tests/test_features.py`:
   - identical records give `f_name_jw == 1` and `f_name_tfidf ≈ 1`
   - there are no NaNs
   - the output has the columns `s1_id, cand_id, *FEATURE_COLUMNS`

**Verify:** tests pass, and a quick LightGBM fit on the warm-up pairs gives a sensible AUC (> 0.9). Post the AUC in chat.

**PR:** `feat(features): base name features and feature context (V1)`. Reviewer: Sajal.

## Step V2: full feature set (H2 → H5)

**Branch:** `vidushi/V2-full-features`

**Do:** add the remaining pairwise features from TRD §5.3.
- **Address:** `f_addr_jw`, `f_addr_tsr`, `f_addr_tfidf`, `f_addr_idf_overlap`.
- **Name IDF overlap:** `f_name_idf_overlap` = sum of IDF(shared tokens) ÷ sum of IDF(union tokens).
- **Structured:** `f_postcode_match` and `f_house_match`, each 1 if equal, 0 if different, −1 if either is empty.
- **Lengths:** `f_name_len_diff` and `f_addr_len_diff`.
- **Source:** `f_is_s3` = 1 if `cand_id` starts with `S3-`.
- **No raw country feature.** France is unseen, so use `f_same_country` (0/1) if you want one.
- After Lavanya's `normalize.py` is merged (H4), switch your dev script to `normalize_df` and delete the temporary normalization.

**Verify:** tests are updated, there are no NaNs, and the warm-up AUC improves or holds. Post it in chat.

**PR:** `feat(features): address, structured, IDF and length features (V2)`. Reviewer: Sajal.

## Step V3: group features (H5 → H6)

**Branch:** `vidushi/V3-group-features`

**Do:** implement `add_group_features(feats)` using `f_name_tfidf` as the base score (TRD §5.4):
- `f_grp_rank_in_s1`: `feats.groupby("s1_id")["f_name_tfidf"].rank(ascending=False, method="min")`
- `f_grp_gap_to_top`: the group max minus the value
- `f_grp_n_s1_competing`: `feats.groupby("cand_id")["s1_id"].transform("nunique")`
- `f_grp_rank_in_cand`: the rank of the S1 among all S1s for that `cand_id`, by the base score

Group features only make sense on **real candidate sets,** so compute them after blocking and never on warm-up pairs. Add them to `FEATURE_COLUMNS`.

**Test:** a small hand-made frame with known ranks and gaps.

**PR:** `feat(features): group features (rank, gap, competition) (V3)`. Reviewer: Lavanya.

## Step V4: train on real candidates → `preds_val` (H6 → H8)

**Branch:** `vidushi/V4-train-val`

**Do:**
1. When Sajal posts "cands ready", load `cands_trainfit` and `cands_val`, then build features and group features and save `data/feats_{trainfit,val}.parquet`.
2. In `src/train.py`, implement `make_labels`, `train_model`, `predict`, `save_model` and `load_model` (TRD §3.4). Use `LGBM_PARAMS` and `LGBM_ROUNDS` from config.
3. Train on trainfit and predict val, then write `data/preds_val.parquet` with the columns `s1_id, cand_id, prob`.
4. Get a quick F0.5 by applying `threshold = 0.5` yourself and calling `evaluate.f05_macro`. This is only for comparing model versions, since Lavanya owns the real decision layer.
5. `reports/model_report.md` includes:
   - positive rate in the candidates
   - val AUC
   - quick F0.5 at 0.5
   - the top 15 features by gain

**Handoff:** post "preds_val ready + numbers" in chat. Lavanya starts tuning.

**PR:** `feat(train): LightGBM training and val predictions (V4)`. Reviewer: Sajal.

## Step V5: final model → `preds_test` (H8 → H9)

**Branch:** `vidushi/V5-final-model`

**Do:**
1. Refit on **all train S1s**, meaning the trainfit and val candidates together, with the same params and rounds.
2. Save the model to `models/lgbm.txt`.
3. Build test features from `cands_test` and write `data/preds_test.parquet`.
4. Make sure the functions work when `run.py` calls them, because Sajal's `--split test` must reproduce this output.

**Handoff:** post "preds_test ready" to Sajal.

**PR:** `feat(train): final refit and test predictions (V5)`. Reviewer: Sajal.

## Step V6: France generalization check (H9 → H10)

**Branch:** `vidushi/V6-country-check`

**Do:**
1. Train only on US S1 pairs and evaluate on India S1 pairs (quick F0.5 at 0.5). Then do the reverse.
2. Compare against the mixed-country val score.
3. If some feature clearly hurts cross-country performance (check the importance shift), try dropping it and report the effect on the normal val score.
4. Record the results in `reports/model_report.md`. Change `FEATURE_COLUMNS` **only** if the normal val F0.5 does not drop.

**PR:** `docs(model): cross-country generalization check (V6)`. Reviewer: Lavanya.

## Step V7 (optional, only if time remains before H10): embedding feature

- Use a small multilingual sentence-embedding model with an **MIT or Apache 2.0** licence, and check its model card.
- Add `f_name_emb_cos` and `f_addr_emb_cos`.
- Merge **only if** val F0.5 improves by at least 0.005, and write the model name and licence in the report.

## Never do

- Use the test set labels (there are none) or any external data.
- Tune the decision threshold for the final submission. That's Lavanya's `decide.py`.
- Change the column names or function signatures in TRD §3.
