# Agent Instructions: Lavanya (Data Science, Data and Evaluation Owner)

You are the coding agent working for **Lavanya**. Read `AGENTS.md` and `docs/04-TRD.md` first. Do the steps below **in order, one at a time**. Each step ends with a PR, and you must not start the next step until your human confirms the PR is open.

## Your context

- **Role:** owner of data truth. You decide how we measure (the scorer and the splits), how text is cleaned (the normalizer), and how probabilities become final matches (the decision layer). **The team's F0.5 number comes from your code.**
- **Files you own (edit only these):**
  - `src/normalize.py`, `src/split.py`, `src/evaluate.py`, `src/decide.py`
  - `tests/test_normalize.py`, `tests/test_evaluate.py`, `tests/test_decide.py`
  - `notebooks/`, `reports/eda_findings.md`, `reports/blocking_report.md`
  - `Documentation_template.md`
  - the Lavanya section of `src/config.py` (add keys only)
- **Files you must not edit:** `io.py`, `blocking.py`, `run.py` (Sajal), and `features.py`, `train.py` (Vidushi).
- **Others depend on you:**
  - EDA answers (H1)
  - `splits.json` + `evaluate.py` (H2), which everyone uses
  - `normalize.py` (H4) → Sajal
  - `decision_params.json` (H9) → Sajal
- **You depend on:** `io.py` (H1), `cands_*` (H6) and `preds_val` (H8).

## Step L1: EDA (H0.5 → H1)

**Branch:** `lavanya/L1-eda`

**Do:** in `notebooks/eda.ipynb`, load the data with `src/io.py` (or with `pd.read_csv(..., sep="\t", dtype=str, keep_default_na=False)` if `io.py` is not merged yet), then answer these questions in `reports/eda_findings.md`:

1. **Row counts** per source, per country.
2. **Singletons:** what % of train S1 entities have an empty match list, overall and per country?
3. **One-to-one check (most important):** does any S2/S3 ID appear in more than one S1's match list?
   ```python
   from collections import Counter
   c = Counter(i for ids in truth.values() for i in ids)
   multi = [k for k, v in c.items() if v > 1]
   ```
   Report `len(multi)`. If it is 0, the one-to-one assignment is safe.
4. **Match counts:** the distribution of matches per S1 (0, 1, 2, 3+), and the split between S2 and S3.
5. **Noise examples:** 10 matched pairs showing the typical noise, covering name suffixes, abbreviations, landmark addresses and missing PINs.
6. **Postcode coverage:** how often is a 5- or 6-digit code present in addresses, per source?

**Handoff at H1:** post answers to questions 2, 3 and 4 in the team chat immediately.

**PR:** `docs(eda): findings and one-to-one check (L1)`. Reviewer: Sajal.

## Step L2: splits (H1 → H1.5)

**Branch:** `lavanya/L2-L3-splits-scorer` (L2 and L3 can share one PR)

**Do:** implement `make_splits(truth, s1, val_fraction, seed)` in `src/split.py`:
- Split **by S1 ID**: 80% train, 20% val.
- Stratify by `country × is_singleton` so val has the same mix as train.
- Write `data/splits.json` as `{"train": [...], "val": [...], "seed": 42}`.
- Add a `__main__` block so `python -m src.split` creates the file.
- Commit `data/splits.json` (allowed by `.gitignore`) so everyone scores on the identical split.

## Step L3: scorer + blocking recall (H1.5 → H2)

**Do:** implement in `src/evaluate.py`, exactly per TRD §5.7:
- `f05_entity(pred, truth)`, with the singleton rules
- `f05_macro(pred, truth)`, which averages over **all keys of `truth`** and treats a missing pred as empty
- `blocking_recall(cands, truth)`, which returns:
  - `pair_recall`: true pairs in the candidates ÷ all true pairs
  - `f05_ceiling`: macro F0.5 if the pred were `truth ∩ candidates`
  - `avg_cands`: mean candidates per S1
  - `pct_s1_zero_cands`

**`tests/test_evaluate.py` must include:**
- **PDF example:** pred `{S2-00047, S2-00193, S3-00812}` against truth `{S2-00047, S3-00812}` → 0.714 (±0.001).
- **Singletons:** a singleton predicted empty scores 1.0, and a singleton predicted `{X}` scores 0.0.
- **Non-empty truth with empty pred** scores 0.0.

**Handoff at H2:** post "scorer + splits merged, use only `src.evaluate`".

**PR:** `feat(eval): splits and official-equivalent F0.5 scorer (L2, L3)`. Reviewer: Vidushi.

## Step L4: normalizer (H2 → H4)

**Branch:** `lavanya/L4-normalizer`

**Do:** implement every function in TRD §3.4 (normalize section), following TRD §5.1.
- Put the constants at the top of the file: `LEGAL_SUFFIXES` (set), `ADDRESS_ABBREVIATIONS` (dict) and `POSTCODE_PATTERNS`.
- **Use your EDA examples** to grow these lists. Every rule you add should come from a real pattern in the data.
- **No `if country == ...` anywhere.** The rules must work for unseen countries, including France.
- `normalize_df(df)` returns exactly `entity_id, country, name_clean, name_nosuffix, addr_clean, postcode, house_no`, with no NaNs.

**`tests/test_normalize.py` must cover:**
- `"Sharma Traders Pvt. Ltd."` → nosuffix `"sharma traders"`
- `"Smith & Sons Corp"` → `"smith and sons"`
- `"Société Générale SARL"` → `"societe generale"`
- `"12 MG Rd, Near SBI ATM, Bengaluru 560 001"` → postcode `"560001"`, house `"12"`, address contains `"road"`
- `"500 Main St, Springfield, IL 62701-1234"` → postcode `"62701"`
- `"15 av. des Champs-Élysées, 75008 Paris"` → postcode `"75008"`, address contains `"avenue"` and `"champs elysees"`

**Handoff at H4:** tell Sajal "normalizer merged".

**PR:** `feat(normalize): country-agnostic name and address normalization (L4)`. Reviewer: Sajal.

## Step L5: blocking analysis (H5 → H7)

**Branch:** `lavanya/L5-blocking-report`

**Do:** when Sajal posts "cands ready", write `reports/blocking_report.md` containing:
1. `blocking_recall(cands_val, truth_val)`: all 4 numbers.
2. **Recall by blocker:** which blockers found each true pair, using the `blockers` column. How many true pairs are found by only one blocker?
3. **Missed matches:** list 20 true pairs that no blocker found, and group them by cause (name very different, different postcode, the name is a DBA or trade name, and so on).
4. **Top 3 suggested fixes, ranked by cheapness.** Suggestions can only be config changes (top-k values) or normalizer rule additions, which go in your own PR.

**Handoff:** send the fixes to Sajal in chat. Make any normalizer fixes in a new PR, `lavanya/L5b-normalizer-fixes`.

**PR:** `docs(blocking): recall analysis and fixes (L5)`. Reviewer: Sajal.

## Step L6: decision layer + tuning (H7 → H9)

**Branch:** `lavanya/L6-decide`

**Do:**
1. Implement `decide(preds, threshold, min_top, one_to_one)` per TRD §5.6. The return value must contain **only S1 IDs present in `preds`**. `run.py` fills in the missing S1s as empty.
2. Implement `tune(preds_val, truth_val)`: grid-search `THRESHOLD_GRID × MIN_TOP_GRID`, with `one_to_one` fixed by the L1 finding. Score with `f05_macro` over **all val S1 IDs**, including S1s that had no candidates. Return the best params plus `val_f05`.
3. When Vidushi posts "preds_val ready", run the tuning, save `data/decision_params.json`, and commit it. `.gitignore` already allows this file and `data/splits.json`, so the test run and the final package can reproduce your decisions.
4. Report the results in the PR description:
   - the best params and val F0.5
   - F0.5 at the plain 0.5 threshold, for comparison
   - F0.5 on singletons only vs non-singletons only

**`tests/test_decide.py`:**
- One-to-one: the same cand for two S1s means only the higher-prob S1 keeps it.
- `min_top` above the max prob returns an empty list for that S1.

**H8 sync:** show the tuning results on screen, and all three agree on the final params.

**PR:** `feat(decide): decision layer and tuned params (L6)`. Reviewer: Sajal.

## Step L7: methodology document (H9 → H12)

**Branch:** `lavanya/L7-documentation`

**Do:** fill in `Documentation_template.md` using the real numbers from `reports/`. It must describe:
1. **Methodology:** the pipeline overview (you can copy the diagram idea from `05-system-architecture.md`).
2. **Candidate generation / blocking:** the blockers, top-k values, country fallback, pair recall, F0.5 ceiling, and average candidates.
3. **Model architecture and feature engineering:** the LightGBM params, every feature group with a one-line reason, and the top features by importance (from Vidushi's report).
4. **Decision layer:** one-to-one assignment (with the EDA evidence), threshold and min_top tuning, and singleton handling.
5. **Validation:** the split method, val F0.5, and the val vs leaderboard gap (from Sajal's log).
6. **Generalization to France:** our country-agnostic design, plus Vidushi's cross-country check.
7. **Compliance:** no external data, the model licence, and reproducibility steps.

**PR:** `docs: methodology document (L7)`. Reviewers: Sajal and Vidushi.

## Never do

- Branch any rule on specific country names.
- Tune on the test set or look at test outputs to pick params.
- Change column names or function signatures in TRD §3.
