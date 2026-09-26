# Agent Instructions: Sajal (Backend & Coding, Repo Owner)

You are the coding agent working for **Sajal**. Read `AGENTS.md` and `docs/04-TRD.md` first. Do the steps below **in order, one at a time**. Each step ends with a PR, and you must not start the next step until your human says the PR is open.

## Your context

- **Role:** pipeline engineer. You build the plumbing that connects everyone's work: IO, blocking, orchestration, outputs, packaging.
- **Files you own (edit only these):** `src/io.py`, `src/config.py` (Sajal section and shared keys), `src/blocking.py`, `src/run.py`, `src/package.py`, `tests/test_io.py`, `tests/test_blocking.py`, `requirements.txt`, `README.md`, `.gitignore`, `.github/`, `reports/leaderboard_log.md`.
- **Files you must not edit:**
  - Vidushi's: `features.py`, `train.py`
  - Lavanya's: `normalize.py`, `split.py`, `evaluate.py`, `decide.py`
  - You **call** their functions through the signatures in TRD §3.4. You never change them.
- **You depend on:** Lavanya's `normalize_df` (H4), `splits.json` + `evaluate` (H2) and `decide` (H9), and Vidushi's `build_features`, `add_group_features` and `train_model` (H6–H9).
- **Others depend on you:** `io.py` (H1) and `cands_*.parquet` (H6).
- Sajal is the **only person who uploads to the leaderboard.**

## Step S0: repo setup (H0 → H0.5)

**Do:**
1. Create the GitHub repo `amazon-er` as **private** and add Vidushi and Lavanya as collaborators with write access.
2. Add `AGENTS.md`, `CLAUDE.md`, `docs/`, `.github/pull_request_template.md` and `.gitignore` from this docs pack. Create empty `src/__init__.py`, `tests/__init__.py`, `reports/`, and `notebooks/`.
3. Create a placeholder module with the right signature raising `NotImplementedError` for each owner's file, so imports work from the start. Owners will replace them.
4. Add `requirements.txt` (list in `06-deployment-and-requirements.md`), install it, and pin the versions.
5. This first commit is the **only** direct push to `main`. Right after it, enable branch protection on `main` under Settings → Branches → Add rule:
   - require a pull request before merging
   - require 1 approval
   - do not allow bypassing

   Branch protection on a **private** repo needs GitHub Pro. Students get Pro free through the GitHub Student Developer Pack, so activate it before S0. If Pro is not available, keep the repo private anyway and enforce the PR-only rule by team agreement, because competition code should not be public before the deadline.
6. Ask a teammate to open a test PR (their name in README) to prove the flow works.

**Verify:** `pytest -q` runs (0 tests is fine), and a direct `git push origin main` is rejected.

## Step S1: `io.py` + `config.py` (H0.5 → H1)

**Branch:** `sajal/S1-io-config`

**Do:**
1. `src/config.py`: copy TRD §4 exactly, with section comments for each owner.
2. `src/io.py`: implement every function in TRD §3.4 (io section). Key details:
   - `read_tsv` uses `pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)`.
   - `load_ground_truth` splits `matched_entity_ids` on `,`, strips spaces, drops empty strings, and returns `dict[s1_id, set]`, including empty sets.
   - `write_id_lists` writes exactly one row for **every** ID in `s1_ids` (in the order given), joins with `,` and no spaces, removes duplicates while keeping order, and writes an empty string for no matches. Use `df.to_csv(path, sep="\t", index=False)`, which must not quote anything.
3. Print the row counts of all 6 sources plus the ground truth, and post them in chat. Lavanya and Vidushi need them.
4. `tests/test_io.py`:
   - An empty list round-trips as an empty set.
   - IDs like `S2-00047` stay strings.
   - The written file has exactly 2 columns with the right header.

**Verify:** `pytest -q` passes.

**PR:** `feat(io): TSV/parquet IO and config (S1)`. Reviewer: Lavanya.

## Step S2: baseline submission (H1 → H2)

**Branch:** `sajal/S2-baseline`

**Do:**
1. In `src/run.py`, add a `--baseline` mode:
   - lowercase and strip the names
   - fit a char TF-IDF on the names
   - for each test S1, take the top 10 S2+S3 by cosine, **within the same country** (fall back to all if the country pool is small)
   - candidates = those 10
   - matches = those with cosine ≥ 0.8
2. Write both TSVs to `output/` using `write_id_lists`.
3. Run the validator:
   ```bash
   python3 utils/validate_submission.py --matching output/matching_results.tsv \
     --candidate output/candidate_pairs.tsv --test-dir dataset/test
   ```
4. Upload `matching_results.tsv` to the portal. Create `reports/leaderboard_log.md` with a table: `time | commit | description | val F0.5 | leaderboard F0.5`.

**Verify:** the validator prints PASS, and the leaderboard shows SCORED.

**PR:** `feat(run): baseline submission mode (S2)`. Reviewer: Vidushi.

## Step S3: blocking v1 (H2 → H4)

**Branch:** `sajal/S3-blocking-v1`

**Do:**
1. Implement `generate_candidates(s1_norm, pool_norm)` from TRD §5.2 in `src/blocking.py`. Until Lavanya's normalizer is merged, write a tiny private `_simple_norm` inside `blocking.py` (lowercase + strip) and build temporary `name_nosuffix` / `addr_clean` / `postcode` columns with it.
2. Use a chunked sparse top-k:
   ```python
   sims = s1_chunk_matrix @ pool_matrix.T          # scipy sparse, L2-normalized TF-IDF rows = cosine
   # for each row: idx = np.argpartition(-row_dense, k)[:k] on the row's nonzeros
   ```
   Never convert the full matrix to dense.
3. Keep the per-country loop, with the fallback to the whole pool when the pool size is below `BLOCK_MIN_COUNTRY_POOL`.
4. The output is `s1_id, cand_id, blockers`, deduplicated, and every S1 ID appears through the `s1_ids` list downstream.
5. `tests/test_blocking.py`:
   - a tiny synthetic dataset where exact duplicate names must be found
   - no `S1-` IDs appear in `cand_id`
   - a country with one pool record falls back to the whole pool

**Verify:** tests pass, and the val candidates build in under 10 minutes.

**PR:** `feat(blocking): multi-blocker candidate generation v1 (S3)`. Reviewer: Lavanya.

## Step S4: blocking v2 with the real normalizer (H4 → H6)

**Branch:** `sajal/S4-blocking-v2`

**Do:**
1. After Lavanya's `normalize.py` is merged, remove `_simple_norm` and call `normalize_df` on all 6 sources. Save `data/norm_{train,test}_s{1,2,3}.parquet`.
2. Build `data/cands_trainfit.parquet` (train-split S1s), `data/cands_val.parquet` (val-split S1s) and `data/cands_test.parquet` (all test S1s), using the splits from `data/splits.json`.
3. Run `evaluate.blocking_recall(cands_val, truth_val)` and post pair_recall, f05_ceiling and avg_cands in chat.
4. If pair_recall is below 95%, raise the top-k values in `config.py` (never add new retrievers today) and re-run.

**Verify:** all three parquet files exist, and recall is posted.

**PR:** `feat(blocking): use shared normalizer, write candidate sets (S4)`. Reviewer: Lavanya.

**Handoff:** tell Vidushi and Lavanya "cands ready" with the recall numbers.

## Step S5: `run.py` end to end (H6 → H8)

**Branch:** `sajal/S5-run-pipeline`

**Do:** implement the stages in TRD §6.

```
python -m src.run --split val   # normalize → block → features → train(trainfit) → predict(val) → tune → print F0.5
python -m src.run --split test  # normalize → block → features → train(all train) → predict(test) → decide(saved params) → write TSVs
```

- Each stage skips work if its output exists, unless `--force` is passed.
- Log the timing of each stage.
- Call the teammates' functions only through their public signatures.
- Test mode: `candidate_pairs.tsv` = `cands_test`, and `matching_results.tsv` = `decide(...)`. **Assert that every match is also a candidate.**

**Verify:** the val run prints F0.5, and the test run writes files that pass the validator.

**PR:** `feat(run): end-to-end val and test pipeline (S5)`. Reviewer: Vidushi.

## Step S6: final test run and upload (H8 → H10)

**Branch:** `sajal/S6-final-run` (log update only)

**Do:**
1. After the H8 sync, and once Lavanya's `decision_params.json` and Vidushi's final model are merged, run `python -m src.run --split test --force`.
2. Run the validator, and upload only if it prints PASS.
3. Log the upload in `reports/leaderboard_log.md` with the commit hash (`git rev-parse --short HEAD`).
4. After merge, tag the commit: `git tag v1-final && git push origin v1-final`.

**PR:** `chore(log): final leaderboard upload (S6)`.

## Step S7: packaging, README, requirements (H10 → H11)

**Branch:** `sajal/S7-package`

**Do:**
1. `src/package.py` builds the zip structure in `06-deployment-and-requirements.md` §6. It copies `src/`, `README.md`, `requirements.txt`, `output/*.tsv` and `Documentation_template.md`, and excludes `data/`, `models/` and `dataset/`.
2. `README.md` covers:
   - setup commands
   - dataset placement
   - `python -m src.run --split val` and `--split test`
   - the validator command
   - the expected runtime
   - the model licence (LightGBM, MIT)
   - pinned Python version

**PR:** `feat(package): submission zip builder and README (S7)`. Reviewer: Lavanya.

## Step S8: reproduce and submit (H11 → H12)

**Do:**
1. Clone the repo into a new folder, create a new venv, install, copy the dataset in, and run `python -m src.run --split test`.
2. `diff` the result against the `output/matching_results.tsv` in the tag. The diff must be empty.
3. Run `python -m src.package --team <team_name>` and check the zip listing with `unzip -l`.
4. Submit the zip.

**Done when:** the zip has been submitted and the leaderboard file matches the zip.
