# 01 — Timeline (one-day build)

H0 is the moment we start. Write the real clock time next to each H mark at kickoff, for example H0 = 09:00 and H12 = 21:00.

## Checkpoints (gates)

| Mark | Gate | Pass criteria |
|---|---|---|
| H1 | Foundation | Repo on GitHub with branch protection; everyone has run `pytest -q`; Lavanya's EDA findings posted in chat |
| H2 | Safety net | Scorer and `splits.json` merged; the baseline submission passes the validator and is uploaded to the leaderboard |
| H4 | Normalizer | `src/normalize.py` merged; Sajal switches blocking to use it |
| H6 | Candidates | `data/cands_val.parquet` and `data/cands_test.parquet` exist; blocking recall is reported; Vidushi starts training on real candidates |
| H8 | End-to-end | `python -m src.run --split val` prints a val F0.5; all three agree the threshold |
| H10 | Code freeze | Test predictions uploaded; only docs, packaging and bug fixes after this point |
| H12 | Submitted | The zip package is built from a clean clone, validated and submitted |

## Hour-by-hour

| Hour | Sajal (Backend & Coding) | Vidushi (AI/ML) | Lavanya (Data Science) |
|---|---|---|---|
| H0–H0.5 | Kickoff: walk everyone through `AGENTS.md` and the contracts (TRD §3). S0: create repo, branch protection, skeleton | Setup: clone, venv, install, `pytest -q` | Setup: clone, venv, install, `pytest -q` |
| H0.5–H1 | S1: `io.py` + `config.py` | V1: base pairwise features on GT positives + random negatives | L1: EDA (noise types, singleton %, one-to-one check) |
| H1–H2 | S2: baseline submission → validator → upload | V1 continued | L2: `splits.json`, L3: F0.5 scorer + blocking-recall util |
| H2–H4 | S3: blocking v1 (simple normalization) | V2: full feature set | L4: normalizer rules → `normalize.py` |
| H4–H5 | S4: plug in Lavanya's normalizer, blocking v2 | V2 continued | L4 fixes from Sajal's feedback; start L5 |
| H5–H6 | S4: write `cands_val` + `cands_test`; unit tests | V3: group features | L5: blocking recall report + missed-match analysis |
| H6–H7 | S5: `run.py` end-to-end wiring | V4: train LightGBM on real candidates → `preds_val` | L5 continued → feedback to Sajal |
| H7–H8 | S5 continued; apply L5 blocking fixes if cheap | V4: feature importance, iterate once | L6: `decide.py` + threshold tuning |
| **H8** | **Sync call (15 min): val F0.5 on screen, lock the decision params** | | |
| H8–H9 | S6: test run, validator, upload, log | V5: final model on all train S1s → `preds_test` | L6: final decision params committed |
| H9–H10 | S6 continued; tag commit `v1-final` | V6: France check (train US → validate India) | L7: start `Documentation_template.md` |
| **H10** | **Code freeze** | | |
| H10–H11 | S7: `package.py`, README, requirements | V7 (optional, only if val gain and time): embedding feature, as a separate PR that is not merged after freeze unless it improves val by ≥ 0.005 | L7: write methodology, blocking stats, results |
| H11–H12 | S8: clean-clone reproduction → zip → submit | Review docs for model section | Final docs PR |

## If behind schedule

| Situation at | Do this |
|---|---|
| H2, baseline not uploaded | Everyone stops and helps Sajal. A valid submission comes first |
| H6, blocking recall < 90% | Raise the top-k limits in `config.py`. Do not add new retrievers |
| H8, the model is worse than the baseline | Ship the baseline + normalizer + one-to-one rule + a tuned threshold |
| H10, still iterating | Stop. Freeze. The best leaderboard commit is the final one |
