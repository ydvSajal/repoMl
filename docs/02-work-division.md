# 02 — Work Division

The work is split by pipeline stage. Each person owns separate files (`AGENTS.md` §2), so nobody edits the same file and PRs never conflict. People connect only through the contracts in `04-TRD.md` §3.

## Who does what

| Person | Role | In one line |
|---|---|---|
| **Sajal** | Backend & coding | Builds the pipeline plumbing: IO, blocking, `run.py`, outputs, upload, packaging |
| **Vidushi** | AI/ML | Builds the model: pairwise features, group features, LightGBM, predictions |
| **Lavanya** | Data science | Owns data truth: EDA, splits, scorer, normalization rules, blocking analysis, decisions, documentation |

## Sajal: tasks

| ID | Task | Files | Done when | PR by |
|---|---|---|---|---|
| S0 | Repo, branch protection, skeleton, gitignore, PR template | root, `.github/` | Everyone has cloned; a test PR has been merged through review | H0.5 |
| S1 | `io.py` + `config.py` | `src/io.py`, `src/config.py`, `tests/test_io.py` | Reads all 7 TSVs with the contract dtypes; writes ID-list TSVs that pass the validator | H1 |
| S2 | Baseline submission | `src/run.py` (baseline mode) | `output/*.tsv` passes the validator and is uploaded; the score is logged | H2 |
| S3 | Blocking v1 | `src/blocking.py`, `tests/test_blocking.py` | Candidates are generated per country with char TF-IDF top-k on name and address + postcode/rare-token blocks | H4 |
| S4 | Blocking v2 with Lavanya's normalizer | `src/blocking.py` | `data/cands_val.parquet` + `data/cands_test.parquet` written; recall handed to Lavanya | H6 |
| S5 | `run.py` end-to-end | `src/run.py` | `python -m src.run --split val` prints F0.5; `--split test` writes both TSVs | H8 |
| S6 | Test run, validation, upload | `reports/leaderboard_log.md` | Upload logged with commit hash; tag `v1-final` | H10 |
| S7 | Packaging + README + requirements | `src/package.py`, `README.md`, `requirements.txt` | `python -m src.package` builds the exact zip structure | H11 |
| S8 | Clean-clone reproduction + submit | – | A fresh clone reproduces `matching_results.tsv` byte-for-byte | H12 |

## Vidushi: tasks

| ID | Task | Files | Done when | PR by |
|---|---|---|---|---|
| V1 | Base pairwise features on GT positives + random same-country negatives | `src/features.py`, `tests/test_features.py` | `build_features()` works on any `(s1_id, cand_id)` DataFrame | H2 |
| V2 | Full feature set (name, address, postcode, house no., IDF overlap, lengths) | `src/features.py` | All features in TRD §5.3 exist and have no NaNs | H5 |
| V3 | Group features (rank, gap, competition) | `src/features.py` | `add_group_features()` merged with tests | H6 |
| V4 | Train LightGBM on real candidates → `preds_val` | `src/train.py`, `reports/model_report.md` | `data/preds_val.parquet` written; feature importance in the report | H8 |
| V5 | Final model on all train S1s → `preds_test` | `src/train.py` | `data/preds_test.parquet` written | H9 |
| V6 | France generalization check | `reports/model_report.md` | Train-US → val-India score reported; country-specific features dropped if they hurt | H10 |
| V7 | (Optional) multilingual embedding cosine feature | `src/features.py` | Merged only if val F0.5 improves ≥ 0.005 | H10 |

## Lavanya: tasks

| ID | Task | Files | Done when | PR by |
|---|---|---|---|---|
| L1 | EDA | `notebooks/eda.ipynb`, `reports/eda_findings.md` | Singleton %, noise examples, **one-to-one answer** posted to chat by H1 | H1 |
| L2 | Splits | `src/split.py`, `data/splits.json` | 80/20 split by S1, stratified by country × singleton | H2 |
| L3 | Scorer + blocking recall | `src/evaluate.py`, `tests/test_evaluate.py` | Matches the official F0.5 examples, including the singleton cases | H2 |
| L4 | Normalizer | `src/normalize.py`, `tests/test_normalize.py` | `normalize_df()` returns contract columns; works for US, India, France-style strings | H4 |
| L5 | Blocking analysis | `reports/blocking_report.md` | Recall, avg candidates, F0.5 ceiling, top 3 miss causes + suggested fixes | H7 |
| L6 | Decision layer + tuning | `src/decide.py`, `tests/test_decide.py`, `data/decision_params.json` | Tuned params beat the plain 0.5 threshold on val | H9 |
| L7 | Methodology document | `Documentation_template.md` | All required sections filled in with our real numbers | H12 |

## Handoffs

| At | From → To | What |
|---|---|---|
| H1 | Lavanya → all | EDA findings, especially whether each S2/S3 record matches at most one S1 |
| H1 | Sajal → all | `io.py` merged; everyone uses it for loading |
| H2 | Lavanya → all | `splits.json` + `evaluate.py` merged; the only allowed way to score |
| H4 | Lavanya → Sajal | `normalize.py` merged |
| H6 | Sajal → Vidushi, Lavanya | `cands_val` / `cands_test` parquet |
| H8 | Vidushi → Lavanya | `preds_val.parquet` |
| H9 | Vidushi → Sajal | `preds_test.parquet` + saved model |
| H9 | Lavanya → Sajal | `decision_params.json` |

## Review pairing

| Author | Reviewer (first choice) | Backup |
|---|---|---|
| Sajal | Lavanya | Vidushi |
| Vidushi | Sajal | Lavanya |
| Lavanya | Sajal | Vidushi |
