# 03 — Product Requirements Document (PRD)

**Project:** Business Entity Resolution pipeline for the Amazon ML Challenge
**Owner:** Sajal (lead), Vidushi (model), Lavanya (data and evaluation)
**Timebox:** one day (H0 → H12)

## 1. Problem

Business records arrive from 3 independent sources, and the names and addresses are noisy: abbreviations, typos, legal suffixes, transliterations, missing PIN codes, and landmark-based addresses. There is no shared ID. For every Source 1 entity (the clean, deduplicated reference list), we must find all matching Source 2 and Source 3 records, which may be zero, one or many.

## 2. How we are scored

- **Metric:** F0.5 is computed **per Source 1 entity** and then averaged (macro). Precision counts twice as much as recall.
- **Singletons count.** An entity with no true matches scores **1.0 if we predict an empty list and 0.0 if we predict anything**.
- **Leaderboards:** the public leaderboard uses a subset of the test set, and the final ranking uses the private leaderboard, which is the rest.
- **The test set includes France,** which does not appear in the training data.

## 3. Goals

1. **G1 (safety):** a valid submission is on the leaderboard by H2.
2. **G2 (recall ceiling):** blocking pair recall is **≥ 95%** on val, with a reasonable number of candidates per entity (target ≤ 40).
3. **G3 (quality):** the final val macro F0.5 clearly beats our baseline, and the leaderboard score tracks val within about 0.03.
4. **G4 (generalization):** performance does not collapse on an unseen country, checked by training on US and validating on India.
5. **G5 (compliance):** the submission zip reproduces our outputs from a clean clone, uses only allowed models and data, and passes the validator.

## 4. Non-goals

| Non-goal | Why |
|---|---|
| LLMs or cross-encoders | Not needed in one day; adds licence, compute and reproducibility risk |
| External data, geocoding, business lookups | Forbidden by the rules, which means disqualification |
| Clustering beyond the one-to-one assignment | Low gain for the time available |
| Per-country hand-written rules | Would break on France and any other unseen country |
| A UI or web service | The output is two TSV files and a zip |

## 5. Users of this pipeline

| User | Needs |
|---|---|
| Leaderboard (the judge) | Correct TSV format, one row per S1, precision-heavy matches |
| Challenge reviewers | Runnable code, a clear README, a methodology document, and a visible blocking stage (`candidate_pairs.tsv`) |
| Our team | One command to run everything, one scorer everyone trusts, and file ownership that avoids merge conflicts |

## 6. Requirements

### P0 (must have)

| ID | Requirement | Acceptance criteria |
|---|---|---|
| P0-1 | Robust TSV IO | All reads use `sep="\t", dtype=str, keep_default_na=False`; empty lists survive round trips |
| P0-2 | Country-agnostic normalization | Legal suffixes (US/IN/FR), address abbreviations, accents, `&`/`and`, postcode and house-number extraction |
| P0-3 | Multi-blocker candidate generation | Name TF-IDF top-k ∪ address TF-IDF top-k ∪ postcode+rare-token, searched within the same country |
| P0-4 | Pairwise + group features | Features listed in TRD §5.3 and §5.4 |
| P0-5 | LightGBM matcher | Probability per candidate pair; seed fixed |
| P0-6 | Decision layer | One-to-one assignment (if EDA confirms it), tuned threshold, empty list when the top probability is weak |
| P0-7 | Official-equivalent scorer | Macro F0.5 including the singleton rules; used for every decision |
| P0-8 | Outputs + validator | `matching_results.tsv` ⊆ `candidate_pairs.tsv`; validator prints PASS |
| P0-9 | Reproducible package | `python -m src.package` builds the required zip structure |

### P1 (nice to have)

- Multilingual embedding cosine feature, using an MIT or Apache 2.0 licensed small model.
- Per-country threshold, used only if it helps val **and** falls back to the global threshold for unseen countries.
- S2↔S3 agreement feature: two candidates that strongly match each other both matching the same S1.

## 7. Success metrics

| Metric | Target | Measured by |
|---|---|---|
| Blocking pair recall (val) | ≥ 95% | `evaluate.blocking_recall` |
| F0.5 ceiling after blocking (val) | ≥ 0.95 | `evaluate.blocking_recall` |
| Val macro F0.5 | Beat baseline by a clear margin | `evaluate.f05_macro` |
| Singleton accuracy (val) | Reported | `evaluate.f05_macro` breakdown |
| Val vs leaderboard gap | ≤ 0.03 | `reports/leaderboard_log.md` |

## 8. Open questions (answer in the first 2 hours)

| Question | Owner | Why it matters |
|---|---|---|
| Can one S2/S3 record belong to more than one S1 in the ground truth? | Lavanya (L1) | Decides whether the one-to-one assignment is safe |
| What % of S1 entities are singletons in train? | Lavanya (L1) | Sets how conservative the threshold should be |
| How many rows are in each source? | Sajal (S1) | Decides whether brute-force TF-IDF fits in RAM or needs chunking |
| Are S2 and S3 noise patterns different? | Lavanya (L1) | Could justify a source indicator feature |
