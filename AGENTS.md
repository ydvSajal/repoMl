# AGENTS.md — Master Instructions (read this first)

This file is for **every human and every AI coding agent** (Claude Code, Copilot, Cursor, etc.) working in this repo. If any other doc conflicts with this file, this file wins.

## 1. Project in one sentence

We are building a Python pipeline for the **Amazon ML Challenge: Business Entity Resolution**. For every Source 1 business record it finds the matching Source 2 and Source 3 records, and it outputs `matching_results.tsv` and `candidate_pairs.tsv`, scored by macro F0.5.

## 2. Team and ownership

| Person | Role | Agent instruction file | Owns (only they edit) |
|---|---|---|---|
| Sajal | Backend & coding, repo owner, leaderboard uploader | `docs/agents/sajal.md` | `src/io.py`, `src/blocking.py`, `src/run.py`, `src/package.py`, `src/config.py`, `requirements.txt`, `README.md`, `.github/`, `tests/test_io.py`, `tests/test_blocking.py` |
| Vidushi | AI/ML, model owner | `docs/agents/vidushi.md` | `src/features.py`, `src/train.py`, `tests/test_features.py`, `reports/model_report.md` |
| Lavanya | Data science, data and evaluation owner | `docs/agents/lavanya.md` | `src/normalize.py`, `src/split.py`, `src/evaluate.py`, `src/decide.py`, `notebooks/`, `reports/eda_findings.md`, `reports/blocking_report.md`, `Documentation_template.md`, `tests/test_normalize.py`, `tests/test_evaluate.py`, `tests/test_decide.py` |

**Shared files:**
- `docs/` changes need Sajal's approval.
- `src/config.py` is owned by Sajal, but each person may **add** keys in their own section; never edit someone else's keys.

## 3. The PR-only rule (non-negotiable)

**Nobody commits to `main`. Every change reaches `main` through a Pull Request.** This applies to humans and AI agents equally. The only exception is Sajal's single bootstrap commit in step S0, which is made before branch protection is switched on.

```bash
# 1. Start from fresh main
git checkout main
git pull --rebase origin main

# 2. Create a branch named <owner>/<task-id>-<short-slug>
git checkout -b vidushi/V3-group-features

# 3. Work, then commit (small commits, conventional messages)
git add src/features.py tests/test_features.py
git commit -m "feat(features): add rank, gap and competition group features (V3)"

# 4. Push and open a PR
git push -u origin vidushi/V3-group-features
gh pr create --base main --fill

# 5. After approval: squash and merge, then delete the branch
gh pr merge --squash --delete-branch
```

### Rules

1. **Branch names:** `sajal/S2-blocking`, `vidushi/V1-base-features`, `lavanya/L3-normalizer`. The task IDs come from `docs/02-work-division.md`.
2. **Commit messages:** `type(scope): what changed (TASK-ID)`, where the type is one of `feat`, `fix`, `test`, `docs`, `refactor`, `chore`.
3. **One task per PR.** Keep PRs small, under about 300 changed lines. Big PRs slow the whole team down.
4. **Every PR needs 1 approval from a teammate** other than the author before merging, and Sajal is the default reviewer. Reviews must happen within **10 minutes** of being requested, because this is a one-day build.
5. **Tests pass before a PR is opened:** `pytest -q` must be green, or the PR must explain why a test is skipped.
6. **Merge with "Squash and merge" only**, then delete the branch.
7. **Pull `main` before every new branch.** If your PR has conflicts, rebase it: `git pull --rebase origin main`, fix, then `git push --force-with-lease`.
8. **Never force-push to `main`.** Never disable branch protection. Sajal has admin rights and must not use the bypass.
9. **Use the PR template** in `.github/pull_request_template.md`, and fill in every section.

### Rules for AI agents specifically

- **Before writing code,** read `AGENTS.md`, then your person's file in `docs/agents/`, then `docs/04-TRD.md`.
- **Only edit files owned by the person you are working for** (see §2). If a change is needed in someone else's file, stop and tell your human to ask the owner.
- **Follow the data contracts in `docs/04-TRD.md` §3 exactly:** file names, column names, dtypes, and function signatures. Do not rename columns or change signatures.
- **Always create a branch, commit, push and open a PR.** Never merge your own PR. Never push to `main`.
- Do not commit anything in `dataset/`, `data/`, `models/`, or large files (> 5 MB). These are gitignored. The only exceptions are `data/splits.json` and `data/decision_params.json`, which are small and needed for reproducibility.
- Do not add dependencies without adding them to `requirements.txt` in a PR that Sajal reviews.
- Do not call any external API, geocoder or web database to look up businesses. That leads to **disqualification** under the competition rules.
- Keep randomness reproducible by using `SEED` from `src/config.py` everywhere.
- **When unsure, ask your human.** Do not guess contracts.

## 4. Competition rules the code must respect

- **Read every TSV** with `pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)`, through `src/io.py` only.
- **Treat `country` as an open set of strings.** The test set contains France, which is not in train. Never hard-code, filter or one-hot `{US, India}`.
- **Output rules:** exactly one row per test Source 1 entity. Empty lists are allowed. No duplicate IDs, and only `S2-`/`S3-` IDs that exist in the test set.
- **Candidates and matches:** `candidate_pairs.tsv` must be the exact set the model scored, and every final match must appear in it.
- **Licensing and data:** the final model must be MIT or Apache 2.0 licensed and ≤ 8B parameters. No external data or lookups.

## 5. Repo layout

```
amazon-er/
├── AGENTS.md                 # this file (master rules)
├── CLAUDE.md                 # points Claude Code to AGENTS.md
├── README.md                 # how to run end-to-end
├── requirements.txt
├── .gitignore                # dataset/, data/* (except splits.json, decision_params.json), models/, *.parquet, .venv/
├── .github/pull_request_template.md
├── docs/                     # PRD, TRD, timeline, division, architecture, per-person agent files
├── dataset/                  # provided data (gitignored)
│   ├── train/{train_source1,2,3,train_ground_truth}.tsv
│   └── test/{test_source1,2,3}.tsv
├── utils/validate_submission.py   # provided validator
├── src/
│   ├── config.py   io.py   normalize.py   split.py   evaluate.py
│   ├── blocking.py features.py train.py   decide.py  run.py  package.py
├── tests/
├── notebooks/eda.ipynb
├── reports/                  # eda_findings.md, blocking_report.md, model_report.md, leaderboard_log.md
├── data/                     # intermediates: norm_*, cands_*, preds_* (gitignored)
├── models/                   # trained models (gitignored)
└── output/                   # matching_results.tsv, candidate_pairs.tsv
```

## 6. How to run

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m src.run --split val      # full pipeline on the validation split, prints F0.5
python -m src.run --split test     # writes output/*.tsv
python3 utils/validate_submission.py --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv --test-dir dataset/test
pytest -q
```

## 7. Communication

- **Blocked for 20 minutes?** Post in the team chat: what you tried, the exact error, and your branch name.
- **Sync checkpoints:** the hour marks in `docs/01-timeline.md`.
- **Leaderboard uploads:** only Sajal uploads, and he logs each upload in `reports/leaderboard_log.md` (time, git commit, val F0.5, leaderboard F0.5).
