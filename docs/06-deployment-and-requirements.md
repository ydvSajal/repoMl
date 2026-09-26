# 06 — Where It Runs and What We Need

## 1. Short answer: do we need servers?

**No.** This is an offline ML pipeline. It runs on our laptops and produces two TSV files and a zip. There is no hosting, no API and no database.

| Need | Where | Cost |
|---|---|---|
| Running the pipeline | Each person's laptop | Free |
| Code + PRs | GitHub **private** repo (make it public only if the rules require it at submission) | Free |
| Optional extra compute (P1 embeddings) | Google Colab or Kaggle notebooks (free tiers) | Free |
| Submission | Challenge portal (leaderboard upload) + final zip | Free |

## 2. Hardware

| Item | Minimum | Recommended |
|---|---|---|
| RAM | 8 GB | 16 GB |
| Disk | 2 GB free | 5 GB free |
| GPU | Not needed | Only for the optional embedding feature |

**Check the dataset size at H0.** If the sources have hundreds of thousands of rows, blocking must use chunked sparse matmul (TRD §5.2), which is already the plan.

## 3. Software requirements (everyone)

| Tool | Version | Check |
|---|---|---|
| Python | 3.10–3.12 | `python --version` |
| Git | Recent | `git --version` |
| GitHub CLI | Recent, logged in | `gh auth status` |
| VS Code (or any editor) + your AI agent | Latest | – |

### Python packages (`requirements.txt`)

```
pandas
numpy
pyarrow
scipy
scikit-learn
rapidfuzz
lightgbm
pytest
# optional (P1 only, add in a separate PR):
# sentence-transformers
```

Sajal installs these at H0 and **pins exact versions**:
```bash
pip install -r requirements.txt
pip freeze | grep -iE "^(pandas|numpy|pyarrow|scipy|scikit-learn|rapidfuzz|lightgbm|pytest)==" > requirements.txt
```

Nobody upgrades packages during the day.

### Windows notes

- Activate the venv with `.venv\Scripts\activate`.
- Run `git config core.autocrlf input` so TSV line endings stay consistent.
- If `lightgbm` fails to install, run `pip install --upgrade pip wheel` first, and then install lightgbm again.

## 4. Data handling

- Put the provided files in `dataset/train/` and `dataset/test/`, and put the validator in `utils/validate_submission.py`.
- `dataset/`, `data/`, `models/` and `*.parquet` are **gitignored**. Never commit competition data.
- Share the dataset zip once, in the team chat or on a shared drive. Everyone uses the same copy.

## 5. Licence and rule compliance checklist

- [ ] The final model is LightGBM (MIT). An optional embedding model must be MIT or Apache 2.0, ≤ 8B parameters (small models are ~100M), and its licence must be written in the README.
- [ ] No external lookups: no geocoding, no business registries, no web data.
- [ ] Everything is trained only on the provided training data.
- [ ] Seeds are fixed (`SEED = 42`) so the outputs are reproducible.

## 6. Submission package

`python -m src.package --team <team_name>` builds this structure:

```
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/              # copy of our src/
│       ├── README.md         # exact reproduce steps
│       └── requirements.txt  # pinned
└── Documentation_template.md
```

### Final checks (Sajal, S8)

- [ ] `python3 utils/validate_submission.py ...` prints **PASS**.
- [ ] A fresh `git clone`, new venv, `pip install -r requirements.txt` and `python -m src.run --split test` reproduce `output/matching_results.tsv` exactly (`diff` shows nothing).
- [ ] The zip contains no dataset files and no `data/` folder.
- [ ] `Documentation_template.md` is filled in (Lavanya).
- [ ] The leaderboard upload matches the zip's `matching_results.tsv`.
