# 07 — Team Playbook

## 1. First 15 minutes (everyone)

```bash
git clone https://github.com/<owner>/amazon-er.git
cd amazon-er
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# copy the provided dataset/ and utils/ folders into the repo root (they are gitignored)
pytest -q
gh auth status                       # must say "Logged in"
```

Then read `AGENTS.md` and your own file in `docs/agents/`.

## 2. The PR loop (repeat for every task)

```bash
git checkout main && git pull --rebase origin main
git checkout -b <yourname>/<TASK-ID>-<slug>
# ...work...
pytest -q
git add <only your files>
git commit -m "feat(scope): what changed (TASK-ID)"
git push -u origin HEAD
gh pr create --base main --fill      # then edit the description using the template
# post the PR link in chat and tag your reviewer
# after approval:
gh pr merge --squash --delete-branch
git checkout main && git pull --rebase origin main
```

### Reviewing a PR (10-minute target)

1. Check that the author only edited files they own (`AGENTS.md` §2).
2. Check that the column names and signatures match `04-TRD.md` §3.
3. Check that tests exist and pass. Run `gh pr checkout <number> && pytest -q` if unsure.
4. Look for hard-coded countries, NaN handling, and data files added by mistake.
5. Approve with `gh pr review <number> --approve`, or request changes with a clear comment.

### Merge conflicts

Conflicts should be rare, because each person owns separate files. If one happens:
```bash
git pull --rebase origin main
# fix the conflicted files, then:
git add <files> && git rebase --continue
git push --force-with-lease
```

## 3. Working with your AI agent

- **Starting a session:** use the starter prompt in `docs/README.md`.
- **One step at a time:** ask the agent to do exactly one step from your agent file, then verify and PR before moving on.
- **Before every commit:** ask the agent to list the files it changed. If any are not yours, revert them.
- **Understand what you ship:** you must be able to explain your code if the challenge reviewers ask. Ask the agent "explain this function in 3 lines" for anything unclear.
- **Know when to stop the agent.** Stop it when it wants to:
  - change someone else's file,
  - rename contract columns,
  - add a new library,
  - call an external API,
  - or push to `main`.

## 4. Communication

| When | What |
|---|---|
| Starting a task | "Starting V2 on branch vidushi/V2-features" |
| Opening a PR | Link + reviewer tag |
| Stuck for 20 min | Tried X and Y, the exact error, branch name |
| Every checkpoint (H1, H2, H4, H6, H8, H10) | 1-line status + any numbers (recall, val F0.5) |

## 5. Definition of done (every task)

- [ ] Acceptance criteria in `02-work-division.md` met
- [ ] Tests added or updated, and `pytest -q` is green
- [ ] Only owned files changed
- [ ] PR approved and squash-merged
- [ ] Numbers (if any) posted in chat

## 6. Glossary

| Term | Meaning |
|---|---|
| S1 / S2 / S3 | Source 1 (clean reference), Source 2, Source 3 |
| Singleton | An S1 entity with no true matches. Predicting empty scores 1.0 |
| Blocking | Cheaply picking a short list of candidates per S1, so the model does not compare everything with everything |
| Candidate | An S2/S3 record the blocking stage thinks might match |
| Pair recall | The share of true matches that survived blocking |
| F0.5 ceiling | The best score possible if the model were perfect on our candidates |
| Group features | Features that compare a candidate with its competitors (rank, gap) |
| One-to-one assignment | Each S2/S3 record is given to at most one S1, whichever has the highest probability |
| Macro F0.5 | F0.5 per S1, averaged. Precision counts twice as much as recall |
