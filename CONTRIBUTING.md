# Getting started (for teammates)

You've been invited as a collaborator on this repo. Here's everything you
need to be productive in about 5 minutes.

## 1. Accept the invite

Check your email for a GitHub invitation from Kevin. Click accept, or go
to https://github.com/kchung35/ndx-alert-pipeline and accept the banner.

## 2. Clone + install (one time)

```bash
git clone https://github.com/kchung35/ndx-alert-pipeline.git
cd ndx-alert-pipeline

# Isolated Python env (Python 3.11 / 3.12 / 3.13 all work)
python3 -m venv .venv && source .venv/bin/activate      # macOS / Linux
# Windows PowerShell:
#   py -m venv .venv ; .venv\Scripts\Activate.ps1

pip install -r requirements.txt

# Tell git who you are (for commits, one-time per machine)
git config user.name "Your Name"
git config user.email "your-github-username@users.noreply.github.com"
```

## 3. See the dashboard immediately (no data fetch needed)

```bash
streamlit run src/dashboard.py
# -> open http://localhost:8501 in your browser
```

The repo ships with a committed data snapshot for **2026-04-21** so this
works instantly. No SEC key, no yfinance pulls, no waiting.

Also, you can double-click `NDX Alert Desk.html` to see the same dashboard
as a static HTML preview — no Python needed at all.

## 4. Make a change

```bash
# Always pull first so you have the latest main
git checkout main
git pull

# Branch off main for your work
git checkout -b feature/my-improvement
# naming convention: feature/* fix/* docs/* chore/*

# ... edit files in VS Code / PyCharm / whatever ...

# Run the test suite before committing — 74 tests, should all pass
python3 -m pytest tests/

# Commit small, descriptive chunks
git add src/some_file.py
git commit -m "Short imperative description of what changed"

# Push your branch to GitHub
git push -u origin feature/my-improvement

# Open a pull request for review (command-line)
gh pr create --fill
# OR use the web UI — GitHub shows a "Compare & pull request" button
```

After your PR is reviewed and approved, merge it on GitHub (Squash and
merge is cleanest), then:

```bash
git checkout main
git pull
git branch -D feature/my-improvement   # delete the local branch
```

## 5. The rules (short)

- **Never commit files under `data/`.** If you run `python3 run_daily.py`
  locally to test, that's fine — just don't `git add data/`. Data
  refreshes are Kevin's job; he'll push a new snapshot weekly.
  If you accidentally staged data, unstage with `git restore --staged data/`.
- **One change per PR.** Easier to review and revert.
- **Descriptive commit messages.** *"Fix bug"* is bad.
  *"Raise IV floor from 5% to 7% to cut stale quotes on thin chains"* is good.
- **Ask before refactoring.** Check with Kevin on Slack / in an issue
  before doing anything bigger than a 20-line change.

## 6. Where to find things

| You want to... | Go to |
|---|---|
| See the project architecture | [README.md](README.md) |
| Understand the alert tiers | README section "What the alerts mean" |
| See the strategy reliability caveats | README section "Scope & methodological notes" |
| Find a specific signal | `src/factors.py` · `src/options_signals.py` · `src/insider_signals.py` |
| See how alerts are combined | `src/alert_engine.py` |
| Read or write tests | `tests/` |
| See the whole pipeline end-to-end | `run_daily.py` |
| Regenerate the HTML dashboard | `python3 scripts/generate_dashboard_html.py` |

## 7. Pick up a task

Open [Issues](https://github.com/kchung35/ndx-alert-pipeline/issues)
and assign yourself something. If the list is empty, suggest one by
opening a new issue.

Planned extensions (from README §9):

1. 200-day SMA regime gate on MOMENTUM_LONG
2. Transaction cost model (3 bps round-trip)
3. 10-year backtest covering 2018–2020 cycle
4. Point-in-time NDX membership from Nasdaq index archives
5. Cross-validated composite weights and tier thresholds

## 8. Need help?

Open an issue or ping Kevin. Don't silently force-push to main — ask first.

Happy hacking.
