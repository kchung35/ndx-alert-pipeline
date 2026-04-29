# Committed Snapshot Data

This directory intentionally ships with a small public baseline snapshot for
`2026-04-21` so the dashboard is reproducible immediately after clone.

The included files are public/free-source research data:

- `form4/*.parquet`: SEC Form 4 cache. This avoids the slow first insider pull.
- `chains/2026-04-21/*.parquet`: yfinance options-chain snapshot.
- `alerts/`, `factors/`, `options_signals/`, `insider_signals/`: computed panels.
- `prices.parquet`, `fundamentals.parquet`, `vix.parquet`, `ff.parquet`,
  `universe.parquet`: baseline market/universe inputs.
- `snapshot_manifest.json`: expected counts for verification.

Run this from the repo root to verify the committed baseline:

```bash
python3 scripts/verify_snapshot.py --date 2026-04-21
```

Future EDGAR runs are incremental by default: cached Form 4 accessions are
skipped before XML fetch, and only new filings are merged into `form4/`.

Local newsletter exports are written under `exports/` and are intentionally
ignored by git.
