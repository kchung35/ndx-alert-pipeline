# NDX 100 · Factor × Options × Insider Alert Pipeline

A standalone research tool that scans the NASDAQ 100 every day and flags
tickers where three independent edges — fundamental factors, options flow,
and SEC Form 4 insider transactions — agree on a directional view, plus
a long-only tactical tier based on 3-month momentum.

All data comes from free public sources (yfinance, SEC EDGAR, Ken French
data library). No paid feeds. No portfolio concepts. No cross-dependencies.

## Architecture

```
 universe.py ──► data_prices.py ──► factors.py ───────────────┐
        │                                                      │
        ├──► data_options.py ──► options_signals.py ───────────┤
        │                                                      ├─► alert_engine.py ─► alerts/{date}.parquet
        ├──► data_edgar.py   ──► insider_signals.py ───────────┤                         │
        │                                                      │                         ▼
        └──► data_ff.py      ──► factors.py (FF overlay) ──────┘                   dashboard.py (Streamlit)
                                                                                         │
                                                              backtest.py  ◄─── prices ──┘
                                                              risk.py (VaR / CVaR)
```

See [workflow-1.png](workflow-1.png) for the original design diagram.

## Quickstart

```bash
# 1. Python 3.13+ required (or 3.11 / 3.12 should work)
pip install -r requirements.txt

# 2. SEC requires a User-Agent with contact info for the EDGAR REST API
export SEC_USER_AGENT="Your Name your@email.com"

# 3. Run the full daily pipeline (takes ~1 hour on first run because
#    EDGAR's rate limit is 10 req/s and we pull all Form 4 filings).
python3 run_daily.py

# 4. Open the institutional dashboard
streamlit run src/dashboard.py
# -> http://localhost:8501
```

Subsequent daily runs are ~10 min since only the delta of new Form 4
filings needs to be fetched.

## What the alerts mean

The pipeline assigns each of the 100 tickers to one of six tiers:

| Tier | Rule | Recommended action |
|---|---|---|
| `STRONG_BULLISH`     | Composite > +1.5 AND all 3 components > +0.3 | Long 1.0× weight |
| `STRONG_BEARISH`     | Composite < −1.5 AND all 3 components < −0.3 | Short 1.0× weight |
| `MOMENTUM_LONG`      | Top 10% by 3-month momentum z-score | Long 0.75× weight |
| `CONFLUENCE_BULLISH` | Composite > +1.0, ≥2 of 3 components > +0.15, no strong disagreement | Long 0.5× weight |
| `CONFLUENCE_BEARISH` | Composite < −1.0, ≥2 of 3 components < −0.15, no strong disagreement | Short 0.5× weight |
| `NO_ALERT`           | Everything else | Ignore |

**Rebalance monthly. Hold 20 trading days.** See the "Strategy reliability"
section before committing real capital.

## Key CLI entry points

| Command | Purpose |
|---|---|
| `python3 run_daily.py`                           | Full daily pipeline |
| `python3 run_daily.py --date YYYY-MM-DD`         | Re-run for a specific date |
| `python3 run_daily.py --refresh-universe`        | Rebuild NDX constituent cache from Wikipedia |
| `python3 run_daily.py --skip-edgar`              | Skip the slow EDGAR step |
| `python3 -m src.universe`                        | Build universe file only |
| `python3 -m src.data_prices`                     | Pull prices + fundamentals + VIX |
| `python3 -m src.data_options`                    | Snapshot today's option chains |
| `python3 -m src.data_edgar`                      | Pull Form 4 filings |
| `python3 -m src.factors --date YYYY-MM-DD`       | Compute factor panel |
| `python3 -m src.alert_engine --date YYYY-MM-DD`  | Rebuild alerts from existing panels |
| `python3 -m src.backtest --start ... --end ...`  | Factor-only L/S backtest |
| `python3 -m src.risk --date YYYY-MM-DD`          | VaR/CVaR on current alert book |
| `streamlit run src/dashboard.py`                 | Launch the dashboard |

## Tests

```bash
# Unit tests (should be 68 passing)
python3 -m pytest tests/ -v

# Data-source sanity check (27/30 should PASS depending on yfinance state)
python3 test_data_availability.py

# 5-year multi-regime backtest validation of MOMENTUM_LONG
python3 tests/validate_momentum_5y.py

# Diagnostic harness — evidence for each of the flagged pipeline issues
python3 tests/diagnose_issues.py
```

## Project layout

```
ndx-alert-pipeline/
├── README.md                         # this file
├── requirements.txt
├── .gitignore
├── run_daily.py                      # daily orchestrator
├── test_data_availability.py         # standalone sanity check for the 3 data sources
├── check_yfinance_20d_oi.py          # deeper diagnostic on yfinance options schema
├── workflow-[1-3].png                # architecture diagrams
├── src/
│   ├── universe.py                   # NDX 100 constituent resolver (cached + Wikipedia fallback)
│   ├── data_prices.py                # yfinance OHLCV + fundamentals + VIX
│   ├── data_options.py               # yfinance option chain snapshotter
│   ├── data_edgar.py                 # SEC Form 4 fetcher + XML parser (raw REST, no edgartools)
│   ├── data_ff.py                    # Ken French 5-factor daily downloader
│   ├── factors.py                    # momentum (12-1, 3m), value, quality, low-vol, sector-neutralized
│   ├── options_signals.py            # V/OI, IV skew, term structure, flow ratio - the six metrics
│   ├── insider_signals.py            # cluster-weighted insider scoring
│   ├── alert_engine.py               # composite + momentum + confluence tier assignment
│   ├── backtest.py                   # walk-forward L/S backtest on factor composite
│   ├── risk.py                       # VaR / CVaR / Sharpe / drawdown
│   ├── dashboard.py                  # single-page Streamlit dashboard
│   ├── trading_day.py                # NY-timezone-aware last-completed-trading-day helper
│   └── lifted/                       # math + visual primitives (pure, no business logic)
│       ├── analytics.py              # VaR, Sharpe, correlation, max drawdown
│       ├── display.py                # column display name mapping
│       ├── index_universe.py         # CIK-based issuer dedup (GOOG/GOOGL etc.)
│       ├── insider_utils.py          # Form 4 classification + corporate-entity filter
│       ├── sec_identity.py           # SEC ticker -> CIK resolver
│       └── ui_style.py               # "Deep Trading" design tokens + Plotly theme
├── tests/
│   ├── test_*.py                     # 68 unit tests across the modules
│   ├── diagnose_issues.py            # investigate-before-fix diagnostic harness
│   └── validate_momentum_5y.py       # multi-regime backtest validation
└── data/
    ├── constituents_ndx100.json      # NDX 100 constituents seed (ticker, company, sector)
    ├── market_caps_ndx100.json       # market cap seed (billions)
    ├── sec_company_tickers_cache.json # SEC CIK lookup cache
    └── {generated on first run}/
        ├── prices.parquet
        ├── fundamentals.parquet
        ├── vix.parquet
        ├── ff.parquet
        ├── universe.parquet
        ├── chains/{date}/{ticker}.parquet
        ├── form4/{ticker}.parquet
        ├── factors/{date}.parquet
        ├── options_signals/{date}.parquet
        ├── insider_signals/{date}.parquet
        └── alerts/{date}.parquet
```

## Strategy reliability — honest caveats

This is a **research-grade prototype**, not an institutional product.

- **Backtest sample**: 5 years, 60 rebalances, ~600 position-months. Academic
  momentum studies use 30+ years and 10,000+ stocks.
- **Selection bias**: strategies were chosen after looking at their own
  backtests. Not true out-of-sample.
- **Regime risk**: momentum-long **amplified the 2022 tech bear market
  drawdown** (-34% vs benchmark -25%). This is the documented "momentum
  crash" phenomenon and you will experience it again.
- **No transaction costs modeled**: realistic friction (~7 %/yr) is not
  baked into the backtest. Real returns will be lower.
- **Survivorship bias**: backtest uses *today's* NDX 100 constituents on
  historical dates. Stocks that were booted from the index are absent.
- **The 3-signal composite itself has never been backtested** — only the
  factor-only and momentum-only legs were. Treat composite tiers as
  exploratory.

**Recommended before any real capital**:
1. Paper-trade with actual entry/exit fills for 3–6 months.
2. Add a regime filter: don't take MOMENTUM_LONG when NDX is below its
   200-day SMA. This alone tames the 2022 fat tail.
3. Add transaction cost model (3 bps/trade round-trip).
4. Run full 10-year backtest.
5. Back-test the 3-signal composite across regimes.

## Design choices & non-goals

- **Parquet-only output layer.** No DB. Everything is a file on disk.
- **SEC raw REST, no edgartools.** The `index.json` + `primary_doc.xml`
  / `edgardoc.xml` probe handles 99% of Form 4 naming conventions.
- **In-process token-bucket rate limiter.** No `/tmp` state files.
- **No portfolio concepts.** The "live book" is the set of currently-open
  alerts, nothing more — no NAV, no holdings, no transactions.
- **Sector-neutralized factor z-scores** with a min-sector-size guard of 5
  (so solo-sector tickers like NDX's lone Real Estate / Basic Materials
  members don't get zeroed out).
- **Institutional UI**: Fraunces serif masthead, IBM Plex Sans body,
  JetBrains Mono for numerics, aged-gold `#E8B84E` accent on a blue-black
  canvas. Single page, three sections, hand-rolled HTML ledger tables.

## License

Personal research use. No warranty. Not investment advice.
