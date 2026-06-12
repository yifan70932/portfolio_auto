# Bridge Package — Portfolio Ops System (Tier 1)

Generated 2026-06-11 by Claude from live Robinhood MCP data.

## What this is
The state layer connecting three things:
- **Robinhood** (single source of truth for positions/watchlists, via MCP)
- **Claude Routine** (Tue/Fri automated read-only audit, cloud)
- **Local quant tool** (`quant_prototype.py`, Tier 2 research engine — reads these same JSONs instead of watchlist.txt)

## Files
| file | role | who writes it |
|---|---|---|
| `watchlists.json` | theme map, 7 lists | exported by Claude when lists change |
| `portfolio.json` | positions snapshot | refreshed by the routine each run |
| `contracts.json` | five-item contracts + doctrine + watch queue | edited in chat with Claude when opening/closing positions |
| `journal.csv` | trade journal (alpha ledger input) | appended in chat with Claude after each trade |
| `routine_prompt.md` | paste into claude.ai/code/routines | static |

## Setup (one-time, ~10 min)
1. Create a **private** GitHub repo (e.g. `yifan-portfolio-ops`), put this folder in it as `bridge/`, create an empty `reports/` directory.
2. Go to claude.ai/code/routines → New routine → paste `routine_prompt.md` as the prompt → attach the repo → connectors: keep **Robinhood only** (remove everything else) → trigger: Schedule, Tue & Fri 4:30 PM ET.
3. First run: trigger once manually, then compare its report against a Tier-1 audit done in chat. Promote to unattended only after they agree.

## Standing doctrine (mirrors contracts.json)
- Routine is read-only; all trading happens in interactive chat with per-order confirmation.
- Every non-ETF position carries a five-item contract before entry.
- Two known open items as of export: SNDK-MAIN (a)/(b) decision and SMCI-MAIN contract adoption — both due 2026-06-12 open; the first routine run will flag them RED/YELLOW until resolved.

## Tier 2 hook (next build)
`quant_prototype.py` upgrade will read `watchlists.json` + `portfolio.json` directly:
portfolio beta vs SPY, FF5 exposures of actual holdings, holdings correlation matrix,
marginal risk contribution, and an Alpha Ledger (per-closed-trade r_trade − β·r_SPY) joined with process grades from `journal.csv`.
