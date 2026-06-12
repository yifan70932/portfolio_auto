#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quant Analysis Tool
=========================================
Pulls watchlist data, computes QVM+L factor ranking and options metrics,
and exports a self-contained bilingual (EN/ZH) HTML report.

Uses yfinance to pull watchlist data, computes QVM+L factor ranking
and options metrics, exports a self-contained bilingual (EN/ZH) HTML report.

Requirements:
    pip install yfinance pandas numpy scipy

Run:
    1. Put your tickers in watchlist.txt (one per line)
    2. python3 quant_prototype.py
    3. Open the generated quant_report.html

Notes:
  - Must run in an environment with access to Yahoo Finance (your own machine is fine).
  - IV / Greeks are solved here via Black-Scholes, not relying on source garbage values.
  - The source provides only a CURRENT options snapshot, no history; IV Rank needs historical IV,
    so this script uses the percentile of trailing-1y realized volatility as a PROXY for IV Rank,
    clearly labeled as such in the report (use OptionMetrics etc. for serious research).
"""

import math
import json
import datetime as dt
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq

import yfinance as yf

# =============================================================
# CONFIG
# =============================================================
# watchlist is read from watchlist.txt in the same folder (one ticker per line, # = comment).
# Edit it via the launcher window, or via the report's "Edit List" button -> save -> re-run.
WATCHLIST_FILE = "watchlist.txt"
DEFAULT_WATCHLIST = ["MU", "NVDA", "AMD", "AVGO", "TSM", "INTC"]
RISK_FREE_RATE = 0.045   # approx risk-free rate (for BS pricing); adjust to current T-bill yield
OUTPUT_HTML = "quant_report.html"

# --- Options data source ---
# By default this tool uses the (hardened) yfinance options source: it filters out junk
# contracts and solves IV via Black-Scholes, avoiding the garbage values the raw scrape produces.
# OPTIONAL: Tradier provides cleaner exchange-grade data, but requires opening an account.
# If you ever get a Tradier token, set the env var TRADIER_TOKEN (or paste below) and it will
# be used automatically; otherwise the tool silently uses yfinance. No account needed by default.
import os as _os
TRADIER_TOKEN = _os.environ.get("TRADIER_TOKEN", "")  # optional; leave empty to use yfinance
TRADIER_BASE = "https://api.tradier.com/v1"  # sandbox alternative: "https://sandbox.tradier.com/v1"

# --- Macro data source (FRED, the St. Louis Fed database) ---
# Free, official. A free API key is recommended (no rate limit) but NOT required:
# without a key the tool falls back to FRED's public CSV download endpoint.
# Get a free key at https://fredaccount.stlouisfed.org/apikeys
FRED_API_KEY = _os.environ.get("FRED_API_KEY", "")  # optional; leave empty to use the CSV fallback

# QVM+L factor weights for composite score (adjustable)
WEIGHTS = {"value": 0.25, "quality": 0.25, "momentum": 0.25, "lowvol": 0.25}


def load_watchlist():
    """Read tickers from watchlist.txt; if absent, use defaults and create one."""
    import os
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), WATCHLIST_FILE)
    if os.path.exists(path):
        syms = []
        with open(path, encoding="utf-8") as fp:
            for line in fp:
                s = line.split("#")[0].strip().upper()
                if s:
                    syms.append(s)
        if syms:
            return syms
    with open(path, "w", encoding="utf-8") as fp:
        fp.write("# one ticker per line. lines starting with # are comments.\n")
        fp.write("\n".join(DEFAULT_WATCHLIST) + "\n")
    print(f"created {WATCHLIST_FILE}")
    return list(DEFAULT_WATCHLIST)


# =============================================================
# Black-Scholes pricing & IV solver
# =============================================================
def bs_price(S, K, T, r, sigma, opt="put"):
    """Black-Scholes theoretical price. T in years."""
    if T <= 0 or sigma <= 0:
        # expired or zero vol: return intrinsic value
        return max(0.0, (K - S) if opt == "put" else (S - K))
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if opt == "call":
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    else:
        return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_delta(S, K, T, r, sigma, opt="put"):
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    return norm.cdf(d1) if opt == "call" else norm.cdf(d1) - 1.0


def implied_vol(price, S, K, T, r, opt="put"):
    """Solve IV from market price. Returns None on failure to avoid garbage values."""
    intrinsic = max(0.0, (K - S) if opt == "put" else (S - K))
    if price <= intrinsic + 1e-6 or T <= 0:
        return None
    try:
        f = lambda sig: bs_price(S, K, T, r, sig, opt) - price
        return brentq(f, 1e-4, 8.0, maxiter=200)
    except Exception:
        return None


# =============================================================
# data fetch
# =============================================================
def fetch_fundamentals(tk):
    info = tk.info
    quote_type = (info.get("quoteType") or "").upper()  # EQUITY / ETF / INDEX / etc.
    is_etf = quote_type in ("ETF", "MUTUALFUND")
    if is_etf:
        sector = "ETF"
    else:
        sector = info.get("sector") or info.get("sectorDisp") or "N/A"
    return {
        "name": info.get("shortName") or info.get("longName") or tk.ticker,
        "price": info.get("currentPrice") or info.get("regularMarketPrice"),
        "pe": info.get("trailingPE"),
        "pb": info.get("priceToBook"),
        "roe": info.get("returnOnEquity"),
        "de": info.get("debtToEquity"),
        "mktcap": info.get("marketCap"),
        "ev_ebitda": info.get("enterpriseToEbitda"),
        "profit_margin": info.get("profitMargins"),
        "sector": sector,
        "is_etf": is_etf,
    }


def fetch_price_history(tk, period="3y"):
    """Pull 3 years by default for backtesting; factor functions slice the recent 1y internally."""
    h = tk.history(period=period)
    return h["Close"].dropna()


def compute_momentum(close):
    """12-1 momentum: trailing 12-month return excluding the most recent month."""
    if len(close) < 252:
        if len(close) < 30:
            return None
        return close.iloc[-1] / close.iloc[0] - 1
    p_12m = close.iloc[-252]
    p_1m = close.iloc[-21]
    return p_1m / p_12m - 1


def compute_realized_vol(close, window=252):
    rets = np.log(close / close.shift(1)).dropna()
    if len(rets) < 20:
        return None
    return rets.tail(window).std() * math.sqrt(252)


def compute_multi_momentum(close):
    """Multi-horizon momentum: 1m/3m/6m/12m cumulative returns."""
    def ret(days):
        if len(close) <= days:
            return None
        return float(close.iloc[-1] / close.iloc[-days] - 1)
    return {"1m": ret(21), "3m": ret(63), "6m": ret(126), "12m": ret(252)}


def compute_drawdown(close):
    """Current drawdown (from peak) and trailing-1y max drawdown."""
    if len(close) < 20:
        return {"current_dd": None, "max_dd": None, "from_52w_high": None}
    arr = close.values
    running_max = np.maximum.accumulate(arr)
    dd = arr / running_max - 1.0
    cur_dd = float(dd[-1])
    max_dd = float(dd.min())
    hi_52w = float(close.tail(252).max())
    from_high = float(close.iloc[-1] / hi_52w - 1) if hi_52w > 0 else None
    return {"current_dd": cur_dd, "max_dd": max_dd, "from_52w_high": from_high}


def downsample_prices(close, n=120):
    """Downsample the price series to ~n points for SVG plotting.
    Normalized to day1=100 for cross-stock comparison."""
    if len(close) < 5:
        return None
    s = close.dropna()
    step = max(1, len(s) // n)
    sampled = s.iloc[::step]
    base = sampled.iloc[0]
    if base <= 0:
        return None
    norm_prices = [round(float(v / base * 100), 2) for v in sampled.values]
    return norm_prices


def iv_rank_proxy(close):
    """
    IV Rank PROXY: percentile of trailing-1y 21-day rolling realized volatility.
    True IV Rank needs historical implied vol (paid data). This is explicitly a proxy.
    """
    rets = np.log(close / close.shift(1)).dropna()
    if len(rets) < 60:
        return None
    rolling = rets.rolling(21).std() * math.sqrt(252)
    rolling = rolling.dropna()
    if len(rolling) < 20:
        return None
    cur = rolling.iloc[-1]
    lo, hi = rolling.min(), rolling.max()
    if hi - lo < 1e-9:
        return None
    return float((cur - lo) / (hi - lo) * 100)


# =============================================================
# Options data source - pluggable adapters (Tradier preferred, yfinance fallback)
# =============================================================
def _pick_expiry(expiries):
    """From a list of 'YYYY-MM-DD' expiries, prefer one 2-12 days out; else the nearest."""
    if not expiries:
        return None
    best = expiries[0]
    for e in expiries:
        try:
            d = (dt.datetime.strptime(e, "%Y-%m-%d").date() - dt.date.today()).days
        except ValueError:
            continue
        if 2 <= d <= 12:
            return e
    return best


def _snapshot_from_chain(puts, calls, S, expiry, days, get_iv, get_oi):
    """Shared logic: given parsed put/call rows, build the snapshot dict.
    puts/calls: list of dicts with keys 'strike' and 'iv' (decimal, may be None).
    get_iv(row): returns IV (decimal) for a row. get_oi(rows): total open interest."""
    T = max(days, 1) / 365.0
    if not puts or not calls:
        return None
    # ATM put (closest strike to spot)
    atm_put = min(puts, key=lambda r: abs(r["strike"] - S))
    atm_iv = atm_put.get("iv")
    exp_move = S * atm_iv * math.sqrt(T) if atm_iv else None
    # skew: OTM put IV - OTM call IV (one strike OTM)
    otm_puts = sorted([r for r in puts if r["strike"] < S], key=lambda r: -r["strike"])
    otm_calls = sorted([r for r in calls if r["strike"] > S], key=lambda r: r["strike"])
    skew = None
    if len(otm_puts) >= 2 and len(otm_calls) >= 2:
        ivp, ivc = otm_puts[1].get("iv"), otm_calls[1].get("iv")
        if ivp and ivc:
            skew = (ivp - ivc) * 100
    # full put smile
    smile = [{"strike": float(r["strike"]), "iv": round(r["iv"] * 100, 2)}
             for r in puts if r.get("iv") and 0.05 < r["iv"] < 5]
    smile.sort(key=lambda x: x["strike"])
    return {
        "expiry": expiry, "days": days,
        "atm_iv": round(atm_iv * 100, 2) if atm_iv else None,
        "exp_move": round(exp_move, 2) if exp_move else None,
        "exp_move_pct": round(exp_move / S * 100, 2) if exp_move else None,
        "skew": round(skew, 2) if skew else None,
        "smile": smile,
        "pc_oi": round(get_oi(puts) / max(get_oi(calls), 1), 2),
        "source": None,  # filled by caller
    }


def _tradier_options(symbol, S):
    """Tradier adapter: clean chain + ready-made Greeks/IV. Returns snapshot or None."""
    if not TRADIER_TOKEN:
        return None
    import urllib.request, urllib.parse, json as _json
    hdr = {"Authorization": "Bearer " + TRADIER_TOKEN, "Accept": "application/json"}

    def _get(path, params):
        url = TRADIER_BASE + path + "?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, headers=hdr)
        with urllib.request.urlopen(req, timeout=20) as r:
            return _json.loads(r.read().decode())

    try:
        # 1) expirations
        exp_data = _get("/markets/options/expirations", {"symbol": symbol})
        exps = exp_data.get("expirations", {}).get("date", [])
        if isinstance(exps, str):
            exps = [exps]
        expiry = _pick_expiry(exps)
        if not expiry:
            return None
        days = (dt.datetime.strptime(expiry, "%Y-%m-%d").date() - dt.date.today()).days
        # 2) chain with greeks
        ch = _get("/markets/options/chains",
                  {"symbol": symbol, "expiration": expiry, "greeks": "true"})
        opts = ch.get("options", {}).get("option", [])
        if not opts:
            return None
        puts, calls = [], []
        for o in opts:
            greeks = o.get("greeks") or {}
            iv = greeks.get("mid_iv") or greeks.get("smv_vol")  # Tradier's IV estimate
            row = {"strike": float(o["strike"]), "iv": (float(iv) if iv else None),
                   "oi": o.get("open_interest") or 0}
            if o.get("option_type") == "put":
                puts.append(row)
            elif o.get("option_type") == "call":
                calls.append(row)
        snap = _snapshot_from_chain(puts, calls, S, expiry, days,
                                    get_iv=lambda r: r.get("iv"),
                                    get_oi=lambda rows: sum(r["oi"] for r in rows))
        if snap:
            snap["source"] = "Tradier"
        return snap
    except Exception as e:
        print(f"    Tradier options failed for {symbol} ({e}); falling back")
        return None


def _yfinance_options(tk, S):
    """yfinance adapter: scrape chain, solve IV via Black-Scholes ourselves. Returns snapshot or None."""
    try:
        exps = tk.options
    except Exception:
        return None
    expiry = _pick_expiry(list(exps) if exps else [])
    if not expiry:
        return None
    days = (dt.datetime.strptime(expiry, "%Y-%m-%d").date() - dt.date.today()).days
    T = max(days, 1) / 365.0
    try:
        chain = tk.option_chain(expiry)
    except Exception:
        return None
    if chain.puts.empty or chain.calls.empty:
        return None

    def _mid(row):
        b, a = row.get("bid", 0) or 0, row.get("ask", 0) or 0
        return (a + b) / 2 if (a > 0 and b > 0) else (row.get("lastPrice", 0) or 0)

    def _is_tradeable(row):
        """Filter out junk contracts that produce garbage IV:
        require a real two-sided quote, a non-absurd bid-ask spread, and some liquidity."""
        b, a = row.get("bid", 0) or 0, row.get("ask", 0) or 0
        vol = row.get("volume", 0) or 0
        oi = row.get("openInterest", 0) or 0
        if b <= 0 or a <= 0:          # no two-sided quote
            return False
        mid = (a + b) / 2
        if mid <= 0:
            return False
        if (a - b) / mid > 0.5:        # spread wider than 50% of mid -> illiquid/stale
            return False
        if vol == 0 and oi == 0:       # nobody trading or holding it
            return False
        return True

    def _rows(df, opt):
        out = []
        for _, r in df.iterrows():
            if not _is_tradeable(r):
                continue
            iv = implied_vol(_mid(r), S, r["strike"], T, RISK_FREE_RATE, opt)
            # drop absurd solved IVs (the BS solver can still return extreme tails)
            if iv is not None and not (0.03 < iv < 4.0):
                iv = None
            out.append({"strike": float(r["strike"]), "iv": iv,
                        "oi": r.get("openInterest", 0) or 0})
        return out

    puts = _rows(chain.puts, "put")
    calls = _rows(chain.calls, "call")
    snap = _snapshot_from_chain(puts, calls, S, expiry, days,
                                get_iv=lambda r: r.get("iv"),
                                get_oi=lambda rows: sum(r["oi"] for r in rows))
    if snap:
        snap["source"] = "yfinance"
    return snap


def fetch_option_snapshot(tk, S, symbol=None):
    """Unified options interface: try Tradier first (if token set), else yfinance.
    Returns a snapshot dict (with 'source' field) or None."""
    sym = symbol or getattr(tk, "ticker", None)
    if TRADIER_TOKEN and sym:
        snap = _tradier_options(sym, S)
        if snap:
            return snap
    return _yfinance_options(tk, S)


# =============================================================
# SEC EDGAR - historical fundamentals & earnings quality (free, official, no token)
# =============================================================
# EDGAR requires a descriptive User-Agent with contact info per its fair-access policy.
EDGAR_UA = {"User-Agent": "QuantTool research contact@example.com"}
_CIK_MAP_CACHE = {}


def _edgar_get_json(url):
    import urllib.request, json as _json
    req = urllib.request.Request(url, headers=EDGAR_UA)
    with urllib.request.urlopen(req, timeout=25) as r:
        return _json.loads(r.read().decode())


def _load_cik_map():
    """Download EDGAR's ticker->CIK map once and cache it."""
    global _CIK_MAP_CACHE
    if _CIK_MAP_CACHE:
        return _CIK_MAP_CACHE
    try:
        data = _edgar_get_json("https://www.sec.gov/files/company_tickers.json")
        for _, row in data.items():
            _CIK_MAP_CACHE[row["ticker"].upper()] = str(row["cik_str"]).zfill(10)
    except Exception as e:
        print(f"  EDGAR ticker map download failed: {e}")
    return _CIK_MAP_CACHE


def _edgar_concept_series(cik, tag, unit="USD"):
    """Fetch a historical series for one XBRL concept (annual, form 10-K).
    Returns list of {fy, end, val} sorted by fiscal year, or []."""
    url = (f"https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}"
           f"/us-gaap/{tag}.json")
    try:
        data = _edgar_get_json(url)
    except Exception:
        return []
    items = data.get("units", {}).get(unit, [])
    annual = {}
    for it in items:
        # keep annual figures from 10-K filings
        if it.get("form") == "10-K" and it.get("fy") and it.get("fp") == "FY":
            fy = it["fy"]
            # later-filed value for the same FY overwrites (restatements -> latest)
            annual[fy] = {"fy": fy, "end": it.get("end"), "val": it.get("val")}
    return [annual[k] for k in sorted(annual)]


def fetch_edgar_fundamentals(symbol):
    """Pull historical fundamentals + earnings-quality signals from EDGAR.
    Returns dict or None. Key insight: cash flow vs net income divergence."""
    cikmap = _load_cik_map()
    cik = cikmap.get(symbol.upper())
    if not cik:
        return None
    # Net income & operating cash flow histories
    ni = _edgar_concept_series(cik, "NetIncomeLoss")
    cfo = _edgar_concept_series(cik, "NetCashProvidedByUsedInOperatingActivities")
    if not ni or not cfo:
        return None
    ni_by_fy = {x["fy"]: x["val"] for x in ni}
    cfo_by_fy = {x["fy"]: x["val"] for x in cfo}
    common = sorted(set(ni_by_fy) & set(cfo_by_fy))[-5:]  # last up to 5 fiscal years
    if not common:
        return None
    history = []
    flags = []
    for fy in common:
        n, c = ni_by_fy[fy], cfo_by_fy[fy]
        # accrual ratio proxy: (NI - CFO) / |NI|. High positive => earnings not backed by cash.
        ratio = ((n - c) / abs(n)) if n else None
        history.append({"fy": fy, "net_income": n, "cfo": c,
                        "accrual_ratio": round(ratio, 3) if ratio is not None else None})
    # quality flag: in recent years, is net income persistently above operating cash flow?
    recent = history[-3:]
    bad = sum(1 for h in recent if h["accrual_ratio"] is not None and h["accrual_ratio"] > 0.2)
    if bad >= 2:
        quality = "weak"   # paper profits not backed by cash in 2+ of last 3 years
    elif bad == 1:
        quality = "mixed"
    else:
        quality = "solid"  # cash flow keeps up with or exceeds reported earnings
    return {"cik": cik, "history": history, "cash_quality": quality}


# =============================================================
# FRED - macro indicators (Fed funds, CPI, Treasury yields, curve, VIX)
# Free, official (St. Louis Fed). API key optional; CSV fallback if absent.
# =============================================================
def _fred_series_csv(series_id, n_obs=400):
    """Fallback: download a series via FRED's public CSV endpoint (no key needed).
    Returns list of (date_str, float_value) or []."""
    import urllib.request, urllib.parse, csv, io, time
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?" + urllib.parse.urlencode({"id": series_id})
    last_err = None
    for attempt in range(3):  # retry a few times; FRED CSV can be flaky
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "Mozilla/5.0 (QuantTool research)",
                "Accept": "text/csv,*/*"})
            with urllib.request.urlopen(req, timeout=25) as r:
                text = r.read().decode()
            rows = list(csv.reader(io.StringIO(text)))
            out = []
            for row in rows[1:]:
                if len(row) < 2:
                    continue
                d, v = row[0], row[1]
                if v in (".", "", None):
                    continue
                try:
                    out.append((d, float(v)))
                except ValueError:
                    continue
            if out:
                return out[-n_obs:]
        except Exception as e:
            last_err = e
            time.sleep(1.0 * (attempt + 1))
    if last_err:
        print(f"    FRED CSV fetch failed for {series_id}: {last_err}")
    return []


def _fred_series_api(series_id, n_obs=400):
    """Use the official API when a key is set (no rate limit)."""
    import urllib.request, urllib.parse, json as _json
    params = {"series_id": series_id, "api_key": FRED_API_KEY, "file_type": "json",
              "sort_order": "asc"}
    url = "https://api.stlouisfed.org/fred/series/observations?" + urllib.parse.urlencode(params)
    try:
        with urllib.request.urlopen(url, timeout=25) as r:
            data = _json.loads(r.read().decode())
        out = []
        for o in data.get("observations", []):
            v = o.get("value")
            if v in (".", "", None):
                continue
            try:
                out.append((o["date"], float(v)))
            except ValueError:
                continue
        if out:
            return out[-n_obs:]
        print(f"    FRED API returned no observations for {series_id}; trying CSV")
        return _fred_series_csv(series_id, n_obs)
    except Exception as e:
        print(f"    FRED API failed for {series_id} ({e}); trying CSV fallback")
        return _fred_series_csv(series_id, n_obs)


def _fred_series(series_id, n_obs=400):
    return _fred_series_api(series_id, n_obs) if FRED_API_KEY else _fred_series_csv(series_id, n_obs)


def fetch_macro():
    """Fetch a macro snapshot from FRED. Returns a dict of indicators, each with
    latest value, recent change, and a small sparkline series. None on total failure."""
    series = {
        "fed_funds": ("DFF",     "%",  "Fed Funds Rate"),
        "cpi_yoy":   ("CPIAUCSL", "%",  "CPI Inflation (YoY)"),  # computed YoY below
        "ust10y":    ("DGS10",   "%",  "10Y Treasury Yield"),
        "curve_10_2":("T10Y2Y",  "%",  "10Y-2Y Spread"),
        "vix":       ("VIXCLS",  "",   "VIX (Volatility)"),
    }
    out = {}
    for key, (sid, unit, label) in series.items():
        obs = _fred_series(sid)
        if not obs:
            continue
        if key == "cpi_yoy":
            # CPI index -> year-over-year % change. Monthly data: ~12 obs per year.
            vals = obs
            yoy = []
            for i in range(12, len(vals)):
                prev = vals[i - 12][1]
                if prev:
                    yoy.append((vals[i][0], round((vals[i][1] / prev - 1) * 100, 2)))
            if not yoy:
                continue
            spark = [v for _, v in yoy[-24:]]
            latest = yoy[-1][1]
            prev_val = yoy[-2][1] if len(yoy) >= 2 else None
            out[key] = {"label": label, "unit": unit, "latest": latest,
                        "date": yoy[-1][0], "change": (round(latest - prev_val, 2) if prev_val is not None else None),
                        "spark": spark}
        else:
            spark = [v for _, v in obs[-60:]]
            latest = obs[-1][1]
            prev_val = obs[-2][1] if len(obs) >= 2 else None
            out[key] = {"label": label, "unit": unit, "latest": latest,
                        "date": obs[-1][0], "change": (round(latest - prev_val, 3) if prev_val is not None else None),
                        "spark": spark}
    if not out:
        return None
    out["_source"] = "FRED (api key)" if FRED_API_KEY else "FRED (csv)"
    return out


# =============================================================
# factor standardization & scoring
# =============================================================
def zscore_rank(series, ascending_good=True):
    """Convert to a 0-100 score. ascending_good=True means lower is better (e.g. P/E)."""
    s = series.astype(float)
    if s.notna().sum() < 2:
        return pd.Series([50] * len(s), index=s.index)
    z = (s - s.mean()) / (s.std() if s.std() > 0 else 1)
    if ascending_good:
        z = -z
    pct = pd.Series(norm.cdf(z.values) * 100, index=s.index)
    # where the raw value is missing, fall back to a neutral 50
    pct[s.isna()] = 50
    return pct.fillna(50)


# =============================================================
# Fama-French 5-factor + Momentum attribution
# =============================================================
FF5_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
MOM_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip"


def _download_ff_csv(url):
    """Download the ZIP from Ken French's site, parse a daily factor DataFrame (in %).
    No pandas-datareader dependency. Strategy: data rows start with an 8-digit date;
    the header is taken from the last non-empty alpha line before data, with generic fallback."""
    import urllib.request, zipfile, io, re
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        zbytes = resp.read()
    zf = zipfile.ZipFile(io.BytesIO(zbytes))
    name = zf.namelist()[0]
    raw = zf.read(name).decode("latin-1")
    lines = raw.splitlines()

    date_re = re.compile(r"^\s*\d{8}\s*,")  # line start: 8-digit date followed by comma
    rows = []
    header = None
    last_text_parts = None  # remember the most recent header-looking line

    for ln in lines:
        if date_re.match(ln):
            parts = [p.strip() for p in ln.split(",")]
            date = parts[0]
            try:
                vals = [float(x) for x in parts[1:] if x != ""]
            except ValueError:
                continue
            if not vals:
                continue
            # on first data row, use the remembered text line as header
            if header is None:
                if last_text_parts and len(last_text_parts) >= len(vals):
                    header = last_text_parts[-len(vals):]
                else:
                    header = [f"F{i+1}" for i in range(len(vals))]
            rows.append([date] + vals[:len(header)])
        else:
            # remember a possible header line: has letters, comma-separated, not a date
            # threshold >=1 to support the momentum file (single "Mom" column)
            p = [x.strip() for x in ln.split(",")]
            named = [x for x in p if x and any(c.isalpha() for c in x)]
            # cap at <=8 tokens to exclude copyright/descriptive sentences
            if 1 <= len(named) <= 8 and len(p) >= 2:
                last_text_parts = named

    if not rows or header is None:
        raise ValueError(f"could not parse {name} (no date data rows found)")

    # align column counts (guard against ragged rows)
    ncol = len(header)
    rows = [r for r in rows if len(r) == ncol + 1]
    df = pd.DataFrame(rows, columns=["date"] + list(header))
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["date"]).set_index("date")
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(how="all")


def fetch_ff_factors():
    """Download and merge FF5 + momentum factors -> daily DataFrame (MKT,SMB,HML,RMW,CMA,MOM,RF; in %).
    Returns None on failure; the main flow then skips attribution."""
    try:
        ff5 = _download_ff_csv(FF5_URL)
        mom = _download_ff_csv(MOM_URL)
    except Exception as e:
        print(f"  FF factor download failed, skipping attribution: {e}")
        return None

    # --- FF5 column normalization (fuzzy match, fall back to known order) ---
    rename5 = {}
    for c in ff5.columns:
        cl = c.lower().replace("-", "").replace(" ", "")
        if "mkt" in cl: rename5[c] = "MKT"
        elif cl == "smb": rename5[c] = "SMB"
        elif cl == "hml": rename5[c] = "HML"
        elif cl == "rmw": rename5[c] = "RMW"
        elif cl == "cma": rename5[c] = "CMA"
        elif cl == "rf": rename5[c] = "RF"
    ff5 = ff5.rename(columns=rename5)
    expected5 = ["MKT", "SMB", "HML", "RMW", "CMA", "RF"]
    if not all(x in ff5.columns for x in expected5):
        # fallback: FF5 daily standard column order is Mkt-RF, SMB, HML, RMW, CMA, RF
        if ff5.shape[1] == 6:
            ff5.columns = expected5
        else:
            print(f"  unexpected FF5 column count ({ff5.shape[1]}), skipping attribution")
            return None

    # --- momentum column (the momentum file usually has one column) ---
    mom_col = None
    for c in mom.columns:
        if "mom" in c.lower():
            mom_col = c; break
    if mom_col is None and mom.shape[1] >= 1:
        mom_col = mom.columns[0]  # fallback: take the first column
    mom = mom.rename(columns={mom_col: "MOM"})

    df = ff5.join(mom[["MOM"]], how="inner")
    keep = [c for c in ["MKT", "SMB", "HML", "RMW", "CMA", "MOM", "RF"] if c in df.columns]
    out = df[keep].dropna()
    print(f"  FF factors ready: {len(out)} trading days, columns={list(out.columns)}")
    return out


def run_attribution(stock_returns, ff, label):
    """Run an FF6 attribution regression on a daily return series (decimal, e.g. 0.012).
    stock_returns: pd.Series indexed by date. ff: factor DataFrame (in %)."""
    import statsmodels.api as sm
    factors = ["MKT", "SMB", "HML", "RMW", "CMA", "MOM"]
    factors = [f for f in factors if f in ff.columns]
    # align dates; convert stock returns to % to match FF units
    sr = (stock_returns * 100).rename("ret")
    merged = pd.concat([sr, ff], axis=1, join="inner").dropna()
    if len(merged) < 60:
        return None
    y = merged["ret"] - merged["RF"]
    X = sm.add_constant(merged[factors])
    m = sm.OLS(y, X).fit()
    betas = {f: {"beta": round(float(m.params[f]), 3),
                 "t": round(float(m.tvalues[f]), 2)} for f in factors}
    return {
        "label": label,
        "n_days": int(len(merged)),
        "alpha_daily": round(float(m.params["const"]), 4),
        "alpha_annual": round(float(m.params["const"]) * 252, 2),  # %
        "alpha_t": round(float(m.tvalues["const"]), 2),
        "alpha_p": round(float(m.pvalues["const"]), 4),
        "r2": round(float(m.rsquared), 3),
        "betas": betas,
    }


# =============================================================
# Backtest -- no look-ahead, includes costs, with safety locks
# =============================================================
TRADING_DAYS = 252


def perf_metrics(daily_returns, rf_annual=RISK_FREE_RATE):
    """Compute performance metrics from a daily return series (decimal)."""
    r = pd.Series(daily_returns).dropna()
    if len(r) < 20:
        return None
    rf_daily = rf_annual / TRADING_DAYS
    excess = r - rf_daily
    ann_ret = (1 + r).prod() ** (TRADING_DAYS / len(r)) - 1
    ann_vol = r.std() * math.sqrt(TRADING_DAYS)
    sharpe = (excess.mean() / r.std() * math.sqrt(TRADING_DAYS)) if r.std() > 0 else None
    # Sortino: penalize only downside volatility
    downside = r[r < 0]
    dvol = downside.std() * math.sqrt(TRADING_DAYS) if len(downside) > 1 else None
    sortino = (excess.mean() * TRADING_DAYS / dvol) if dvol and dvol > 0 else None
    # max drawdown
    equity = (1 + r).cumprod()
    runmax = equity.cummax()
    dd = (equity / runmax - 1)
    max_dd = float(dd.min())
    calmar = (ann_ret / abs(max_dd)) if max_dd < 0 else None
    return {
        "ann_return": round(float(ann_ret) * 100, 2),
        "ann_vol": round(float(ann_vol) * 100, 2),
        "sharpe": round(float(sharpe), 2) if sharpe is not None else None,
        "sortino": round(float(sortino), 2) if sortino is not None else None,
        "max_dd": round(max_dd * 100, 2),
        "calmar": round(float(calmar), 2) if calmar is not None else None,
        "n_days": int(len(r)),
        "equity_curve": [round(float(v), 4) for v in equity.iloc[::max(1, len(equity)//120)].values],
    }


def _factor_score_asof(price_panel, asof_idx, lookback_mom=252, skip=21, vol_win=252):
    """No look-ahead: using only data up to asof_idx, compute each stock's (momentum+low-vol) score.
    Returns {sym: score}. Momentum = cross-sectional rank of 12-1 return; low-vol = inverse rank of realized vol.
    Note: V/Q factors need historical fundamentals (unavailable without look-ahead), so the backtest uses the price-volume subset only."""
    mom, vol = {}, {}
    for sym in price_panel.columns:
        s = price_panel[sym].iloc[:asof_idx + 1].dropna()
        if len(s) < lookback_mom + 5:
            continue
        # 12-1 momentum
        mom[sym] = s.iloc[-skip] / s.iloc[-lookback_mom] - 1
        # realized volatility
        rets = np.log(s / s.shift(1)).dropna()
        vol[sym] = rets.tail(vol_win).std() * math.sqrt(TRADING_DAYS)
    if len(mom) < 2:
        return {}
    mom_s = pd.Series(mom)
    vol_s = pd.Series(vol)
    # cross-sectional percentile (higher momentum better; lower vol better)
    mom_rank = mom_s.rank(pct=True)
    vol_rank = (-vol_s).rank(pct=True)
    score = (mom_rank + vol_rank) / 2
    return score.to_dict()


def run_backtest(price_panel, rebalance_days=21, top_frac=0.5,
                 cost_bps=10, oos_split=0.5, lookback_mom=252):
    """
    No-look-ahead rolling factor backtest, comparing "factor strategy" vs "equal-weight benchmark".
    - price_panel: DataFrame(index=date, columns=ticker, values=close)
    - rebalance_days: rebalance period (trading days)
    - top_frac: strategy buys the top fraction by score
    - cost_bps: one-way transaction cost (bps), charged on turnover at rebalance
    - oos_split: out-of-sample split point (0.5 = first half in-sample, second half OOS)
    Returns a dict with strategy/benchmark perf, in/out-sample comparison, equity curve; None on failure.
    """
    if price_panel.shape[1] < 2 or len(price_panel) < lookback_mom + 60:
        return None
    daily_ret = price_panel.pct_change().fillna(0)
    n = len(price_panel)
    start = lookback_mom + 1  # first index where a factor score can be computed

    strat_rets, bench_rets = [], []
    dates = []
    cur_weights = None
    cost_rate = cost_bps / 10000.0
    nstocks = price_panel.shape[1]
    ntop = max(1, int(round(nstocks * top_frac)))

    for i in range(start, n):
        # rebalance day: set weights from info up to i-1 (yesterday), hold today -> no look-ahead
        if (i - start) % rebalance_days == 0:
            scores = _factor_score_asof(price_panel, i - 1, lookback_mom=lookback_mom)
            new_w = {s: 0.0 for s in price_panel.columns}
            if scores:
                ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
                picks = [s for s, _ in ranked[:ntop]]
                for s in picks:
                    new_w[s] = 1.0 / len(picks)
            # turnover cost
            turnover = 0.0
            if cur_weights is not None:
                turnover = sum(abs(new_w[s] - cur_weights.get(s, 0)) for s in new_w)
            cost = turnover * cost_rate
            cur_weights = new_w
        else:
            cost = 0.0
        # daily strategy return = weighted holdings - cost (charged once at rebalance)
        day_r = sum(cur_weights[s] * daily_ret[s].iloc[i] for s in cur_weights) - cost
        strat_rets.append(day_r)
        bench_rets.append(daily_ret.iloc[i].mean())  # equal-weight benchmark
        dates.append(price_panel.index[i])

    strat = pd.Series(strat_rets, index=dates)
    bench = pd.Series(bench_rets, index=dates)

    # in-sample / out-of-sample split
    split = int(len(strat) * oos_split)
    result = {
        "full": {"strategy": perf_metrics(strat), "benchmark": perf_metrics(bench)},
        "in_sample": {"strategy": perf_metrics(strat.iloc[:split]),
                      "benchmark": perf_metrics(bench.iloc[:split])},
        "out_sample": {"strategy": perf_metrics(strat.iloc[split:]),
                       "benchmark": perf_metrics(bench.iloc[split:])},
        "params": {"rebalance_days": rebalance_days, "top_frac": top_frac,
                   "cost_bps": cost_bps, "oos_split": oos_split},
        "window": {"start": str(price_panel.index[start].date()),
                   "end": str(price_panel.index[-1].date())},
        "n_stocks": nstocks,
    }
    return result


def param_sensitivity(price_panel):
    """Parameter sensitivity: sweep (rebalance period x top fraction), check if OOS Sharpe is stable.
    Wild swings across parameters -> overfitting warning."""
    grid = []
    for rb in [10, 21, 42, 63]:
        for tf in [0.33, 0.5, 0.67]:
            bt = run_backtest(price_panel, rebalance_days=rb, top_frac=tf)
            if bt and bt["out_sample"]["strategy"]:
                grid.append({
                    "rebalance_days": rb, "top_frac": tf,
                    "oos_sharpe": bt["out_sample"]["strategy"]["sharpe"],
                    "oos_return": bt["out_sample"]["strategy"]["ann_return"],
                })
    if not grid:
        return None
    sharpes = [g["oos_sharpe"] for g in grid if g["oos_sharpe"] is not None]
    return {
        "grid": grid,
        "sharpe_min": round(min(sharpes), 2) if sharpes else None,
        "sharpe_max": round(max(sharpes), 2) if sharpes else None,
        "sharpe_spread": round(max(sharpes) - min(sharpes), 2) if len(sharpes) > 1 else None,
    }


def main():
    import bridge_module as bm
    _bridge = bm.load_bridge()
    watchlist = bm.watchlist_symbols(_bridge)   # falls back to watchlist.txt if bridge absent
    print(f"watchlist: {watchlist}")
    RISK_FREE_RATE = bm.risk_free_rate()        # FRED DGS3MO, falls back to 0.045
    rows = []
    returns_map = {}   # for correlation matrix: symbol -> daily return Series
    price_map = {}     # for backtest: symbol -> full close-price Series (3y)
    print("Fetching data ...")
    for sym in watchlist:
        print(f"  {sym} ...")
        try:
            tk = yf.Ticker(sym)
            f = fetch_fundamentals(tk)
            S = f["price"]
            close = fetch_price_history(tk)   # 3y
            if S is None and len(close):
                S = float(close.iloc[-1])
            # display factors use a recent ~1y (252 trading-day) slice
            close_1y = close.tail(TRADING_DAYS) if len(close) > TRADING_DAYS else close
            mom = compute_momentum(close_1y)
            rvol = compute_realized_vol(close_1y)
            ivr = iv_rank_proxy(close_1y)
            multi_mom = compute_multi_momentum(close_1y)
            dd = compute_drawdown(close_1y)
            price_series = downsample_prices(close_1y)
            opt = fetch_option_snapshot(tk, S, symbol=sym) if S else None
            # per-stock performance: run metrics on recent-1y simple daily returns
            stock_perf = None
            simple_ret_1y = close_1y.pct_change().dropna()
            if len(simple_ret_1y) >= 60:
                sp = perf_metrics(simple_ret_1y)
                if sp:
                    sp.pop("equity_curve", None)  # drop curve to save size
                    stock_perf = sp
            if len(close) >= 30:
                returns_map[sym] = np.log(close_1y / close_1y.shift(1)).dropna()
            if len(close) >= 60:
                price_map[sym] = close
            # EDGAR historical fundamentals & earnings quality (skip ETFs; US filers only)
            edgar = None
            if not f["is_etf"]:
                try:
                    edgar = fetch_edgar_fundamentals(sym)
                except Exception as e:
                    print(f"    EDGAR skipped for {sym}: {e}")
            rows.append({
                "symbol": sym, "name": f["name"], "price": S,
                "sector": f["sector"], "is_etf": f["is_etf"],
                "pe": f["pe"], "pb": f["pb"], "roe": f["roe"], "de": f["de"],
                "ev_ebitda": f["ev_ebitda"], "profit_margin": f["profit_margin"],
                "momentum": mom, "realized_vol": rvol, "iv_rank_proxy": ivr,
                "multi_mom": multi_mom, "drawdown": dd, "price_series": price_series,
                "perf": stock_perf, "edgar": edgar,
                "option": opt,
            })
        except Exception as e:
            print(f"    skipped {sym}: {e}")
            rows.append({"symbol": sym, "name": sym, "price": None,
                         "sector": "N/A", "is_etf": False,
                         "pe": None, "pb": None, "roe": None, "de": None,
                         "ev_ebitda": None, "profit_margin": None,
                         "momentum": None, "realized_vol": None, "iv_rank_proxy": None,
                         "multi_mom": None, "drawdown": None, "price_series": None,
                         "perf": None, "edgar": None,
                         "option": None})

    df = pd.DataFrame(rows)

    # factor scoring
    df["value_score"] = (zscore_rank(df["pe"], True) + zscore_rank(df["pb"], True)) / 2
    df["quality_score"] = (zscore_rank(df["roe"], False) + zscore_rank(df["profit_margin"], False)
                           + zscore_rank(df["de"], True)) / 3
    df["momentum_score"] = zscore_rank(df["momentum"], False)
    df["lowvol_score"] = zscore_rank(df["realized_vol"], True)
    df["composite"] = (
        WEIGHTS["value"] * df["value_score"] +
        WEIGHTS["quality"] * df["quality_score"] +
        WEIGHTS["momentum"] * df["momentum_score"] +
        WEIGHTS["lowvol"] * df["lowvol_score"]
    )
    df = df.sort_values("composite", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1

    # correlation matrix (daily returns, aligned on intersection dates)
    corr = None
    returns_panel = None
    if len(returns_map) >= 2:
        ret_df = pd.DataFrame(returns_map).dropna()
        if len(ret_df) >= 20:
            cm = ret_df.corr()
            corr = {"symbols": list(cm.columns),
                    "matrix": [[round(float(cm.iloc[i, j]), 2) for j in range(len(cm))]
                               for i in range(len(cm))]}
            # for the portfolio sandbox: aligned daily simple-return matrix
            simple = (np.exp(ret_df) - 1)  # log returns -> simple returns
            returns_panel = {
                "symbols": list(simple.columns),
                "dates": [d.strftime("%Y-%m-%d") for d in simple.index],
                # one column of daily returns (decimal) per stock, rounded to save size
                "returns": {c: [round(float(v), 5) for v in simple[c].values]
                            for c in simple.columns},
            }

    # Fama-French 5-factor + momentum attribution
    print("Downloading Fama-French factors and running attribution ...")
    attribution = None
    try:
        import statsmodels.api  # noqa: F401
        _has_sm = True
    except ImportError:
        _has_sm = False
        print("  statsmodels not installed, skipping attribution. Install: pip install statsmodels")
    ff = fetch_ff_factors() if _has_sm else None
    if ff is not None and returns_map:
        ff.index = ff.index.tz_localize(None)
        singles = []
        for sym, rets in returns_map.items():
            r = rets.copy()
            r.index = pd.to_datetime(r.index).tz_localize(None)
            res = run_attribution(r, ff, sym)
            if res:
                singles.append(res)
        # equal-weight portfolio attribution
        port = None
        if len(returns_map) >= 2:
            rdf = pd.DataFrame({s: r for s, r in returns_map.items()})
            rdf.index = pd.to_datetime(rdf.index).tz_localize(None)
            port_ret = rdf.mean(axis=1).dropna()
            port = run_attribution(port_ret, ff, "PORTFOLIO")
        attribution = {"singles": singles, "portfolio": port,
                       "factors": [c for c in ["MKT","SMB","HML","RMW","CMA","MOM"] if c in ff.columns],
                       "window_start": str(ff.index.min().date()),
                       "window_end": str(ff.index.max().date())}

    # backtest (no-look-ahead rolling factor + equal-weight benchmark + OOS split + costs + sensitivity)
    print("Backtesting ...")
    backtest = None
    if len(price_map) >= 2:
        panel = pd.DataFrame(price_map)
        panel.index = pd.to_datetime(panel.index).tz_localize(None)
        panel = panel.sort_index().dropna(how="all")
        bt = run_backtest(panel)
        if bt:
            bt["sensitivity"] = param_sensitivity(panel)
            bt["note_small_sample"] = (len(price_map) < 20)
            backtest = bt
            print(f"  backtest done: {bt['window']['start']} -> {bt['window']['end']}, {bt['n_stocks']} stocks")
        else:
            print("  insufficient data, skipping backtest")

    # macro snapshot from FRED (fed funds, CPI, treasury yields, curve, VIX)
    print("Fetching macro indicators from FRED ...")
    macro = None
    try:
        macro = fetch_macro()
        if macro:
            print(f"  macro ready: {', '.join(k for k in macro if not k.startswith('_'))}")
        else:
            print("  macro fetch returned nothing, skipping")
    except Exception as e:
        print(f"  macro fetch failed: {e}")

    # export HTML
    payload = df.to_dict(orient="records")
    html = build_html(payload, corr, watchlist, attribution, backtest, returns_panel, macro)
    html = html.replace("'__THEMES__'",
    json.dumps(json.dumps(bm.themes_for_template(_bridge), ensure_ascii=False), ensure_ascii=False))
    with open(OUTPUT_HTML, "w", encoding="utf-8") as fp:
        fp.write(html)
    print(f"\nDone -> {OUTPUT_HTML}")


def build_html(data, corr, watchlist, attribution=None, backtest=None, returns_panel=None, macro=None):
    """Generate the self-contained bilingual HTML. All datasets injected as JSON."""
    import os
    data_json = json.dumps(data, ensure_ascii=False, default=lambda o: None)
    corr_json = json.dumps(corr, ensure_ascii=False) if corr else "null"
    wl_json = json.dumps(watchlist, ensure_ascii=False)
    attr_json = json.dumps(attribution, ensure_ascii=False) if attribution else "null"
    bt_json = json.dumps(backtest, ensure_ascii=False) if backtest else "null"
    rp_json = json.dumps(returns_panel, ensure_ascii=False) if returns_panel else "null"
    macro_json = json.dumps(macro, ensure_ascii=False) if macro else "null"
    generated = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    tpl_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "quant_report_template.html")
    with open(tpl_path, encoding="utf-8") as fp:
        tpl = fp.read()
    return (tpl.replace("__DATA__", data_json)
               .replace("__CORR__", corr_json)
               .replace("__WATCHLIST__", wl_json)
               .replace("__ATTRIBUTION__", attr_json)
               .replace("__BACKTEST__", bt_json)
               .replace("__RETURNS_PANEL__", rp_json)
               .replace("__MACRO__", macro_json)
               .replace("__GENERATED__", generated))

if __name__ == "__main__":
    main()
