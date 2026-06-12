#!/usr/bin/env python3
"""
bridge_module.py — Tier 2 companion to quant_prototype.py
===========================================================
Connects the local research engine to the Robinhood bridge files
(watchlists.json / portfolio.json / contracts.json / journal.csv)
exported by Claude (Tier 1) and refreshed by the Tue/Fri routine.

New capabilities on top of quant_prototype:
  1. Bridge ingestion (replaces watchlist.txt; falls back if absent)
  2. Portfolio analytics on ACTUAL holdings:
       - per-position beta vs SPY, value-weighted portfolio beta
         (cash included at beta=0 — "cash is a position")
       - FF5+Momentum exposure of the live portfolio
       - holdings correlation matrix
       - marginal contribution to risk (MCR)
       - theme weights via watchlist membership
  3. Alpha Ledger: per closed trade, r_trade − beta·r_SPY over the
     holding period, joined with process grades from journal.csv —
     the "three-ledger" split: beta money / alpha money / luck.
  4. FRED risk-free rate (replaces the hardcoded 0.045)

Usage:
    python bridge_module.py                 # full portfolio report
    python bridge_module.py --ledger-only   # just the alpha ledger
Integration with quant_prototype.py: see README_TIER2.md (3-line patch).

Dependencies: same as quant_prototype (yfinance, pandas, numpy, scipy).
"""

import os, sys, json, csv, datetime as dt
import numpy as np
import pandas as pd

import yfinance as yf
import quant_prototype as qp   # reuse: fetch_price_history, fetch_ff_factors,
                               # run_attribution, _fred_series, build_html styling

def _hist(symbol, period="1y"):
    """quant_prototype.fetch_price_history expects a yf.Ticker OBJECT, not a str.
    All price fetches in this module go through this wrapper."""
    return qp.fetch_price_history(yf.Ticker(symbol), period=period)

BRIDGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bridge")

# ----------------------------------------------------------------
# 0. Risk-free from FRED (3-month T-bill), with graceful fallback
# ----------------------------------------------------------------
def risk_free_rate():
    try:
        s = qp._fred_series("DGS3MO", n_obs=10)
        if s is not None and len(s):
            return float(s.dropna().iloc[-1]) / 100.0
    except Exception:
        pass
    return qp.RISK_FREE_RATE  # fallback to the prototype's constant

# ----------------------------------------------------------------
# 1. Bridge ingestion
# ----------------------------------------------------------------
def load_bridge(bridge_dir=BRIDGE_DIR):
    """Load all four bridge files. Returns dict; missing files -> None values."""
    out = {"watchlists": None, "portfolio": None, "contracts": None, "journal": None}
    def _j(name):
        p = os.path.join(bridge_dir, name)
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as fp:
                return json.load(fp)
        return None
    out["watchlists"] = _j("watchlists.json")
    out["portfolio"]  = _j("portfolio.json")
    out["contracts"]  = _j("contracts.json")
    jp = os.path.join(bridge_dir, "journal.csv")
    if os.path.exists(jp):
        with open(jp, "r", encoding="utf-8") as fp:
            out["journal"] = list(csv.DictReader(fp))
    return out

def watchlist_symbols(bridge):
    """Union of all theme-list symbols -> replaces load_watchlist()."""
    if not bridge or not bridge.get("watchlists"):
        return qp.load_watchlist()                      # fallback: old behavior
    syms = []
    for wl in bridge["watchlists"]["watchlists"]:
        syms.extend(wl["symbols"])
    return sorted(set(syms))

def theme_map(bridge):
    """symbol -> [themes] (a symbol can live on several lists)."""
    m = {}
    if bridge and bridge.get("watchlists"):
        for wl in bridge["watchlists"]["watchlists"]:
            for s in wl["symbols"]:
                m.setdefault(s, []).append(wl["theme"])
    return m

def themes_for_template(bridge):
    """{watchlist display name: [symbols]} — feeds the screener's theme tabs."""
    if not bridge or not bridge.get("watchlists"):
        return {}
    return {wl["name"]: wl["symbols"] for wl in bridge["watchlists"]["watchlists"]}

def positions_frame(bridge):
    """Flatten accounts -> DataFrame [account, symbol, qty, avg_cost, bucket]."""
    rows = []
    cash = {}
    for acct in bridge["portfolio"]["accounts"]:
        cash[acct["label"]] = float(acct.get("cash_usd", 0.0))
        for p in acct["positions"]:
            rows.append({"account": acct["label"], "symbol": p["symbol"],
                         "qty": float(p["qty"]), "avg_cost": float(p["avg_cost"]),
                         "bucket": p.get("bucket", "")})
    return pd.DataFrame(rows), cash

def _capm_ols(asset_ret, mkt_ret, rf_annual):
    """OLS of daily excess returns; returns dict or None."""
    df = pd.concat([asset_ret, mkt_ret], axis=1).dropna()
    if len(df) < 120: return None
    rf_d = rf_annual / 252.0
    y = df.iloc[:, 0].values - rf_d
    x = df.iloc[:, 1].values - rf_d
    n = len(y); vx = np.var(x, ddof=1)
    if vx <= 0: return None
    beta = float(np.cov(y, x, ddof=1)[0, 1] / vx)
    alpha_d = float(np.mean(y) - beta * np.mean(x))
    resid = y - (alpha_d + beta * x)
    s2 = float(resid @ resid) / (n - 2)
    se_a = np.sqrt(s2 * (1.0/n + np.mean(x)**2 / ((n-1)*vx)))
    se_b = np.sqrt(s2 / ((n-1)*vx))
    ss_tot = float(((y - y.mean())**2).sum())
    return {"alpha_annual": round(float(alpha_d*252*100), 2),
            "t_alpha": round(float(alpha_d/se_a), 2) if se_a > 0 else None,
            "beta": round(float(beta), 2),
            "t_beta": round(float(beta/se_b), 1) if se_b > 0 else None,
            "r2": round(float(1.0 - (resid @ resid)/ss_tot), 2) if ss_tot > 0 else None,
            "n": int(n)}

def attach_capm(payload, price_map, lookback=252):
    """Screener hook: attach a 'capm' dict to each payload record.
    price_map: {symbol: close Series} (main()'s backtest panel). One SPY fetch total."""
    try:
        spy_ret = _hist("SPY", period="3y").pct_change().dropna().tail(lookback)
    except Exception:
        spy_ret = None
    rf = risk_free_rate()
    for rec in payload:
        rec["capm"] = None
        if spy_ret is None: continue
        c = price_map.get(rec.get("symbol"))
        if c is None or len(c) < 130: continue
        try:
            res = _capm_ols(c.pct_change().dropna().tail(lookback), spy_ret, rf)
            if res and res["t_alpha"] is not None and res["t_beta"] is not None and res["r2"] is not None:
                rec["capm"] = res
        except Exception:
            pass
    return payload

# ----------------------------------------------------------------
# 2. Portfolio analytics
# ----------------------------------------------------------------
def _beta(asset_ret, mkt_ret):
    """OLS beta, aligned, NaN-safe."""
    df = pd.concat([asset_ret, mkt_ret], axis=1).dropna()
    if len(df) < 60:
        return np.nan
    x = df.iloc[:, 1].values
    y = df.iloc[:, 0].values
    vx = np.var(x)
    return float(np.cov(y, x)[0, 1] / vx) if vx > 0 else np.nan

def portfolio_analytics(bridge, period="1y"):
    """Compute the full portfolio X-ray. Returns dict of frames/scalars."""
    pos, cash = positions_frame(bridge)
    tmap = theme_map(bridge)
    symbols = sorted(set(pos["symbol"]))

    # --- price panel (holdings + SPY benchmark)
    panel = {}
    for tk in symbols + ["SPY"]:
        close = _hist(tk, period=period)
        if close is not None and len(close) > 60:
            panel[tk] = close
    prices = pd.DataFrame(panel).dropna(how="all")
    rets = prices.pct_change().dropna(how="all")
    last = prices.ffill().iloc[-1]

    # --- per-position market value & weights (cash included)
    pos["price"] = pos["symbol"].map(last)
    pos["value"] = pos["qty"] * pos["price"]
    pos["pnl"]   = (pos["price"] - pos["avg_cost"]) * pos["qty"]
    total_cash   = sum(cash.values())
    total_value  = pos["value"].sum() + total_cash
    pos["weight"] = pos["value"] / total_value
    cash_weight   = total_cash / total_value

    # --- betas
    spy_ret = rets["SPY"]
    pos["beta"] = pos["symbol"].map(lambda s: _beta(rets[s], spy_ret) if s in rets else np.nan)
    # value-weighted; cash enters at beta = 0 — this IS "cash is a position"
    port_beta_ex_cash = float(np.nansum(pos["beta"] * pos["value"]) / max(pos["value"].sum(), 1e-9))
    port_beta_incl_cash = float(np.nansum(pos["beta"] * pos["weight"]))  # weights already vs total incl cash

    # --- theme weights
    pos["themes"] = pos["symbol"].map(lambda s: ",".join(tmap.get(s, ["unmapped"])))
    theme_w = {}
    for _, r in pos.iterrows():
        for th in (tmap.get(r["symbol"]) or ["unmapped"]):
            theme_w[th] = theme_w.get(th, 0.0) + r["weight"]
    theme_w["cash"] = cash_weight
    theme_w = pd.Series(theme_w).sort_values(ascending=False)

    # --- correlation matrix (holdings only)
    held = [s for s in symbols if s in rets.columns]
    corr = rets[held].corr() if len(held) >= 2 else pd.DataFrame()

    # --- portfolio daily return series (current weights, ex-cash drift ignored)
    w = pos.set_index("symbol")["weight"]
    w = w.groupby(level=0).sum()           # merge duplicate symbols across accounts
    common = [s for s in w.index if s in rets.columns]
    port_ret = (rets[common] * w[common]).sum(axis=1)   # cash adds 0 daily

    # --- marginal contribution to risk
    mcr = pd.Series(dtype=float)
    if len(common) >= 2:
        cov = rets[common].cov() * 252
        wv = w[common].values
        port_var = float(wv @ cov.values @ wv)
        if port_var > 0:
            mcr = pd.Series((cov.values @ wv) * wv / port_var, index=common).sort_values(ascending=False)

    # --- FF5+Mom exposure of the live portfolio (reuse prototype machinery)
    attribution = None
    try:
        ff = qp.fetch_ff_factors()
        if ff is not None:
            attribution = qp.run_attribution(port_ret, ff, label="LIVE PORTFOLIO")
    except Exception as e:
        attribution = {"error": f"FF attribution unavailable: {e}"}

    # --- per-holding analytics (computed locally; no signature guessing)
    def _h_detail(sym):
        if sym not in prices.columns: return None
        c = prices[sym].dropna()
        if len(c) < 60: return None
        def ret(d): return float(c.iloc[-1] / c.iloc[-d] - 1) if len(c) > d else np.nan
        r = c.pct_change().dropna()
        vol = float(r.std() * np.sqrt(252))
        dd = float((c / c.cummax() - 1).min())
        hi52 = float(c.iloc[-1] / c.max() - 1)
        roll = r.rolling(21).std().dropna() * np.sqrt(252)
        ivr = float((roll.iloc[-1] <= roll).mean() * 100) if len(roll) > 30 else np.nan  # realized-vol rank proxy
        return {"symbol": sym, "r_1m": ret(21), "r_3m": ret(63), "r_6m": ret(126),
                "r_12m": ret(252), "vol": vol, "max_dd": dd, "from_52w_high": hi52, "rvol_rank": ivr}
    hd = [d for d in (_h_detail(s) for s in sorted(set(pos["symbol"]))) if d]
    holdings_detail = pd.DataFrame(hd)

    rf_annual = risk_free_rate()
    held_syms = sorted(set(pos["symbol"]))

    # --- CAPM per holding: OLS of daily excess returns on SPY excess returns
    def _capm(sym):
        if sym not in rets.columns or "SPY" not in rets.columns: return None
        df = pd.concat([rets[sym], rets["SPY"]], axis=1).dropna()
        if len(df) < 120: return None
        rf_d = rf_annual / 252.0
        y = df.iloc[:, 0].values - rf_d
        x = df.iloc[:, 1].values - rf_d
        n = len(y)
        vx = np.var(x, ddof=1)
        if vx <= 0: return None
        beta = float(np.cov(y, x, ddof=1)[0, 1] / vx)
        alpha_d = float(np.mean(y) - beta * np.mean(x))
        resid = y - (alpha_d + beta * x)
        s2 = float(resid @ resid) / (n - 2)
        se_a = np.sqrt(s2 * (1.0/n + np.mean(x)**2 / ((n-1)*vx)))
        se_b = np.sqrt(s2 / ((n-1)*vx))
        ss_tot = float(((y - y.mean())**2).sum())
        r2 = 1.0 - float(resid @ resid)/ss_tot if ss_tot > 0 else np.nan
        return {"symbol": sym, "alpha_annual": alpha_d*252*100, "t_alpha": alpha_d/se_a if se_a > 0 else np.nan,
                "beta": beta, "t_beta": beta/se_b if se_b > 0 else np.nan, "r2": r2, "n": n}
    capm = pd.DataFrame([c for c in (_capm(s) for s in held_syms) if c])

    # --- FF5+Mom per holding (reuse the prototype's attribution machinery)
    ff_singles = []
    try:
        ff = qp.fetch_ff_factors()
        if ff is not None:
            for s_ in held_syms:
                if s_ not in rets.columns: continue
                try:
                    res = qp.run_attribution(rets[s_], ff, label=s_)
                    if res: ff_singles.append(res)
                except Exception:
                    pass
    except Exception:
        pass

    # --- QVM+L cross-sectional scores WITHIN the holdings set
    # Same factor logic as the screener (P/E,P/B | ROE,margin,D/E | 12-1 mom | -vol),
    # percentile-ranked across current holdings only.
    def _pct_rank(sr, low_is_good=False):
        r = sr.rank(pct=True, na_option="keep") * 100
        return (100 - r) if low_is_good else r
    qvml = None
    try:
        fund = {}
        for s_ in held_syms:
            try:
                f = qp.fetch_fundamentals(yf.Ticker(s_)) or {}
            except Exception:
                f = {}
            c = prices[s_].dropna() if s_ in prices.columns else pd.Series(dtype=float)
            mom = float(c.iloc[-21] / c.iloc[-252] - 1) if len(c) > 252 else np.nan   # 12-1 momentum
            vol = float(c.pct_change().std() * np.sqrt(252)) if len(c) > 60 else np.nan
            fund[s_] = {"pe": f.get("pe"), "pb": f.get("pb"), "roe": f.get("roe"),
                        "pm": f.get("profit_margin"), "de": f.get("de"), "mom": mom, "vol": vol}
        fd = pd.DataFrame(fund).T.astype(float)
        if len(fd) >= 3:
            value    = pd.concat([_pct_rank(fd["pe"], True), _pct_rank(fd["pb"], True)], axis=1).mean(axis=1)
            quality  = pd.concat([_pct_rank(fd["roe"]), _pct_rank(fd["pm"]), _pct_rank(fd["de"], True)], axis=1).mean(axis=1)
            momentum = _pct_rank(fd["mom"])
            lowvol   = _pct_rank(fd["vol"], True)
            qvml = pd.DataFrame({"V": value, "Q": quality, "M": momentum, "L": lowvol})
            qvml["composite"] = qvml.mean(axis=1)
            qvml = qvml.sort_values("composite", ascending=False)
    except Exception:
        qvml = None

    # --- per-scope aggregates for the account tabs (all / main / agentic)
    def _scope(label):
        if label == "all":
            p = pos; csh = total_cash
        else:
            p = pos[pos["account"] == label]
            csh = cash.get(label, 0.0)
        tv = float(p["value"].sum()) + csh
        if tv <= 0:
            return None
        w_sym = p.groupby("symbol")["value"].sum() / tv
        b_ex = float(np.nansum(p["beta"] * p["value"]) / max(p["value"].sum(), 1e-9)) if len(p) else float("nan")
        b_in = float(np.nansum(p["beta"] * (p["value"] / tv))) if len(p) else 0.0
        tw = {}
        for _, r_ in p.iterrows():
            for th_ in (tmap.get(r_["symbol"]) or ["unmapped"]):
                tw[th_] = tw.get(th_, 0.0) + r_["value"] / tv
        tw["cash"] = csh / tv
        tw = pd.Series(tw).sort_values(ascending=False)
        syms_ = [s_ for s_ in sorted(set(p["symbol"])) if s_ in rets.columns]
        corr_ = rets[syms_].corr() if len(syms_) >= 2 else pd.DataFrame()
        mcr_ = pd.Series(dtype=float)
        if len(syms_) >= 2:
            cov_ = rets[syms_].cov() * 252
            wv_ = w_sym.reindex(syms_).fillna(0).values
            pv_ = float(wv_ @ cov_.values @ wv_)
            if pv_ > 0:
                mcr_ = pd.Series((cov_.values @ wv_) * wv_ / pv_, index=syms_).sort_values(ascending=False)
        return {"total_value": tv, "cash": csh, "cash_weight": csh / tv,
                "beta_incl": b_in, "beta_ex": b_ex,
                "theme_weights": tw, "corr": corr_, "mcr": mcr_}
    scopes = {k: v for k, v in (("all", _scope("all")), ("main", _scope("main")),
                                ("agentic", _scope("agentic"))) if v}
    sym_accts = pos.groupby("symbol")["account"].apply(lambda x: ",".join(sorted(set(x)))).to_dict()

    return {"positions": pos.sort_values("value", ascending=False),
            "scopes": scopes, "sym_accts": sym_accts,
            "holdings_detail": holdings_detail,
            "capm": capm, "ff_singles": ff_singles, "qvml": qvml,
            "cash": cash, "cash_weight": cash_weight, "total_value": total_value,
            "port_beta_incl_cash": port_beta_incl_cash,
            "port_beta_ex_cash": port_beta_ex_cash,
            "theme_weights": theme_w, "corr": corr, "mcr": mcr,
            "port_ret": port_ret, "attribution": attribution,
            "rf": risk_free_rate()}

# ----------------------------------------------------------------
# 3. Alpha Ledger
# ----------------------------------------------------------------
def _num(x):
    """Parse journal numerics that may carry ~, +, est markers; NaN if hopeless."""
    if x is None: return np.nan
    t = str(x).strip().replace("~", "").replace("+", "")
    for junk in ["(est; user to fill exact)", "(est)", "est"]:
        t = t.replace(junk, "")
    t = t.strip().rstrip(",")
    try:
        return float(t)
    except ValueError:
        return np.nan

def alpha_ledger(bridge, period="1y"):
    """
    For each CLOSED trade with parseable entry/exit and a date:
        alpha_$ = realized_pnl − beta_i × r_SPY(holding window) × entry_notional
    Holding window: trade date -> trade date (intraday) uses 1-day SPY return as
    the beta leg; multi-day trades need entry/exit dates (journal v2 fields
    'opened'/'closed' supported if present).
    Output: per-trade table + the three-ledger aggregate.
    """
    rows = bridge.get("journal") or []
    if not rows:
        return None
    spy = _hist("SPY", period=period)
    spy_ret_by_day = spy.pct_change()

    out = []
    for r in rows:
        pnl = _num(r.get("realized_pnl"))
        if np.isnan(pnl):
            continue                       # open trade or unparseable -> skip
        sym   = r.get("symbol", "?")
        entry = _num(r.get("entry")); qty = abs(_num(r.get("qty")))
        notional = entry * qty if not (np.isnan(entry) or np.isnan(qty)) else np.nan
        d_open  = pd.to_datetime(r.get("opened") or r.get("date"))
        d_close = pd.to_datetime(r.get("closed") or r.get("date"))
        # beta of the symbol over the period (best effort)
        try:
            close = _hist(sym, period=period)
            b = _beta(close.pct_change(), spy.pct_change())
        except Exception:
            b = np.nan
        # SPY return over holding window (inclusive); intraday -> that day's return
        try:
            win = spy_ret_by_day.loc[d_open:d_close]
            spy_r = float((1 + win.fillna(0)).prod() - 1) if len(win) else 0.0
        except Exception:
            spy_r = 0.0
        beta_dollars  = (b * spy_r * notional) if not any(np.isnan(v) for v in [b, notional]) else np.nan
        alpha_dollars = pnl - beta_dollars if not np.isnan(beta_dollars) else np.nan
        out.append({"date": str(r.get("date")), "account": r.get("account"), "symbol": sym,
                    "side": r.get("side"), "pnl_$": pnl,
                    "beta": None if np.isnan(b) else round(b, 2),
                    "beta_$": None if np.isnan(beta_dollars) else round(beta_dollars, 2),
                    "alpha_$": None if np.isnan(alpha_dollars) else round(alpha_dollars, 2),
                    "rule_based": (r.get("rule_based", "").strip().lower().startswith("yes")),
                    "grade": r.get("process_grade", "")})
    ledger = pd.DataFrame(out)
    if ledger.empty:
        return None
    agg = {
        "closed_trades": len(ledger),
        "total_pnl_$": round(float(ledger["pnl_$"].sum()), 2),
        "rule_based_pnl_$": round(float(ledger.loc[ledger.rule_based, "pnl_$"].sum()), 2),
        "discretionary_pnl_$": round(float(ledger.loc[~ledger.rule_based, "pnl_$"].sum()), 2),
        "alpha_$_sum": round(float(ledger["alpha_$"].dropna().sum()), 2),
        "hit_rate": round(float((ledger["pnl_$"] > 0).mean()), 2),
    }
    return {"trades": ledger, "aggregate": agg}

# ----------------------------------------------------------------
# 4. HTML report (matches quant_report_template design language)
# ----------------------------------------------------------------
HTML_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "portfolio_report.html")

_CSS = """
:root{--bg:#131a26;--panel:#1b2433;--panel2:#222d3f;--line:#374559;--ink:#e3eaf5;
--dim:#94a2ba;--faint:#5a6880;--green:#4ade9a;--red:#ff6b70;--amber:#f5bd3f;--blue:#6bb3ff;
--accent:#4ade9a;--mono:'JetBrains Mono','SF Mono',ui-monospace,Menlo,Consolas,monospace;
--serif:'Spectral',Georgia,'Songti SC','Noto Serif SC',serif}
*{box-sizing:border-box;margin:0;padding:0}
body{background:radial-gradient(1000px 560px at 85% -10%,rgba(74,222,154,.08),transparent),
radial-gradient(760px 440px at 0% 110%,rgba(107,179,255,.06),transparent),var(--bg);
color:var(--ink);font-family:var(--mono);font-feature-settings:"tnum" 1;line-height:1.5;padding:0 0 80px}
.wrap{max-width:1280px;margin:0 auto;padding:0 40px}
header{display:flex;align-items:flex-end;justify-content:space-between;padding:38px 0 22px;
border-bottom:1px solid var(--line);flex-wrap:wrap;gap:16px}
.brand h1{font-family:var(--serif);font-weight:600;font-size:32px}
.brand .sub{color:var(--dim);font-size:13px;margin-top:7px;letter-spacing:.5px}
.gen{color:var(--faint);font-size:11px}
.epigraph{margin:24px 0 8px;padding:18px 26px;border-left:3px solid var(--accent);
background:linear-gradient(90deg,rgba(63,207,142,.05),transparent);border-radius:0 10px 10px 0}
.epi-zh{font-family:var(--serif);font-size:17px;line-height:1.7;letter-spacing:.5px}
.epi-en{font-family:var(--serif);font-style:italic;font-size:13px;color:var(--dim);margin-top:8px}
.epi-by{font-size:11px;color:var(--faint);margin-top:10px;text-align:right;letter-spacing:1px}
.section-label{font-size:12.5px;letter-spacing:2.5px;color:var(--dim);text-transform:uppercase;
margin:36px 0 14px;display:flex;align-items:center;gap:12px}
.section-label .sl-rule{flex:1;height:1px;background:var(--line)}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:16px 18px}
.card .k{font-size:11px;color:var(--dim);letter-spacing:1.5px;text-transform:uppercase}
.card .v{font-size:25px;font-weight:700;margin-top:6px}
.tbl{width:100%;border-collapse:collapse;font-size:14px}
.tbl thead th{text-align:right;color:var(--dim);font-weight:500;font-size:11.5px;letter-spacing:1px;
padding:10px 12px;border-bottom:1px solid var(--line);text-transform:uppercase;white-space:nowrap}
.tbl thead th.l{text-align:left}
.tbl tbody td{padding:11px 12px;text-align:right;border-bottom:1px solid rgba(34,45,61,.5)}
.tbl tbody td.l{text-align:left}
.sym{font-weight:700;font-size:14px}
.pos{color:var(--green)}.neg{color:var(--red)}
.bar{height:5px;border-radius:3px;background:var(--line);overflow:hidden;margin-top:5px;min-width:90px}
.bar>i{display:block;height:100%;border-radius:3px;background:var(--accent)}
.lamp{display:inline-block;padding:2px 10px;border-radius:20px;font-size:11px;font-weight:700}
.lamp.g{background:rgba(74,222,154,.15);color:var(--green);border:1px solid var(--green)}
.lamp.y{background:rgba(245,189,63,.15);color:var(--amber);border:1px solid var(--amber)}
.lamp.r{background:rgba(255,107,112,.15);color:var(--red);border:1px solid var(--red)}
pre{background:var(--panel);border:1px solid var(--line);border-radius:10px;
padding:16px 18px;font-size:12px;overflow-x:auto;color:var(--dim)}
.note{color:var(--faint);font-size:12.5px;margin-top:8px;font-style:italic}
.sig{color:var(--green);font-weight:700}.insig{color:var(--faint)}
.closing-poem{font-family:var(--serif);text-align:center;color:var(--dim);font-size:15px;
line-height:2.1;margin:64px 0 10px;letter-spacing:1px}
.epi-quote{margin-bottom:16px}.epi-quote:last-child{margin-bottom:0}
.acct-tabs{display:flex;gap:8px;flex-wrap:wrap;margin:26px 0 0}
.acct-tab{background:transparent;border:1px solid var(--line);color:var(--dim);
padding:7px 16px;border-radius:8px;font-family:var(--mono);font-size:12.5px;cursor:pointer;transition:.15s}
.acct-tab:hover{border-color:var(--accent);color:var(--accent)}
.acct-tab.on{background:var(--accent);color:#06231a;border-color:var(--accent);font-weight:700}
.lang-toggle{display:flex;border:1px solid var(--line);border-radius:7px;overflow:hidden}
.lang-toggle button{background:transparent;color:var(--dim);border:0;padding:7px 15px;
font-family:var(--mono);font-size:12px;cursor:pointer;letter-spacing:.5px;transition:.18s}
.lang-toggle button.on{background:var(--accent);color:#06231a;font-weight:700}
.hm{border-collapse:separate;border-spacing:5px;margin:8px auto 0;width:100%;max-width:880px;table-layout:fixed}
.hm td,.hm th{height:62px;text-align:center;font-size:15px;border-radius:8px;padding:0}
.hm th{color:var(--dim);font-weight:600;font-size:13px;background:transparent;letter-spacing:.5px}
.hm td{font-weight:700;color:#0a0e14}
.hm .corner{width:62px}
.hm .diag{color:var(--faint);background:var(--panel2)!important}
"""

def _fmt_pnl(v):
    cls = "pos" if v >= 0 else "neg"
    return f'<span class="{cls}">{v:+,.2f}</span>'

def _sec(title):
    return f'<div class="section-label"><span>{title}</span><span class="sl-rule"></span></div>'

I18N_P = {
 "zh": {"title":"一飞云霄夜航帆 - 股市持仓分析","sub":"CAPM, QVM+L, and Fama-French 量化分析 · 两账户实时持仓",
   "overview":"总览","contracts":"合同审计","positions":"当前持仓","hold":"持仓个股指标",
   "capm":"CAPM 归因(个股)","ff":"Fama-French 五因子 + 动量(个股)","qvml":"QVM+L 因子评分(持仓组内横截面)",
   "themes":"主题权重","mcr":"边际风险贡献","corr":"持仓相关性","attr":"组合因子归因","ledger":"Alpha 账本",
   "k_total":"组合总值","k_cash":"现金占比","k_bic":"组合 β(计入现金)","k_bxc":"组合 β(剔除现金)","k_rf":"无风险利率(FRED)",
   "n_beta":"「计入现金」口径将全部现金按 β = 0 纳入组合权重——现金本身是一个仓位,而非缺席。",
   "n_ledger":"盈亏 = β$ + α$:β$ 为市场暴露所致部分,α$ 为剔除暴露后的主动盈亏。「规则内」与「自由裁量」之分栏,构成「系统优于冲动」假说的周度检验。",
   "n_corr":"红为正相关,蓝为负相关,色标与 Screener 一致。普遍的高正相关,意味着名义上的多标的持有并未带来有效分散。",
   "n_capm":"对每只持仓的日度超额收益对市场超额收益做回归。α 年化为截距,|t| ≥ 2 视为统计显著;R² 为市场可解释比例。",
   "n_ff":"在 CAPM 之上加入规模、价值、盈利、投资与动量五个系统性因子。若 α 不显著,则该股收益可由已知因子暴露完全解释。",
   "n_qvml":"与 Screener 同一套因子构造(估值 V、质量 Q、动量 M、低波动 L),但仅在当前持仓内做横截面排名——分数为组内相对值,样本较小,宜作参考而非定论。",
   "h_date":"日期","h_acct":"账户","h_rule":"规则","h_grade":"过程评分",
   "k_trades":"已平仓笔数","k_pnl":"总盈亏","k_rule":"规则内盈亏","k_disc":"自由裁量盈亏","k_hit":"胜率",
   "tab_all":"全部","tab_main":"主账户","tab_agentic":"Agentic 账户(由 Claude 操作)"},
 "en": {"title":"Yifan the Nightfarer - Portfolio Holdings Analysis","sub":"CAPM, QVM+L, and Fama-French Quantitative Analysis · Live holdings across both accounts",
   "overview":"Overview","contracts":"Contract Audit","positions":"Positions","hold":"Holdings Metrics",
   "capm":"CAPM Attribution (per holding)","ff":"Fama-French 5 + Momentum (per holding)","qvml":"QVM+L Factor Scores (cross-section within holdings)",
   "themes":"Theme Weights","mcr":"Marginal Risk Contribution","corr":"Holdings Correlation","attr":"Portfolio Factor Attribution","ledger":"Alpha Ledger",
   "k_total":"Total Value","k_cash":"Cash Weight","k_bic":"Portfolio β (cash included)","k_bxc":"Portfolio β (ex cash)","k_rf":"Risk-free Rate (FRED)",
   "n_beta":"The cash-included measure weights all cash at β = 0 — cash is itself a position, not an absence.",
   "n_ledger":"P&L = β$ + α$: β$ is the component attributable to market exposure; α$ is the active residual. The rule-based vs discretionary split provides a weekly test of the system-over-impulse hypothesis.",
   "n_corr":"Red denotes positive and blue negative correlation, on the same scale as the Screener. Pervasively high positive correlation indicates that holding many names has produced nominal rather than effective diversification.",
   "n_capm":"Each holding's daily excess return is regressed on the market excess return. Annualized α is the intercept; |t| ≥ 2 is treated as statistically significant; R² is the share explained by the market.",
   "n_ff":"Extends CAPM with size, value, profitability, investment, and momentum factors. An insignificant α implies the holding's return is fully explained by known factor exposures.",
   "n_qvml":"Same factor construction as the Screener (Value, Quality, Momentum, Low-volatility), ranked cross-sectionally within current holdings only — scores are relative to this small set and should be read as indicative.",
   "h_date":"Date","h_acct":"Account","h_rule":"Rule","h_grade":"Process Grade",
   "k_trades":"Closed Trades","k_pnl":"Total P&L","k_rule":"Rule-based P&L","k_disc":"Discretionary P&L","k_hit":"Hit Rate",
   "tab_all":"Combined","tab_main":"Main","tab_agentic":"Agentic (Claude-operated)"}}

def _corr_color(v):
    """Same ramp as the screener's corrCellColor: blue(neg) -> dark grey(0) -> red(pos)."""
    v = max(-1.0, min(1.0, 0.0 if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)))
    if v >= 0:
        a = v
        return f"rgb({round(34+(255-34)*a)},{round(45+(107-45)*a)},{round(63+(112-63)*a)})"
    a = -v
    return f"rgb({round(34+(107-34)*a)},{round(45+(179-45)*a)},{round(63+(255-63)*a)})"

def _i(key):
    """dual-language span; JS toggles visibility."""
    return f'<span class="t-zh" data-k="{key}"></span>'

def build_portfolio_html(pa, ledger, contracts, out_path=HTML_OUT):
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    def sec(key):
        return f'<div class="section-label"><span data-i18n="{key}"></span><span class="sl-rule"></span></div>'
    def card(key, v):
        return f'<div class="card"><div class="k" data-i18n="{key}"></div><div class="v">{v}</div></div>'

    h = [f"""<!DOCTYPE html><html lang="zh"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Portfolio X-Ray · {now}</title><style>{_CSS}</style></head><body><div class="wrap">
<header><div class="brand"><h1 data-i18n="title"></h1>
<div class="sub" data-i18n="sub"></div></div>
<div style="display:flex;align-items:center;gap:14px">
<div class="lang-toggle"><button id="btn-zh" class="on" onclick="setLang('zh')">中文</button><button id="btn-en" onclick="setLang('en')">EN</button></div>
<div class="gen">generated {now}</div></div></header>
<div class="epigraph">
<div class="epi-quote">
<div class="epi-zh">"终朝只恨聚无多,及到多时眼闭了。"</div>
<div class="epi-en">"All day he frets that his hoard is still too small; the day it is enough, his eyes close for good."</div>
<div class="epi-by">— 曹雪芹《红楼梦 · 好了歌》</div></div>
<div class="epi-quote">
<div class="epi-zh">"这是尘寰中消长数应当,何必枉悲伤?"</div>
<div class="epi-en">"Such waxing and waning is the appointed course of the mortal world — why grieve in vain?"</div>
<div class="epi-by">— 曹雪芹《红楼梦 · 乐中悲》</div></div></div>
<div class="acct-tabs">
<button class="acct-tab on" id="atab-all" onclick="setScope('all')" data-i18n="tab_all"></button>
<button class="acct-tab" id="atab-main" onclick="setScope('main')" data-i18n="tab_main"></button>
<button class="acct-tab" id="atab-agentic" onclick="setScope('agentic')" data-i18n="tab_agentic"></button>
</div>"""]

    h.append(sec("overview"))
    h.append('<div class="cards">')
    h.append(card("k_total", f'<span id="ov_total">${pa["total_value"]:,.0f}</span>'))
    h.append(card("k_cash", f'<span id="ov_cash">{pa["cash_weight"]*100:.1f}%</span>'))
    h.append(card("k_bic", f'<span id="ov_bic">{pa["port_beta_incl_cash"]:.2f}</span>'))
    h.append(card("k_bxc", f'<span id="ov_bxc">{pa["port_beta_ex_cash"]:.2f}</span>'))
    h.append(card("k_rf", f"{pa['rf']*100:.2f}%"))
    h.append('</div><div class="note" data-i18n="n_beta"></div>')

    if contracts:
        h.append(sec("contracts"))
        h.append('<table class="tbl"><thead><tr><th class="l">ID</th><th class="l">Thesis</th><th>Status</th></tr></thead><tbody>')
        for c in contracts.get("contracts", []):
            st = c.get("status", "")
            cls = "r" if ("VIOLATION" in st.upper()) else ("y" if ("PENDING" in st.upper() or "watch" in st) else "g")
            h.append(f'<tr data-acct="{c.get("account","")}"><td class="l sym">{c["id"]}</td><td class="l">{c.get("thesis_type","")}</td>'
                     f'<td><span class="lamp {cls}">{st}</span></td></tr>')
        h.append('</tbody></table>')

    h.append(sec("positions"))
    h.append('<table class="tbl"><thead><tr><th class="l">Symbol</th><th class="l">Acct</th>'
             '<th>Qty</th><th>Cost</th><th>Price</th><th>Value</th><th>P&L</th><th>Weight</th><th>β</th><th class="l">Themes</th></tr></thead><tbody>')
    for _, r in pa["positions"].iterrows():
        beta_txt = "" if pd.isna(r["beta"]) else f'{r["beta"]:.2f}'
        h.append(f'<tr data-acct="{r["account"]}"><td class="l sym">{r["symbol"]}</td><td class="l">{r["account"]}</td>'
                 f'<td>{r["qty"]:g}</td><td>{r["avg_cost"]:,.2f}</td><td>{r["price"]:,.2f}</td>'
                 f'<td>{r["value"]:,.2f}</td><td>{_fmt_pnl(r["pnl"])}</td>'
                 f'<td>{r["weight"]*100:.2f}%</td><td>{beta_txt}</td>'
                 f'<td class="l" style="color:var(--dim);font-size:11px">{r["themes"]}</td></tr>')
    h.append('</tbody></table>')

    # --- per-holding analytics
    hd = pa.get("holdings_detail")
    if hd is not None and len(hd):
        h.append(sec("hold"))
        h.append('<table class="tbl"><thead><tr><th class="l">Symbol</th><th>1M</th><th>3M</th><th>6M</th>'
                 '<th>12M</th><th>Vol (ann.)</th><th>Max DD</th><th>vs 52W High</th><th>RVol Rank</th></tr></thead><tbody>')
        def pc(v):
            if v is None or (isinstance(v, float) and np.isnan(v)): return "—"
            cls = "pos" if v >= 0 else "neg"
            return f'<span class="{cls}">{v*100:+.1f}%</span>'
        for _, r in hd.iterrows():
            rv = "—" if pd.isna(r["rvol_rank"]) else f'{r["rvol_rank"]:.0f}'
            h.append(f'<tr data-acct="{pa.get("sym_accts",{}).get(r["symbol"],"")}"><td class="l sym">{r["symbol"]}</td><td>{pc(r["r_1m"])}</td><td>{pc(r["r_3m"])}</td>'
                     f'<td>{pc(r["r_6m"])}</td><td>{pc(r["r_12m"])}</td><td>{r["vol"]*100:.0f}%</td>'
                     f'<td>{pc(r["max_dd"])}</td><td>{pc(r["from_52w_high"])}</td><td>{rv}</td></tr>')
        h.append('</tbody></table>')

    # --- CAPM per holding
    capm = pa.get("capm")
    if capm is not None and len(capm):
        h.append(sec("capm"))
        h.append('<table class="tbl"><thead><tr><th class="l">Symbol</th><th>α (ann.)</th><th>t(α)</th>'
                 '<th>β</th><th>t(β)</th><th>R²</th><th>N</th></tr></thead><tbody>')
        for _, r in capm.iterrows():
            sig = abs(r["t_alpha"]) >= 2
            acls = ("pos" if r["alpha_annual"] >= 0 else "neg") if sig else "insig"
            h.append(f'<tr data-acct="{pa.get("sym_accts",{}).get(r["symbol"],"")}"><td class="l sym">{r["symbol"]}</td>'
                     f'<td><span class="{acls}">{r["alpha_annual"]:+.1f}%</span></td>'
                     f'<td class="{"sig" if sig else "insig"}">{r["t_alpha"]:.2f}</td>'
                     f'<td>{r["beta"]:.2f}</td><td>{r["t_beta"]:.1f}</td>'
                     f'<td>{r["r2"]:.2f}</td><td>{int(r["n"])}</td></tr>')
        h.append('</tbody></table><div class="note" data-i18n="n_capm"></div>')

    # --- FF5+Mom per holding
    ffs = pa.get("ff_singles") or []
    ffs = [r for r in ffs if isinstance(r, dict) and "betas" in r]
    if ffs:
        h.append(sec("ff"))
        factors = list(ffs[0]["betas"].keys())
        h.append('<table class="tbl"><thead><tr><th class="l">Symbol</th><th>α (ann.)</th><th>t(α)</th>'
                 + "".join(f"<th>{f_}</th>" for f_ in factors) + '<th>R²</th></tr></thead><tbody>')
        for r in ffs:
            sig = abs(r.get("alpha_t", 0)) >= 2
            acls = ("pos" if r.get("alpha_annual", 0) >= 0 else "neg") if sig else "insig"
            cells = ""
            for f_ in factors:
                b = r["betas"].get(f_, {})
                bsig = abs(b.get("t", 0)) >= 2
                cells += f'<td class="{"sig" if bsig else "insig"}">{b.get("beta", float("nan")):+.2f}</td>'
            h.append(f'<tr data-acct="{pa.get("sym_accts",{}).get(r.get("label",""),"")}"><td class="l sym">{r.get("label","")}</td>'
                     f'<td><span class="{acls}">{r.get("alpha_annual",0):+.1f}%</span></td>'
                     f'<td class="{"sig" if sig else "insig"}">{r.get("alpha_t",0):.2f}</td>'
                     + cells + f'<td>{r.get("r2",float("nan")):.2f}</td></tr>')
        h.append('</tbody></table><div class="note" data-i18n="n_ff"></div>')

    # --- QVM+L within holdings
    qv = pa.get("qvml")
    if qv is not None and len(qv):
        h.append(sec("qvml"))
        h.append('<table class="tbl"><thead><tr><th class="l">Symbol</th><th>V</th><th>Q</th>'
                 '<th>M</th><th>L</th><th>Composite</th></tr></thead><tbody>')
        def _scell(v):
            if v is None or (isinstance(v, float) and np.isnan(v)): return '<td>—</td>'
            return (f'<td><span style="font-weight:600">{v:.0f}</span>'
                    f'<div class="bar"><i style="width:{max(2,min(100,v)):.0f}%"></i></div></td>')
        for sym_, r in qv.iterrows():
            h.append(f'<tr data-acct="{pa.get("sym_accts",{}).get(sym_,"")}"><td class="l sym">{sym_}</td>' + _scell(r["V"]) + _scell(r["Q"])
                     + _scell(r["M"]) + _scell(r["L"])
                     + f'<td><span style="font-weight:700;font-size:16px">{r["composite"]:.0f}</span></td></tr>')
        h.append('</tbody></table><div class="note" data-i18n="n_qvml"></div>')

    scopes = pa.get("scopes") or {"all": {"theme_weights": pa["theme_weights"], "mcr": pa["mcr"], "corr": pa["corr"],
                                          "total_value": pa["total_value"], "cash_weight": pa["cash_weight"],
                                          "beta_incl": pa["port_beta_incl_cash"], "beta_ex": pa["port_beta_ex_cash"]}}

    def _themes_tbl(tw):
        out = ['<table class="tbl"><tbody>']
        mx = max(tw.max(), 1e-9)
        for th, w in tw.items():
            out.append(f'<tr><td class="l sym" style="width:160px">{th}</td>'
                       f'<td style="width:90px">{w*100:.1f}%</td>'
                       f'<td class="l"><div class="bar"><i style="width:{w/mx*100:.0f}%"></i></div></td></tr>')
        out.append('</tbody></table>')
        return "".join(out)

    def _mcr_tbl(mcr):
        if not len(mcr): return ""
        out = ['<table class="tbl"><tbody>']
        for s_, v in mcr.items():
            out.append(f'<tr><td class="l sym" style="width:160px">{s_}</td><td style="width:90px">{v*100:.1f}%</td>'
                       f'<td class="l"><div class="bar"><i style="width:{max(v,0)/max(mcr.max(),1e-9)*100:.0f}%"></i></div></td></tr>')
        out.append('</tbody></table>')
        return "".join(out)

    def _hm_tbl(corr):
        if not isinstance(corr, pd.DataFrame) or corr.empty: return ""
        syms_ = list(corr.columns)
        out = ['<table class="hm"><thead><tr><th class="corner"></th>' + "".join(f"<th>{x}</th>" for x in syms_) + '</tr></thead><tbody>']
        for a in syms_:
            cells = []
            for b in syms_:
                v = float(corr.loc[a, b])
                if a == b:
                    cells.append(f'<td class="diag">{v:.2f}</td>')
                else:
                    txtcol = "#fff" if abs(v) > 0.45 else "#0a0e14"
                    cells.append(f'<td style="background:{_corr_color(v)};color:{txtcol}">{v:.2f}</td>')
            out.append(f'<tr><th>{a}</th>' + "".join(cells) + '</tr>')
        out.append('</tbody></table>')
        return "".join(out)

    h.append(sec("themes"))
    for sc, d in scopes.items():
        vis = "" if sc == "all" else ' style="display:none"'
        h.append(f'<div class="scope-blk" data-scope="{sc}" data-kind="themes"{vis}>' + _themes_tbl(d["theme_weights"]) + '</div>')

    h.append(sec("mcr"))
    for sc, d in scopes.items():
        vis = "" if sc == "all" else ' style="display:none"'
        h.append(f'<div class="scope-blk" data-scope="{sc}" data-kind="mcr"{vis}>' + _mcr_tbl(d["mcr"]) + '</div>')

    h.append(sec("corr"))
    for sc, d in scopes.items():
        vis = "" if sc == "all" else ' style="display:none"'
        h.append(f'<div class="scope-blk" data-scope="{sc}" data-kind="corr"{vis}>' + _hm_tbl(d["corr"]) + '</div>')
    h.append('<div class="note" data-i18n="n_corr"></div>')

    if pa.get("attribution"):
        h.append(sec("attr"))
        h.append('<pre>' + str(pa["attribution"]) + '</pre>')

    if ledger:
        h.append(sec("ledger"))
        h.append('<table class="tbl"><thead><tr><th class="l" data-i18n="h_date"></th><th class="l" data-i18n="h_acct"></th><th class="l">Symbol</th>'
                 '<th>P&L $</th><th>β</th><th>β $</th><th>α $</th><th data-i18n="h_rule"></th><th class="l" data-i18n="h_grade"></th></tr></thead><tbody>')
        for _, r in ledger["trades"].iterrows():
            h.append(f'<tr data-acct="{r["account"]}"><td class="l">{r["date"]}</td><td class="l">{r["account"]}</td><td class="l sym">{r["symbol"]}</td>'
                     f'<td>{_fmt_pnl(r["pnl_$"])}</td><td>{r["beta"] if r["beta"] is not None else ""}</td>'
                     f'<td>{r["beta_$"] if r["beta_$"] is not None else ""}</td>'
                     f'<td>{r["alpha_$"] if r["alpha_$"] is not None else ""}</td>'
                     f'<td>{"✓" if r["rule_based"] else "✗"}</td><td class="l">{r["grade"]}</td></tr>')
        a = ledger["aggregate"]
        h.append('</tbody></table>')
        h.append('<div class="cards" style="margin-top:14px">')
        h.append(card("k_trades", a["closed_trades"]))
        h.append(card("k_pnl", f"${a['total_pnl_$']:,.2f}"))
        h.append(card("k_rule", f"${a['rule_based_pnl_$']:,.2f}"))
        h.append(card("k_disc", f"${a['discretionary_pnl_$']:,.2f}"))
        h.append(card("k_hit", f"{a['hit_rate']*100:.0f}%"))
        h.append('</div><div class="note" data-i18n="n_ledger"></div>')

    h.append('<div class="closing-poem">浮生着甚苦奔忙,盛席华筵终散场。<br>悲喜千般同幻渺,古今一梦尽荒唐。</div>')
    scopes_js = {k: {"total": f"${v['total_value']:,.0f}", "cash": f"{v['cash_weight']*100:.1f}%",
                     "bic": f"{v['beta_incl']:.2f}", "bxc": f"{v['beta_ex']:.2f}"}
                 for k, v in scopes.items()}
    h.append(f"""</div>
<script>
const I18N = {json.dumps(I18N_P, ensure_ascii=False)};
const SCOPES = {json.dumps(scopes_js, ensure_ascii=False)};
let SCOPE = "all";
function setScope(sc){{
  if(!SCOPES[sc]) return;
  SCOPE = sc;
  ["all","main","agentic"].forEach(k=>{{
    const b=document.getElementById("atab-"+k);
    if(b) b.className = "acct-tab" + (k===sc ? " on" : "");
  }});
  const d=SCOPES[sc];
  document.getElementById("ov_total").textContent=d.total;
  document.getElementById("ov_cash").textContent=d.cash;
  document.getElementById("ov_bic").textContent=d.bic;
  document.getElementById("ov_bxc").textContent=d.bxc;
  document.querySelectorAll("tr[data-acct]").forEach(tr=>{{
    const a=tr.getAttribute("data-acct");
    tr.style.display = (sc==="all" || a.split(",").includes(sc)) ? "" : "none";
  }});
  document.querySelectorAll(".scope-blk").forEach(el=>{{
    el.style.display = (el.getAttribute("data-scope")===sc) ? "" : "none";
  }});
}}
let LANG = "zh";
function setLang(l){{
  LANG = l;
  document.getElementById("btn-zh").className = l==="zh" ? "on" : "";
  document.getElementById("btn-en").className = l==="en" ? "on" : "";
  document.querySelectorAll("[data-i18n]").forEach(el=>{{
    const k = el.getAttribute("data-i18n");
    if(I18N[l][k] !== undefined) el.textContent = I18N[l][k];
  }});
  document.querySelectorAll(".epi-zh").forEach(el=>{{ el.style.display = (l==="en") ? "none" : "block"; }});
}}
setLang("zh");
</script></body></html>""")
    with open(out_path, "w", encoding="utf-8") as fp:
        fp.write("\n".join(h))
    return out_path

# ----------------------------------------------------------------
# 5. Report (terminal + optional HTML fragment in prototype style)
# ----------------------------------------------------------------
def print_report(pa, ledger, contracts):
    line = "=" * 64
    print(line); print("PORTFOLIO X-RAY  (live holdings, both accounts)"); print(line)
    print(f"total value ${pa['total_value']:,.2f}   cash {pa['cash_weight']*100:.1f}%   rf(FRED) {pa['rf']*100:.2f}%")
    print(f"portfolio beta INCL cash: {pa['port_beta_incl_cash']:.2f}    ex-cash: {pa['port_beta_ex_cash']:.2f}")
    print("\n-- theme weights --");  print((pa["theme_weights"] * 100).round(1).to_string())
    cols = ["account", "symbol", "qty", "avg_cost", "price", "value", "pnl", "weight", "beta", "themes"]
    print("\n-- positions --");      print(pa["positions"][cols].round(3).to_string(index=False))
    if len(pa["mcr"]):
        print("\n-- marginal contribution to risk --"); print((pa["mcr"] * 100).round(1).to_string())
    if isinstance(pa["corr"], pd.DataFrame) and not pa["corr"].empty:
        print("\n-- holdings correlation --"); print(pa["corr"].round(2).to_string())
    if pa["attribution"]:
        print("\n-- FF5+Mom exposure of live portfolio --"); print(pa["attribution"])
    if ledger:
        print("\n" + line); print("ALPHA LEDGER  (closed trades: pnl = beta$ + alpha$)"); print(line)
        print(ledger["trades"].to_string(index=False))
        print("\naggregate:", json.dumps(ledger["aggregate"], indent=2))
    if contracts:
        pend = [c["id"] for c in contracts.get("contracts", [])
                if "PENDING" in c.get("status", "") or "VIOLATION" in c.get("status", "")]
        if pend:
            print("\n!! contracts needing attention:", ", ".join(pend))

def main():
    bridge = load_bridge()
    if not bridge.get("portfolio"):
        sys.exit("bridge/portfolio.json not found — export it from Claude (Tier 1) first.")
    ledger_only = "--ledger-only" in sys.argv
    ledger = alpha_ledger(bridge)
    if ledger_only:
        print_report({"total_value": 0, "cash_weight": 0, "rf": risk_free_rate(),
                      "port_beta_incl_cash": float("nan"), "port_beta_ex_cash": float("nan"),
                      "theme_weights": pd.Series(dtype=float), "positions": pd.DataFrame(),
                      "mcr": pd.Series(dtype=float), "corr": pd.DataFrame(),
                      "attribution": None} if ledger else {}, ledger, bridge.get("contracts"))
        return
    pa = portfolio_analytics(bridge)
    print_report(pa, ledger, bridge.get("contracts"))
    out = build_portfolio_html(pa, ledger, bridge.get("contracts"))
    print(f"\nHTML report written: {out}")
    if "--no-open" not in sys.argv:
        import webbrowser
        webbrowser.open("file://" + out.replace(os.sep, "/"))

if __name__ == "__main__":
    main()
