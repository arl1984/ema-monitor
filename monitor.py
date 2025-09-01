import os, sys, json
import argparse
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from dateutil import tz
import yaml
from typing import List, Dict, Tuple, Optional

# ---------- Verbose logging ----------
VERBOSE = os.getenv("VERBOSE", "0") == "1"
def log(*a, **k):
    if VERBOSE:
        print(*a, **k, flush=True)

# ---------- Env / endpoints ----------
ALPACA_KEY    = os.environ.get("ALPACA_KEY_ID", "")
ALPACA_SECRET = os.environ.get("ALPACA_SECRET_KEY", "")
ALPACA_ENV    = os.environ.get("ALPACA_ENV", "paper").lower()  # 'paper' or 'live'

BASE_DATA   = "https://data.alpaca.markets/v2"
BASE_BROKER = "https://paper-api.alpaca.markets/v2" if ALPACA_ENV == "paper" else "https://api.alpaca.markets/v2"

HEADERS = {
    "APCA-API-KEY-ID": ALPACA_KEY,
    "APCA-API-SECRET-KEY": ALPACA_SECRET
}

CENTRAL_TZ = tz.gettz("America/Chicago")
EASTERN_TZ = tz.gettz("America/New_York")

# ---------- Helpers ----------
def utc_now() -> pd.Timestamp:
    ts = pd.Timestamp.utcnow()
    return ts if ts.tzinfo is not None else ts.tz_localize("UTC")

def load_cfg():
    with open("config.yaml","r") as f:
        return yaml.safe_load(f)

def save_json(path: str, obj: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def load_json(path: str) -> Optional[dict]:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None

def rsi(series: pd.Series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100 - (100/(1+rs))
    return out.fillna(50)

def ema(series: pd.Series, span:int):
    return series.ewm(span=span, adjust=False).mean()

def adx_df(df: pd.DataFrame, period=14) -> pd.Series:
    h,l,c = df["h"], df["l"], df["c"]
    up = h.diff()
    dn = -l.diff()
    plusDM  = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    minusDM = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    tr1 = (h - l)
    tr2 = (h - c.shift()).abs()
    tr3 = (l - c.shift()).abs()
    tr  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    pdi = 100 * plusDM.ewm(alpha=1/period, adjust=False).mean() / atr.replace(0, np.nan)
    mdi = 100 * minusDM.ewm(alpha=1/period, adjust=False).mean() / atr.replace(0, np.nan)
    dx  = ((pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)) * 100
    return dx.ewm(alpha=1/period, adjust=False).mean()

def vwap_series(df: pd.DataFrame) -> pd.Series:
    # Typical price * volume cumulative / cumulative volume
    tp = (df["h"] + df["l"] + df["c"]) / 3.0
    pv = tp * df["v"]
    cum_pv = pv.cumsum()
    cum_v  = df["v"].cumsum()
    return cum_pv / cum_v.replace(0, np.nan)

def is_rth(ts_utc: pd.Timestamp, start="09:30", end="16:00", tzname="America/New_York"):
    et = ts_utc.tz_convert(tz.gettz(tzname))
    sh,sm = map(int, start.split(":"))
    eh,em = map(int, end.split(":"))
    t = et.time()
    return (t >= datetime(et.year,et.month,et.day,sh,sm).time() and
            t <= datetime(et.year,et.month,et.day,eh,em).time())

def today_et(dt_utc: pd.Timestamp) -> datetime.date:
    return dt_utc.tz_convert(EASTERN_TZ).date()

def get_et_midnight_bounds(now_utc: pd.Timestamp) -> Tuple[pd.Timestamp, pd.Timestamp]:
    et_dt = now_utc.tz_convert(EASTERN_TZ)
    start_et = et_dt.replace(hour=0, minute=0, second=0, microsecond=0)
    start_utc = pd.Timestamp(start_et).tz_convert(timezone.utc)
    return start_utc, now_utc

# ---------- Alpaca fetch ----------
def get_nyse_symbols(limit_universe:int) -> List[str]:
    """Pull active US equities and filter to NYSE via Alpaca /assets."""
    try:
        r = requests.get(f"{BASE_BROKER}/assets", headers=HEADERS, timeout=30)
        if r.status_code in (401,403):
            raise RuntimeError(
                f"Alpaca auth failed ({r.status_code}). "
                f"Check ALPACA_KEY_ID/ALPACA_SECRET_KEY and ALPACA_ENV='{ALPACA_ENV}'."
            )
        r.raise_for_status()
    except Exception as e:
        print("Error calling /v2/assets:", e, file=sys.stderr)
        raise
    assets = r.json()
    nyse = [a["symbol"] for a in assets
            if a.get("class")=="us_equity" and a.get("status")=="active" and a.get("exchange")=="NYSE"]
    out = sorted(set(nyse))
    if limit_universe and limit_universe > 0:
        return out[:max(limit_universe, 1)]
    return out

def fetch_bars_chunk(symbols: List[str], start_iso: str, end_iso: str, timeframe: str, feed: str) -> pd.DataFrame:
    if not symbols: 
        return pd.DataFrame()
    syms = ",".join(symbols)
    params = {
        "symbols": syms,
        "timeframe": timeframe,
        "start": start_iso,
        "end": end_iso,
        "limit": 10000,
        "feed": feed,
        "adjustment": "all",
        "sort": "asc"
    }
    r = requests.get(f"{BASE_DATA}/stocks/bars", headers=HEADERS, params=params, timeout=60)
    r.raise_for_status()
    data = r.json().get("bars", {})
    rows = []
    for sym, bars in data.items():
        for b in bars:
            rows.append({
                "symbol": sym,
                "t": pd.Timestamp(b["t"]), "o": b["o"], "h": b["h"],
                "l": b["l"], "c": b["c"], "v": b["v"]
            })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values(["symbol","t"]).reset_index(drop=True)
    df["t"] = pd.to_datetime(df["t"], utc=True)
    return df

def fetch_daily_bars(symbols: List[str], start_iso: str, end_iso: str, feed: str) -> pd.DataFrame:
    # timeframe 1Day for RVOL
    return fetch_bars_chunk(symbols, start_iso, end_iso, "1Day", feed)

# ---------- Universe ranking ----------
def rank_by_today_dollar_vol(symbols: List[str], tf: str, feed: str, now_utc: pd.Timestamp, top_n: int) -> pd.DataFrame:
    start_utc, end_utc = get_et_midnight_bounds(now_utc)
    start_iso, end_iso = start_utc.isoformat(), end_utc.isoformat()

    rows = []
    chunk = 50
    for i in range(0, len(symbols), chunk):
        part = symbols[i:i+chunk]
        bars = fetch_bars_chunk(part, start_iso, end_iso, tf, feed)
        log(f"[rank$] chunk symbols={len(part)} rows={0 if bars is None else len(bars)}")
        if bars.empty:
            continue
        g = bars.groupby("symbol")
        for sym, df in g:
            dollar_vol = float((df["c"] * df["v"]).sum())
            vol = float(df["v"].sum())
            rows.append({"symbol": sym, "dollar_vol_today": dollar_vol, "shares_today": vol})
    if not rows:
        return pd.DataFrame(columns=["symbol","dollar_vol_today","shares_today"])
    ranked = pd.DataFrame(rows).sort_values("dollar_vol_today", ascending=False).head(top_n)
    return ranked

def compute_rvol_for(symbols: List[str], feed: str, now_utc: pd.Timestamp, lookback_days=30) -> Dict[str, float]:
    # Fetch ~ (lookback_days+1) days of 1D bars; compute today / median(hist)
    end_utc = now_utc
    start_utc = now_utc - timedelta(days=lookback_days+10)
    daily = pd.DataFrame()
    chunk = 50
    for i in range(0, len(symbols), chunk):
        part = symbols[i:i+chunk]
        df = fetch_daily_bars(part, start_utc.isoformat(), end_utc.isoformat(), feed)
        log(f"[rvol] symbols={len(part)} rows={0 if df is None else len(df)}")
        if not df.empty:
            daily = pd.concat([daily, df], ignore_index=True)
    rvol_map: Dict[str, float] = {}
    if daily.empty:
        return {s: 1.0 for s in symbols}
    daily["date_et"] = daily["t"].dt.tz_convert(EASTERN_TZ).dt.date
    for sym, g in daily.groupby("symbol"):
        dvol = g.groupby("date_et")["v"].sum().sort_index()
        if len(dvol) < 5:
            rvol_map[sym] = 1.0
            continue
        today_vol = dvol.iloc[-1]
        hist = dvol.iloc[-(lookback_days+1):-1] if len(dvol) > (lookback_days+1) else dvol.iloc[:-1]
        med = float(hist.median()) if len(hist) else np.nan
        rvol_map[sym] = float(today_vol / med) if med and med > 0 else 1.0
    return rvol_map

# ---------- Technicals / signals ----------
def enrich_with_indicators(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    df = df.sort_values("t").copy()
    df["ema_fast"] = ema(df["c"], cfg["ema_fast"])
    df["ema_slow"] = ema(df["c"], cfg["ema_slow"])
    df["rsi"] = rsi(df["c"], cfg["rsi_period"])
    df["vol_ma20"] = df["v"].rolling(20, min_periods=1).mean()
    df["adx"] = adx_df(df, period=cfg.get("adx_period",14))
    # spread sign for crosses
    spread = df["ema_fast"] - df["ema_slow"]
    sign = np.sign(spread)
    df["bull_cross"] = (sign.shift(1) <= 0) & (sign > 0)
    df["bear_cross"] = (sign.shift(1) >= 0) & (sign < 0)
    return df

def session_vwap(df: pd.DataFrame, now_utc: pd.Timestamp) -> pd.Series:
    # compute VWAP only for today's ET bars
    today = today_et(now_utc)
    dft = df[df["t"].dt.tz_convert(EASTERN_TZ).dt.date == today]
    if dft.empty:
        return pd.Series(index=df.index, dtype=float)
    vwap = vwap_series(dft)
    # reindex back to full df, forward-fill so last today's bar has correct vwap
    vw = vwap.reindex(df.index, method="ffill")
    return vw

def slope_positive(series: pd.Series) -> bool:
    if len(series) < 2: return False
    return bool(series.iloc[-1] > series.iloc[-2])

def fetch_confirm_slope(symbols: List[str], timeframe: str, feed: str, now_utc: pd.Timestamp, ema_len: int) -> Dict[str, Tuple[bool,bool]]:
    # returns {sym: (ema_slow_slope_up, ema_slow_slope_down)}
    start_utc = now_utc - timedelta(days=10)  # enough bars for slope
    df = pd.DataFrame()
    chunk = 50
    for i in range(0, len(symbols), chunk):
        part = symbols[i:i+chunk]
        bars = fetch_bars_chunk(part, start_utc.isoformat(), now_utc.isoformat(), timeframe, feed)
        log(f"[confirm] symbols={len(part)} rows={0 if bars is None else len(bars)}")
        if not bars.empty:
            df = pd.concat([df, bars], ignore_index=True)
    out = {}
    if df.empty:
        return {s: (True, True) for s in symbols}  # neutral if no data
    for sym, g in df.groupby("symbol"):
        g = g.sort_values("t")
        ema_slow = ema(g["c"], ema_len)
        up = slope_positive(ema_slow)
        down = not up and len(ema_slow) >= 2 and (ema_slow.iloc[-1] < ema_slow.iloc[-2])
        out[sym] = (up, down)
    return out

def composite_score(row, rvol: float, is_bull: bool, cfg: dict, ema50=None, ema200=None, vwap=None) -> float:
    score = 0.0
    # RVOL contribution (cap at +2)
    score += min(max((rvol - 1.0), 0.0), 2.0)
    # ADX gate / contribution
    if row.get("adx", np.nan) >= cfg["adx_min"]:
        score += 1.0
    # Structure: EMA alignment + VWAP side
    if ema50 is not None and ema200 is not None:
        if is_bull and (row["c"] > ema50 > ema200):
            score += 0.5
        if (not is_bull) and (row["c"] < ema50 < ema200):
            score += 0.5
    if vwap is not None:
        if is_bull and (row["c"] > vwap):
            score += 0.5
        if (not is_bull) and (row["c"] < vwap):
            score += 0.5
    # Momentum proxy (normalized): use last close change vs previous, scaled by ATR-ish volatility via rolling TR
    # Simple, light-weight proxy:
    score += 0.25 if (row["ema_fast"] - row["ema_slow"]) / max(abs(row["ema_slow"]), 1e-9) > 0.0025 else 0.0
    return float(score)

def load_last_run_time(now_utc: pd.Timestamp, default_hours: int) -> pd.Timestamp:
    meta = load_json("data/last_run.json")
    if meta and "run_time_utc" in meta:
        try:
            ts = pd.Timestamp(meta["run_time_utc"])
            return ts if ts.tzinfo else ts.tz_localize("UTC")
        except Exception:
            pass
    return now_utc - timedelta(hours=default_hours)

def read_signals_csv(path="data/signals.csv") -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame(columns=["ticker","last_price","timestamp","direction","session","id"])

def make_signal_id(ticker:str, direction:str, iso_ts:str) -> str:
    return f"{ticker}|{direction}|{iso_ts}"

# ---------- Webhook ----------
def send_webhook(cfg, payload_text: Optional[str]=None, payload_obj: Optional[dict]=None):
    url = cfg.get("webhook_url","")
    if not url:
        return
    typ = cfg.get("webhook_type","generic").lower()
    try:
        if typ == "slack":
            if payload_text:
                requests.post(url, json={"text": payload_text}, timeout=10)
            elif payload_obj:
                requests.post(url, json=payload_obj, timeout=10)
        elif typ == "discord":
            if payload_text:
                requests.post(url, json={"content": payload_text}, timeout=10)
            elif payload_obj:
                # Discord webhooks accept embeds/content—keep it simple
                requests.post(url, json=payload_obj, timeout=10)
        else:
            # generic JSON
            obj = payload_obj if payload_obj else {"text": payload_text}
            requests.post(url, json=obj, timeout=10)
    except Exception as e:
        print("Webhook error:", e, file=sys.stderr)

# ---------- Core pipeline ----------
def run_monitor(cfg, override_tickers: list | None = None, dry_run: bool = False):
    now_utc = utc_now()
    # allow env to override secrets
    cfg["webhook_url"]  = os.getenv("WEBHOOK_URL",  cfg.get("webhook_url",""))
    cfg["webhook_type"] = os.getenv("WEBHOOK_TYPE", cfg.get("webhook_type","discord"))

    log(f"[env] ALPACA_ENV={ALPACA_ENV}  feed={cfg['feed']}  timeframe={cfg['timeframe']}  now_utc={now_utc}")

    # Universe selection
    if override_tickers:
        universe = [t.strip().upper() for t in override_tickers if t.strip()]
        log(f"[smoke] overriding universe: {universe}")
    else:
        all_nyse = get_nyse_symbols(cfg["universe_limit"]*3)
        log(f"[universe] active NYSE symbols from Alpaca: {len(all_nyse)} (sample: {all_nyse[:10]})")
        ranked = rank_by_today_dollar_vol(all_nyse, cfg["timeframe"], cfg["feed"], now_utc, cfg["universe_top_n"])
        universe = ranked["symbol"].tolist()
        # RVOL filter (compute on top bucket to save quota)
        rvol_map = compute_rvol_for(universe, cfg["feed"], now_utc, lookback_days=cfg["rvol_lookback_days"])
        universe = [s for s in universe if rvol_map.get(s,1.0) >= cfg["rvol_min"]]
        log(f"[universe] after RVOL≥{cfg['rvol_min']}: {len(universe)}")

    # Fetch working timeframe bars for last 7 days
    start_utc = now_utc - timedelta(days=7)
    allbars = pd.DataFrame()
    chunk = 50
    for i in range(0, len(universe), chunk):
        part = universe[i:i+chunk]
        bars = fetch_bars_chunk(part, start_utc.isoformat(), now_utc.isoformat(), cfg["timeframe"], cfg["feed"])
        log(f"[bars] symbols={len(part)} rows={0 if bars is None else len(bars)}")
        if not bars.empty:
            allbars = pd.concat([allbars, bars], ignore_index=True)

    if allbars.empty:
        meta = {"run_time_utc": now_utc.isoformat(), "universe_size": len(universe), "signals_count": 0, "sample_symbols": universe[:10]}
        print(meta)
        return

    # Enrich per-symbol with indicators and session VWAP
    out_signals = []
    rankings_bull, rankings_bear = [], []
    last_run_cutoff = load_last_run_time(now_utc, cfg["signal_lookback_hours"])
    prev_signals_df = read_signals_csv(cfg.get("signals_csv","data/signals.csv"))
    dedupe_days = cfg.get("dedupe_window_days", 5)
    cutoff_dedupe = now_utc - timedelta(days=dedupe_days)
    if not prev_signals_df.empty and "timestamp" in prev_signals_df.columns:
        prev_signals_df["ts"] = pd.to_datetime(prev_signals_df["timestamp"], utc=True, errors="coerce")
        prev_signals_df = prev_signals_df[prev_signals_df["ts"] >= cutoff_dedupe]

    # Multi-timeframe confirm
    confirm_tf = cfg.get("confirm_timeframe", "")
    ema_slow_len = cfg["ema_slow"]
    confirm_map = {}
    if confirm_tf:
        confirm_map = fetch_confirm_slope(universe, confirm_tf, cfg["feed"], now_utc, ema_slow_len)

    # Precompute RVOL again for final universe if we didn't have it above (override provided)
    if override_tickers:
        rvol_map = compute_rvol_for(universe, cfg["feed"], now_utc, lookback_days=cfg["rvol_lookback_days"])
    # Per-symbol loop
    for sym, g in allbars.groupby("symbol"):
        g = enrich_with_indicators(g, cfg)
        # EMA 50/200 for structure
        g["ema50"] = ema(g["c"], 50)
        g["ema200"] = ema(g["c"], 200)
        # Session VWAP
        g["sess_vwap"] = session_vwap(g, now_utc)
        if len(g) < cfg["ema_slow"]+1:
            continue

        # Multi-timeframe slope confirm (if configured)
        slope_up, slope_down = confirm_map.get(sym, (True, True))

        # Cross detection within lookback window
        recent = g[g["t"] >= last_run_cutoff]
        if recent.empty:
            continue

        # Evaluate each crossing bar
        crosses = recent[(recent["bull_cross"]) | (recent["bear_cross"])]
        for _, row in crosses.iterrows():
            is_bull = bool(row["bull_cross"])
            # Gates
            if row["c"] <= cfg["price_min"]:
                continue
            if row["v"] < cfg["volume_mult"] * row["vol_ma20"]:
                continue
            if is_bull and row["rsi"] <= 50:
                continue
            if (not is_bull) and row["rsi"] >= 50:
                continue
            if row["adx"] < cfg["adx_min"]:
                continue
            # Confirm TF slope
            if confirm_tf:
                if is_bull and not slope_up: 
                    continue
                if (not is_bull) and not slope_down:
                    continue
            # RVOL gate
            rvol = float(rvol_map.get(sym, 1.0))
            if rvol < cfg["rvol_min"]:
                continue

            # Composite score at the cross bar
            score = composite_score(
                row,
                rvol=rvol,
                is_bull=is_bull,
                cfg=cfg,
                ema50=row.get("ema50"),
                ema200=row.get("ema200"),
                vwap=row.get("sess_vwap")
            )
            if score < cfg.get("min_score", 2.0):
                continue

            # Dedupe
            ts_iso_ct = row["t"].tz_convert(CENTRAL_TZ).isoformat()
            sig_id = make_signal_id(sym, "bullish" if is_bull else "bearish", ts_iso_ct)
            if not prev_signals_df.empty and "id" in prev_signals_df.columns:
                if (prev_signals_df["id"] == sig_id).any():
                    continue

            session = "regular" if is_rth(
                row["t"],
                cfg["regular_hours"]["start"],
                cfg["regular_hours"]["end"],
                cfg["regular_hours"]["tz"]
            ) else "extended"

            out_signals.append({
                "ticker": sym,
                "last_price": round(float(row["c"]), 4),
                "timestamp": ts_iso_ct,
                "direction": "bullish" if is_bull else "bearish",
                "session": session,
                "id": sig_id,
                "score": round(score, 3),
                "rvol": round(rvol, 2),
                "adx": round(float(row["adx"]), 2)
            })

        # Build ranking snapshot at the LAST bar (no cross required)
        last = g.iloc[-1]
        if last["c"] <= cfg["price_min"]:
            continue
        rvol_last = float(rvol_map.get(sym, 1.0))
        if rvol_last < cfg["rvol_min"]:
            continue
        # Build score for bull/bear contexts
        bull_ctx = (last["ema_fast"] > last["ema_slow"]) and (last["rsi"] > 50)
        bear_ctx = (last["ema_fast"] < last["ema_slow"]) and (last["rsi"] < 50)
        # Confirm TF slope
        if confirm_tf:
            bull_ctx = bull_ctx and slope_up
            bear_ctx = bear_ctx and slope_down

        if bull_ctx:
            sc = composite_score(last, rvol_last, True, cfg, last.get("ema50"), last.get("ema200"), last.get("sess_vwap"))
            if sc >= cfg.get("ranking_min_score", 1.75):
                rankings_bull.append((sym, float(last["c"]), float(sc), float(rvol_last), float(last.get("adx", np.nan))))
        if bear_ctx:
            sc = composite_score(last, rvol_last, False, cfg, last.get("ema50"), last.get("ema200"), last.get("sess_vwap"))
            if sc >= cfg.get("ranking_min_score", 1.75):
                rankings_bear.append((sym, float(last["c"]), float(sc), float(rvol_last), float(last.get("adx", np.nan))))

    # Sort rankings
    rankings_bull.sort(key=lambda x: (-x[2], -x[3]))  # score desc, then rvol desc
    rankings_bear.sort(key=lambda x: (-x[2], -x[3]))

    # Limit outgoing alerts
    max_alerts = cfg.get("max_alerts_per_run", 25)
    out_signals = sorted(out_signals, key=lambda s: -s["score"])[:max_alerts]

    log(f"[signals] total={len(out_signals)}  rankings: bull={len(rankings_bull)} bear={len(rankings_bear)}")

    meta = {
        "run_time_utc": now_utc.isoformat(),
        "universe_size": len(universe),
        "signals_count": len(out_signals),
        "sample_symbols": universe[:10]
    }

    # Output / side effects
    if dry_run:
        print({
            **meta,
            "top_bull": rankings_bull[:10],
            "top_bear": rankings_bear[:10],
        })
        return

    # Send new cross signals
    if out_signals:
        # CSV append with header detection + IDs
        os.makedirs(os.path.dirname(cfg.get("signals_csv","data/signals.csv")), exist_ok=True)
        df = pd.DataFrame(out_signals)
        csv_path = cfg.get("signals_csv","data/signals.csv")
        if os.path.exists(csv_path):
            df.to_csv(csv_path, mode="a", header=False, index=False)
        else:
            df.to_csv(csv_path, index=False)

        # Webhook text
        def fmt(s):
            return f"**{s['ticker']}** {s['direction'].upper()} @ ${s['last_price']} — {s['timestamp']} ({s['session']}) | score {s['score']} | RVOL {s['rvol']} | ADX {s['adx']}"
        text = "\n".join(fmt(s) for s in out_signals)
        send_webhook(cfg, payload_text=f"📈 NYSE 20/50 EMA crosses (since last run)\n{text}")

    # Ranking snapshot (optional)
    rank_cfg = cfg.get("ranking_alert", {"enabled": False})
    if rank_cfg and rank_cfg.get("enabled", False):
        tb = rankings_bull[: rank_cfg.get("top_n", 10)]
        trb = "\n".join([f"{i+1}. **{t}** ${p:.2f} | score {s:.2f} | RVOL {rv:.2f} | ADX {ax:.1f}"
                         for i,(t,p,s,rv,ax) in enumerate(tb)])
        te = rankings_bear[: rank_cfg.get("top_n", 10)]
        tre = "\n".join([f"{i+1}. **{t}** ${p:.2f} | score {s:.2f} | RVOL {rv:.2f} | ADX {ax:.1f}"
                         for i,(t,p,s,rv,ax) in enumerate(te)])
        msg = "🏁 **Top Trending (composite score)**\n**BULLISH**\n" + (trb or "_none_") + "\n\n**BEARISH**\n" + (tre or "_none_")
        send_webhook(cfg, payload_text=msg)

    # Save meta / last run
    os.makedirs("data", exist_ok=True)
    save_json("data/last_run.json", meta)
    print(meta)

# ---------- CLI ----------
def main():
    if not (ALPACA_KEY and ALPACA_SECRET):
        print("Missing ALPACA_KEY_ID / ALPACA_SECRET_KEY")
        sys.exit(1)

    cfg = load_cfg()
    parser = argparse.ArgumentParser(description="NYSE trend monitor")
    parser.add_argument("--tickers", type=str, default="", help="Comma-separated tickers to override universe (smoke test)")
    parser.add_argument("--dry-run", action="store_true", help="Run everything but do not send webhook or write CSV")
    args = parser.parse_args()

    override = [t for t in args.tickers.split(",")] if args.tickers else None
    run_monitor(cfg, override_tickers=override, dry_run=args.dry_run)

if __name__ == "__main__":
    main()
