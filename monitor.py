import os, sys, json, time, random
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

# ---------- Telemetry (populated per run) ----------
METRICS = {
    "http_gets": 0,
    "pages_fetched": 0,
    "symbols_requested": 0,
    "symbols_with_bars": 0,
    "bars_rows": 0,
    "universe_size": 0,
    "run_started_utc": None,
    "run_finished_utc": None,
    "rate_limit_remaining": None,
    "rate_limit_reset": None,
    "errors": []
}

# ---------- HTTP w/ retries ----------
def http_get(url: str, params: Optional[dict] = None, headers: Optional[dict] = None,
             timeout: int = 60, max_retries: int = 3) -> requests.Response:
    attempt = 0
    last_exc = None
    while attempt <= max_retries:
        try:
            r = requests.get(url, headers=headers, params=params, timeout=timeout)
            METRICS["http_gets"] += 1
            METRICS["rate_limit_remaining"] = r.headers.get("X-RateLimit-Remaining", METRICS["rate_limit_remaining"])
            METRICS["rate_limit_reset"] = r.headers.get("X-RateLimit-Reset", METRICS["rate_limit_reset"])
            if r.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"Retryable status {r.status_code}", response=r)
            r.raise_for_status()
            return r
        except Exception as e:
            last_exc = e
            attempt += 1
            if attempt > max_retries:
                break
            sleep_s = (2 ** (attempt - 1)) + random.uniform(0, 0.5)
            log(f"[retry] {url} attempt={attempt}/{max_retries} sleeping {sleep_s:.2f}s reason={e}")
            time.sleep(sleep_s)
    METRICS["errors"].append(str(last_exc))
    raise last_exc if last_exc else RuntimeError("HTTP GET failed unexpectedly")

# ---------- File lock ----------
class FileLock:
    def __init__(self, path: str):
        self.lock_path = path + ".lock"
        self.fh = None
        self._is_windows = os.name == "nt"
    def __enter__(self):
        os.makedirs(os.path.dirname(self.lock_path), exist_ok=True)
        self.fh = open(self.lock_path, "a+")
        try:
            if self._is_windows:
                import msvcrt
                msvcrt.locking(self.fh.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl
                fcntl.flock(self.fh.fileno(), fcntl.LOCK_EX)
        except Exception as e:
            log(f"[lock] non-fatal: {e}")
        return self
    def __exit__(self, exc_type, exc, tb):
        try:
            if self.fh:
                if self._is_windows:
                    import msvcrt
                    try: msvcrt.locking(self.fh.fileno(), msvcrt.LK_UNLCK, 1)
                    except Exception: pass
                else:
                    import fcntl
                    try: fcntl.flock(self.fh.fileno(), fcntl.LOCK_UN)
                    except Exception: pass
                self.fh.close()
        except Exception:
            pass

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
    tp = (df["h"] + df["l"] + df["c"]) / 3.0
    pv = tp * df["v"]
    return pv.cumsum() / df["v"].cumsum().replace(0, np.nan)

def is_rth(ts_utc: pd.Timestamp, start="09:30", end="16:00", tzname="America/New_York"):
    et = ts_utc.tz_convert(tz.gettz(tzname))
    sh,sm = map(int, start.split(":")); eh,em = map(int, end.split(":"))
    t = et.time()
    return (t >= datetime(et.year,et.month,et.day,sh,sm).time() and
            t <= datetime(et.year,et.month,et.day,eh,em).time())

def today_et(dt_utc: pd.Timestamp) -> datetime.date:
    return dt_utc.tz_convert(EASTERN_TZ).date()

def session_open_utc(now_utc: pd.Timestamp, start="09:30", tzname="America/New_York") -> pd.Timestamp:
    et = now_utc.tz_convert(tz.gettz(tzname))
    sh, sm = map(int, start.split(":"))
    so = et.replace(hour=sh, minute=sm, second=0, microsecond=0)
    if et.time() < so.time():
        so = so - pd.Timedelta(days=1)
    return pd.Timestamp(so).tz_convert("UTC")

def get_et_midnight_bounds(now_utc: pd.Timestamp) -> Tuple[pd.Timestamp, pd.Timestamp]:
    et_dt = now_utc.tz_convert(EASTERN_TZ)
    start_et = et_dt.replace(hour=0, minute=0, second=0, microsecond=0)
    start_utc = pd.Timestamp(start_et).tz_convert(timezone.utc)
    return start_utc, now_utc

# ---------- Alpaca fetch ----------
def get_us_equities(limit_universe:int, exchanges: List[str]) -> List[str]:
    try:
        r = http_get(f"{BASE_BROKER}/assets", headers=HEADERS, timeout=30)
    except Exception as e:
        print("Error calling /v2/assets:", e, file=sys.stderr)
        raise
    assets = r.json()
    exchanges_set = set(exchanges) if exchanges else {"NYSE","NASDAQ","AMEX"}
    us = [a["symbol"] for a in assets
          if a.get("class")=="us_equity"
          and a.get("status")=="active"
          and a.get("exchange") in exchanges_set]
    out = sorted(set(us))
    if limit_universe and limit_universe > 0:
        return out[:max(limit_universe, 1)]
    return out

def fetch_all_pages(url, headers, params, timeout=60):
    all_rows = []
    page_token = None
    while True:
        q = dict(params)
        if page_token:
            q["page_token"] = page_token
        r = http_get(url, headers=headers, params=q, timeout=timeout)
        j = r.json()
        data = j.get("bars", {})
        page_rows = 0
        for sym, bars in data.items():
            for b in bars:
                all_rows.append({
                    "symbol": sym,
                    "t": pd.Timestamp(b["t"]),
                    "o": b["o"], "h": b["h"], "l": b["l"], "c": b["c"], "v": b["v"]
                })
                page_rows += 1
        METRICS["pages_fetched"] += 1
        page_token = j.get("next_page_token")
        if not page_token:
            break
    return all_rows

def fetch_bars_chunk(symbols: List[str], start_iso: str, end_iso: str, timeframe: str, feed: str) -> pd.DataFrame:
    if not symbols:
        return pd.DataFrame()
    params = {
        "symbols": ",".join(symbols),
        "timeframe": timeframe,
        "start": start_iso,
        "end": end_iso,
        "limit": 10000,
        "feed": feed,
        "adjustment": "all",
        "sort": "asc"
    }
    rows = fetch_all_pages(f"{BASE_DATA}/stocks/bars", HEADERS, params, timeout=60)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values(["symbol","t"]).reset_index(drop=True)
    df["t"] = pd.to_datetime(df["t"], utc=True)
    return df

def fetch_daily_bars(symbols: List[str], start_iso: str, end_iso: str, feed: str) -> pd.DataFrame:
    return fetch_bars_chunk(symbols, start_iso, end_iso, "1Day", feed)

# ---------- Universe ranking ----------
def rank_by_today_dollar_vol(symbols: List[str], tf: str, feed: str, now_utc: pd.Timestamp, top_n: int) -> pd.DataFrame:
    start_utc, end_utc = get_et_midnight_bounds(now_utc)
    rows = []
    chunk = 50
    for i in range(0, len(symbols), chunk):
        part = symbols[i:i+chunk]
        bars = fetch_bars_chunk(part, start_utc.isoformat(), end_utc.isoformat(), tf, feed)
        log(f"[rank$] chunk symbols={len(part)} rows={0 if bars is None else len(bars)}")
        if bars.empty: continue
        g = bars.groupby("symbol")
        for sym, df in g:
            dollar_vol = float((df["c"] * df["v"]).sum())
            vol = float(df["v"].sum())
            rows.append({"symbol": sym, "dollar_vol_today": dollar_vol, "shares_today": vol})
    if not rows:
        return pd.DataFrame(columns=["symbol","dollar_vol_today","shares_today"])
    ranked = pd.DataFrame(rows).sort_values("dollar_vol_today", ascending=False).head(top_n)
    return ranked

def compute_rvol_for(symbols: List[str], feed: str, now_utc: pd.Timestamp, lookback_days=30, use_same_weekday=True) -> Dict[str, float]:
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
    daily["weekday"] = pd.to_datetime(daily["date_et"]).dt.weekday
    for sym, g in daily.groupby("symbol"):
        dvol = g.groupby("date_et")["v"].sum().sort_index()
        if len(dvol) < 5:
            rvol_map[sym] = 1.0
            continue
        today_date = dvol.index[-1]
        today_vol = dvol.iloc[-1]
        hist = dvol.iloc[:-1]
        if use_same_weekday:
            wd = pd.Timestamp(today_date).weekday()
            hist_idx = [d for d in hist.index if pd.Timestamp(d).weekday() == wd]
            hist2 = hist.loc[hist_idx] if len(hist_idx) >= max(5, min(lookback_days//4, 10)) else hist
        else:
            hist2 = hist
        med = float(hist2.tail(lookback_days).median()) if len(hist2) else np.nan
        rvol_map[sym] = float(today_vol / med) if med and med > 0 else 1.0
    return rvol_map

# ---------- Technicals / indicators ----------
def enrich_with_indicators(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    df = df.sort_values("t").copy()

    # EMAs & spread
    df["ema_fast"] = ema(df["c"], cfg["ema_fast"])
    df["ema_slow"] = ema(df["c"], cfg["ema_slow"])
    df["spread"] = df["ema_fast"] - df["ema_slow"]
    df["spread_pct"] = df["spread"] / df["ema_slow"].replace(0, np.nan)
    df["spread_roc"] = df["spread"].diff()

    # RSI/ADX
    df["rsi"] = rsi(df["c"], cfg["rsi_period"])
    df["adx"] = adx_df(df, period=cfg.get("adx_period",14))

    # Volume stats
    df["vol_ma20"] = df["v"].rolling(20, min_periods=1).mean()

    # Bollinger for squeeze
    bb_len = int(cfg.get("bb_period", 20))
    bb_k = float(cfg.get("bb_k", 2.0))
    mid = df["c"].rolling(bb_len, min_periods=1).mean()
    std = df["c"].rolling(bb_len, min_periods=1).std()
    df["bb_mid"] = mid
    df["bb_up"]  = mid + bb_k*std
    df["bb_dn"]  = mid - bb_k*std
    df["bb_width"] = (df["bb_up"] - df["bb_dn"]) / df["bb_mid"].replace(0, np.nan)

    # MACD (classic 12/26/9 on close)
    macd_fast = int(cfg.get("macd_fast", 12))
    macd_slow = int(cfg.get("macd_slow", 26))
    macd_sig  = int(cfg.get("macd_signal", 9))
    df["ema_f_macd"] = ema(df["c"], macd_fast)
    df["ema_s_macd"] = ema(df["c"], macd_slow)
    df["macd"] = df["ema_f_macd"] - df["ema_s_macd"]
    df["macd_signal"] = ema(df["macd"], macd_sig)
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # Cross flags (20/50)
    sign = np.sign(df["spread"])
    df["bull_cross"] = (sign.shift(1) <= 0) & (sign > 0)
    df["bear_cross"] = (sign.shift(1) >= 0) & (sign < 0)

    return df

def session_vwap(df: pd.DataFrame, now_utc: pd.Timestamp) -> pd.Series:
    today = today_et(now_utc)
    dft = df[df["t"].dt.tz_convert(EASTERN_TZ).dt.date == today]
    if dft.empty:
        return pd.Series(index=df.index, dtype=float)
    vwap = vwap_series(dft)
    return vwap.reindex(df.index, method="ffill")

def slope_positive(series: pd.Series) -> bool:
    if len(series) < 2: return False
    return bool(series.iloc[-1] > series.iloc[-2])

def fetch_confirm_slope(symbols: List[str], timeframe: str, feed: str, now_utc: pd.Timestamp, ema_len: int) -> Dict[str, Tuple[bool,bool]]:
    start_utc = now_utc - timedelta(days=10)
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
        return {s: (True, True) for s in symbols}
    for sym, g in df.groupby("symbol"):
        g = g.sort_values("t")
        ema_slow = ema(g["c"], ema_len)
        up = slope_positive(ema_slow)
        down = (not up) and len(ema_slow) >= 2 and (ema_slow.iloc[-1] < ema_slow.iloc[-2])
        out[sym] = (up, down)
    return out

# ---------- Scoring ----------
def composite_score(row, rvol: float, is_bull: bool, cfg: dict, ema50=None, ema200=None, vwap=None) -> float:
    score = 0.0
    adx_val = float(row.get("adx", np.nan))
    if not np.isfinite(adx_val) or adx_val < cfg["adx_min"]:
        return -1.0
    score += min(max((rvol - 1.0), 0.0), 2.0)   # 0..2 from RVOL
    score += 1.0                                 # ADX passed
    if ema50 is not None and ema200 is not None:
        if is_bull and (row["c"] > ema50 > ema200): score += 0.5
        if (not is_bull) and (row["c"] < ema50 < ema200): score += 0.5
    if vwap is not None and np.isfinite(vwap):
        if is_bull and (row["c"] > vwap): score += 0.5
        if (not is_bull) and (row["c"] < vwap): score += 0.5
    try:
        score += 0.25 if (row["ema_fast"] - row["ema_slow"]) / max(abs(row["ema_slow"]), 1e-9) > 0.0025 else 0.0
    except Exception:
        pass
    return float(score)

# ---------- IO helpers ----------
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
        return pd.DataFrame(columns=["ticker","last_price","timestamp","direction","session","id","score","rvol","adx","kind"])

def make_signal_id(ticker:str, direction:str, iso_ts:str) -> str:
    return f"{ticker}|{direction}|{iso_ts}"

# ---------- Webhook ----------
def send_webhook(cfg, payload_text: Optional[str]=None, payload_obj: Optional[dict]=None):
    url = cfg.get("webhook_url","")
    if not url: return
    typ = cfg.get("webhook_type","generic").lower()
    try:
        if typ == "slack":
            body = {"text": payload_text} if payload_text else (payload_obj or {})
            requests.post(url, json=body, timeout=10)
        elif typ == "discord":
            body = {"content": payload_text} if payload_text else (payload_obj or {})
            requests.post(url, json=body, timeout=10)
        else:
            body = payload_obj if payload_obj else {"text": payload_text}
            requests.post(url, json=body, timeout=10)
    except Exception as e:
        print("Webhook error:", e, file=sys.stderr)

# ---------- Health / stats ----------
def send_health_webhook(cfg, text: str, extra: Optional[dict]=None):
    payload = {"content": text} if cfg.get("webhook_type","discord") == "discord" else {"text": text}
    if extra: payload.update(extra)
    try: send_webhook(cfg, payload_text=text)
    except Exception as e: log(f"[health webhook] failed: {e}")

def summarize_stats_text(level: str, universe_size: int, signals_count: int) -> str:
    dur = None
    try:
        if METRICS["run_started_utc"] and METRICS["run_finished_utc"]:
            dur = (pd.Timestamp(METRICS["run_finished_utc"]) - pd.Timestamp(METRICS["run_started_utc"])).total_seconds()
    except Exception:
        pass
    coverage_pct = 100.0 * (METRICS["symbols_with_bars"] / max(universe_size,1)) if universe_size else 0.0
    header = f"🩺 Scan stats — analyzed={METRICS['symbols_with_bars']}/{universe_size} ({coverage_pct:.1f}%) | bars={METRICS['bars_rows']} | pages={METRICS['pages_fetched']} | http={METRICS['http_gets']} | signals={signals_count}"
    if dur is not None: header += f" | duration={dur:.1f}s"
    if level == "verbose":
        header += f"\nrate_limit_remaining={METRICS['rate_limit_remaining']} reset={METRICS['rate_limit_reset']} errors={len(METRICS['errors'])}"
    return header

def health_check(cfg, universe_size: int, symbols_with_bars: int) -> Tuple[bool, str]:
    h = cfg.get("health", {}) or {}
    if not h.get("enabled", True): return True, ""
    min_uni = h.get("min_universe", 50)
    min_sym_bars = h.get("min_symbols_with_bars", 30)
    min_cov = h.get("min_bar_coverage_pct", 60)
    if universe_size < min_uni:
        return False, f"⚠️ Degraded run: universe too small ({universe_size} < {min_uni})."
    coverage_pct = 100.0 * (symbols_with_bars / max(universe_size,1))
    if symbols_with_bars < min_sym_bars or coverage_pct < min_cov:
        return False, f"⚠️ Degraded run: low bar coverage {symbols_with_bars}/{universe_size} ({coverage_pct:.1f}% < {min_cov}%)."
    return True, ""

# ---------- Core pipeline ----------
def run_monitor(cfg, override_tickers: list | None = None, dry_run: bool = False):
    now_utc = utc_now()
    # per-run metrics baseline
    METRICS.update({
        "http_gets": 0, "pages_fetched": 0, "symbols_requested": 0,
        "symbols_with_bars": 0, "bars_rows": 0, "universe_size": 0,
        "run_started_utc": now_utc.isoformat(), "run_finished_utc": None, "errors": []
    })

    # allow env to override webhook
    cfg["webhook_url"]  = os.getenv("WEBHOOK_URL",  cfg.get("webhook_url",""))
    cfg["webhook_type"] = os.getenv("WEBHOOK_TYPE", cfg.get("webhook_type","discord"))

    # Sensitivity knobs
    bull_rsi_min = cfg.get("bull_rsi_min", 52)
    bear_rsi_max = cfg.get("bear_rsi_max", 48)
    volume_mult  = cfg.get("volume_mult", 0.8)
    rvol_same_wd = cfg.get("rvol_use_same_weekday", True)
    allow_persist = cfg.get("allow_persistent", True)

    log(f"[env] ALPACA_ENV={ALPACA_ENV}  feed={cfg['feed']}  timeframe={cfg['timeframe']}  now_utc={now_utc}")

    # Universe selection
    if override_tickers:
        universe = [t.strip().upper() for t in override_tickers if t.strip()]
        log(f"[smoke] overriding universe: {universe}")
    else:
        exchanges = cfg.get("universe_exchanges", ["NYSE","NASDAQ","AMEX"])
        all_us = get_us_equities(cfg["universe_limit"] * 3, exchanges)
        log(f"[universe] active US symbols from Alpaca: {len(all_us)} (sample: {all_us[:10]})")
        ranked = rank_by_today_dollar_vol(all_us, cfg["timeframe"], cfg["feed"], now_utc, cfg["universe_top_n"])
        universe = ranked["symbol"].tolist()
        rvol_map = compute_rvol_for(universe, cfg["feed"], now_utc,
                                    lookback_days=cfg["rvol_lookback_days"],
                                    use_same_weekday=rvol_same_wd)
        universe = [s for s in universe if rvol_map.get(s,1.0) >= cfg["rvol_min"]]
        log(f"[universe] after RVOL≥{cfg['rvol_min']}: {len(universe)}")

    METRICS["universe_size"] = len(universe)
    METRICS["symbols_requested"] = len(universe)

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

    if not allbars.empty:
        METRICS["bars_rows"] = len(allbars)
        METRICS["symbols_with_bars"] = allbars["symbol"].nunique()
    else:
        METRICS["bars_rows"] = 0
        METRICS["symbols_with_bars"] = 0

    # Early exit if no bars: warn + stats
    if allbars.empty:
        meta = {"run_time_utc": now_utc.isoformat(), "universe_size": len(universe), "signals_count": 0, "sample_symbols": universe[:10]}
        if cfg.get("health",{}).get("enabled", True):
            if len(universe) == 0:
                send_health_webhook(cfg, "🚫 No symbols in universe after filtering. Check feed, RVOL gates, or exchanges.")
            else:
                send_health_webhook(cfg, "⚠️ No bars returned for the selected window. Possible holiday/closed market/pagination/window issue.")
        if cfg.get("stats_report",{}).get("enabled", True):
            text = summarize_stats_text(cfg.get("stats_report",{}).get("level","compact"), len(universe), 0)
            send_health_webhook(cfg, text)
        METRICS["run_finished_utc"] = utc_now().isoformat()
        print(meta)
        return

    # Enrich per-symbol with indicators and session VWAP
    out_signals = []
    early_signals = []  # keep “predictive-ish” separate for formatting
    rankings_bull, rankings_bear = [], []

    last_run_cutoff_hours = cfg.get("signal_lookback_hours", 8)
    last_run_cutoff = load_last_run_time(now_utc, last_run_cutoff_hours)
    last_session_open = session_open_utc(now_utc, cfg["regular_hours"]["start"], cfg["regular_hours"]["tz"])
    last_run_cutoff = min(last_run_cutoff, last_session_open)

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

    # If override tickers, compute rvol_map now
    if override_tickers:
        rvol_map = compute_rvol_for(universe, cfg["feed"], now_utc,
                                    lookback_days=cfg["rvol_lookback_days"],
                                    use_same_weekday=rvol_same_wd)

    # Helper to push a signal dict
    def push_signal(container, sym, row, direction, rvol_value, kind):
        ts_iso_ct = row["t"].tz_convert(CENTRAL_TZ).isoformat()
        sig_id = f"{sym}|{kind}-{direction}|{ts_iso_ct}"
        if not prev_signals_df.empty and "id" in prev_signals_df.columns:
            if (prev_signals_df["id"] == sig_id).any():
                return
        session = "regular" if is_rth(
            row["t"], cfg["regular_hours"]["start"], cfg["regular_hours"]["end"], cfg["regular_hours"]["tz"]
        ) else "extended"
        sc = composite_score(row, rvol_value, direction=="bullish", cfg, row.get("ema50"), row.get("ema200"), row.get("sess_vwap"))
        container.append({
            "ticker": sym, "last_price": round(float(row["c"]), 4),
            "timestamp": ts_iso_ct, "direction": direction, "session": session,
            "id": sig_id, "score": round(float(sc), 3), "rvol": round(rvol_value, 2),
            "adx": round(float(row.get("adx", np.nan)), 2), "kind": kind
        })

    # Per-symbol loop
    for sym, g in allbars.groupby("symbol"):
        g = enrich_with_indicators(g, cfg)
        g["ema50"] = ema(g["c"], 50)
        g["ema200"] = ema(g["c"], 200)
        g["sess_vwap"] = session_vwap(g, now_utc)
        if len(g) < cfg["ema_slow"]+1:
            continue

        slope_up, slope_down = confirm_map.get(sym, (True, True))
        recent = g[g["t"] >= last_run_cutoff]
        crosses = recent[(recent["bull_cross"]) | (recent["bear_cross"])]

        # RVOL for latest bar
        last = g.iloc[-1]
        rvol_last = float(rvol_map.get(sym, 1.0))
        vol_ok_last = last["v"] >= volume_mult * last["vol_ma20"]

        # ---------------- Traditional cross alerts ----------------
        for _, row in crosses.iterrows():
            is_bull = bool(row["bull_cross"])
            if row["c"] <= cfg["price]()
