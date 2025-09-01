import os, sys, json
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from dateutil import tz
import yaml
from typing import List, Dict

ALPACA_KEY   = os.environ.get("ALPACA_KEY_ID", "")
ALPACA_SECRET= os.environ.get("ALPACA_SECRET_KEY", "")
BASE_DATA    = "https://data.alpaca.markets/v2"
ALPACA_ENV = os.environ.get("ALPACA_ENV", "paper").lower()  # 'paper' or 'live'
BASE_BROKER  = "https://api.alpaca.markets/v2"  # for /assets
HEADERS = {"APCA-API-KEY-ID": ALPACA_KEY, "APCA-API-SECRET-KEY": ALPACA_SECRET}

def load_cfg():
    with open("config.yaml","r") as f:
        return yaml.safe_load(f)

def rsi(series: pd.Series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100 - (100/(1+rs))
    return out.fillna(50)

def ema(series: pd.Series, span:int):
    return series.ewm(span=span, adjust=False).mean()

def is_rth(ts_utc: pd.Timestamp, start="09:30", end="16:00", tzname="America/New_York"):
    et = ts_utc.tz_convert(tz.gettz(tzname))
    t = et.time()
    sh,sm = map(int, start.split(":"))
    eh,em = map(int, end.split(":"))
    return (t >= datetime(et.year,et.month,et.day,sh,sm).time() and
            t <= datetime(et.year,et.month,et.day,eh,em).time())

def get_nyse_symbols(limit_universe:int) -> List[str]:
    # Pull all active US equity assets and filter to NYSE
    r = requests.get(f"{BASE_BROKER}/assets", headers=HEADERS, timeout=30)
    r.raise_for_status()
    assets = r.json()
    nyse = [a["symbol"] for a in assets
            if a.get("class")=="us_equity" and a.get("status")=="active" and a.get("exchange")=="NYSE"]
    return sorted(list(set(nyse)))  # dedupe

def fetch_bars_chunk(symbols: List[str], start_iso: str, end_iso: str, timeframe: str, feed: str) -> pd.DataFrame:
    if not symbols: return pd.DataFrame()
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
    if not rows: return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values(["symbol","t"]).reset_index(drop=True)
    df["t"] = pd.to_datetime(df["t"], utc=True)
    return df

def rank_top_by_today_volume(all_symbols: List[str], tf: str, feed: str, now_utc: pd.Timestamp) -> List[str]:
    # Fetch today's bars (extended hours included) and sum volume per symbol
    et = now_utc.tz_convert(tz.gettz("America/New_York"))
    start_et = et.replace(hour=0, minute=0, second=0, microsecond=0)
    start_utc = pd.Timestamp(start_et).tz_convert(timezone.utc)
    start_iso = start_utc.isoformat()
    end_iso = now_utc.isoformat()

    vols = []
    chunk = 50
    for i in range(0, len(all_symbols), chunk):
        part = all_symbols[i:i+chunk]
        bars = fetch_bars_chunk(part, start_iso, end_iso, tf, feed)
        if bars.empty:
            continue
        agg = bars.groupby("symbol")["v"].sum().reset_index().rename(columns={"v":"vol"})
        vols.append(agg)
    if not vols:
        return all_symbols[:100]
    voldf = pd.concat(vols, ignore_index=True).sort_values("vol", ascending=False)
    return voldf["symbol"].head(100).tolist()

def compute_signals(bars: pd.DataFrame, cfg: dict, now_utc: pd.Timestamp):
    out = []
    if bars.empty: return out
    for sym, g in bars.groupby("symbol"):
        g = g.sort_values("t").copy()
        g["ema_fast"] = ema(g["c"], cfg["ema_fast"])
        g["ema_slow"] = ema(g["c"], cfg["ema_slow"])
        g["rsi"] = rsi(g["c"], cfg["rsi_period"])
        g["vol_ma20"] = g["v"].rolling(20, min_periods=1).mean()

        if len(g) < cfg["ema_slow"]+1:
            continue
        last = g.iloc[-1]; prev = g.iloc[-2]

        # price filter
        if last["c"] <= cfg["price_min"]:
            continue
        # volume filter
        if last["v"] < cfg["volume_mult"] * last["vol_ma20"]:
            continue

        bull = (prev["ema_fast"] <= prev["ema_slow"]) and (last["ema_fast"] > last["ema_slow"]) and (last["rsi"] > 50)
        bear = (prev["ema_fast"] >= prev["ema_slow"]) and (last["ema_fast"] < last["ema_slow"]) and (last["rsi"] < 50)
        if not (bull or bear):
            continue

        session = "regular" if is_rth(last["t"], cfg["regular_hours"]["start"], cfg["regular_hours"]["end"], cfg["regular_hours"]["tz"]) else "extended"
        out.append({
            "ticker": sym,
            "last_price": round(float(last["c"]), 4),
            "timestamp": last["t"].tz_convert(tz.gettz("America/Chicago")).isoformat(),
            "direction": "bullish" if bull else "bearish",
            "session": session
        })
    return out

def send_webhook(cfg, signals):
    url = cfg.get("webhook_url","")
    if not url or not signals:
        return
    typ = cfg.get("webhook_type","generic").lower()
    if typ == "slack":
        text = "\n".join([f"*{s['ticker']}* {s['direction'].upper()} @ ${s['last_price']} — {s['timestamp']} ({s['session']})" for s in signals])
        payload = {"text": f":chart_with_upwards_trend: NYSE 20/50 EMA crosses\n{text}"}
    elif typ == "discord":
        content = "\n".join([f"**{s['ticker']}** {s['direction'].upper()} @ ${s['last_price']} — {s['timestamp']} ({s['session']})" for s in signals])
        payload = {"content": f"📈 NYSE 20/50 EMA crosses\n{content}"}
    else:
        payload = {"signals": signals}
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print("Webhook error:", e, file=sys.stderr)

def append_log(signals, path="data/signals.csv"):
    if not signals: return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame(signals)
    if os.path.exists(path):
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        df.to_csv(path, index=False)

def main():
    if not (ALPACA_KEY and ALPACA_SECRET):
        print("Missing ALPACA_KEY_ID / ALPACA_SECRET_KEY")
        sys.exit(1)
    cfg = load_cfg()
    now_utc = pd.Timestamp.now(tz="UTC")

    # Universe
    all_nyse = get_nyse_symbols(cfg["universe_limit"]*3)
    top100 = rank_top_by_today_volume(all_nyse, cfg["timeframe"], cfg["feed"], now_utc)

    # Fetch last 60 bars for signals
    start_utc = now_utc - timedelta(days=7)
    df_all = pd.DataFrame()
    chunk = 50
    for i in range(0, len(top100), chunk):
        part = top100[i:i+chunk]
        bars = fetch_bars_chunk(part, start_utc.isoformat(), now_utc.isoformat(), cfg["timeframe"], cfg["feed"])
        if not bars.empty:
            df_all = pd.concat([df_all, bars], ignore_index=True)

    signals = compute_signals(df_all, cfg, now_utc)
    if signals:
        send_webhook(cfg, signals)
        append_log(signals)

    meta = {"run_time_utc": now_utc.isoformat(), "universe_size": len(top100), "signals_count": len(signals)}
    os.makedirs("data", exist_ok=True)
    with open("data/last_run.json","w") as f:
        json.dump(meta, f, indent=2)
    print(meta)

if __name__ == "__main__":
    main()
