import pandas as pd
import numpy as np

from datetime import datetime, timedelta, timezone

def ema(series: pd.Series, span:int):
    return series.ewm(span=span, adjust=False).mean()

def rsi(series: pd.Series, period=14):
    d = series.diff()
    gain = d.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100 - (100/(1+rs))
    return out.fillna(50)

def make_bars(n:int, start_price:float, drift:float=0.0, vol:float=0.5, start_ts:pd.Timestamp=None, tf_minutes:int=5):
    if start_ts is None:
        start_ts = pd.Timestamp.utcnow().floor("T").tz_localize("UTC")
    t = [start_ts + pd.Timedelta(minutes=tf_minutes*i) for i in range(n)]
    noise = np.random.normal(loc=drift, scale=vol, size=n).cumsum()
    c = start_price + noise
    c = pd.Series(c).clip(lower=max(0.5, start_price*0.2))
    o = c.shift(1).fillna(c.iloc[0])
    h = pd.concat([o, c], axis=1).max(axis=1) + np.random.rand(n)*0.1
    l = pd.concat([o, c], axis=1).min(axis=1) - np.random.rand(n)*0.1
    v = pd.Series(1_000 + np.random.randint(0, 500, size=n))
    df = pd.DataFrame({"t": pd.to_datetime(t, utc=True), "o": o, "h": h, "l": l, "c": c, "v": v})
    return df

def enrich(df: pd.DataFrame, ema_fast=20, ema_slow=50, rsi_period=14):
    df = df.sort_values("t").copy()
    df["ema_fast"] = ema(df["c"], ema_fast)
    df["ema_slow"] = ema(df["c"], ema_slow)
    df["rsi"] = rsi(df["c"], rsi_period)
    df["vol_ma20"] = df["v"].rolling(20, min_periods=1).mean()
    df["adx"] = 25.0
    spread = df["ema_fast"] - df["ema_slow"]
    sign = np.sign(spread)
    df["bull_cross"] = (sign.shift(1) <= 0) & (sign > 0)
    df["bear_cross"] = (sign.shift(1) >= 0) & (sign < 0)
    return df

def session_open_utc(now_utc: pd.Timestamp, start="09:30", tzname="America/New_York"):
    from dateutil import tz as _tz
    et = now_utc.tz_convert(_tz.gettz(tzname))
    sh, sm = map(int, start.split(":"))
    so = et.replace(hour=sh, minute=sm, second=0, microsecond=0)
    if et.time() < so.time():
        so = so - pd.Timedelta(days=1)
    return pd.Timestamp(so).tz_convert("UTC")

def scenario_cross_earlier_than_lookback():
    now = pd.Timestamp.utcnow().tz_localize("UTC")
    start = now - pd.Timedelta(hours=6)
    df = make_bars(70, start_price=50, drift=0.02, vol=0.15, start_ts=start, tf_minutes=5)
    jump_idx = 12
    df.loc[jump_idx:jump_idx+5, "c"] += 2.0
    df = enrich(df)
    last_run_cutoff = now - pd.Timedelta(hours=2)
    win = min(last_run_cutoff, session_open_utc(now))
    recent = df[df["t"] >= win]
    crosses = recent[recent["bull_cross"] | recent["bear_cross"]]
    print("\n--- Scenario 1: Cross earlier than lookback, catch via session-open ---")
    print(f"Bars total: {len(df)}, recent window bars: {len(recent)}")
    print(f"Cross rows in recent? {len(crosses)}")
    if not crosses.empty:
        last_cross = crosses.iloc[-1]
        print(f"✓ caught cross: time={last_cross['t']} type={'BULL' if last_cross['bull_cross'] else 'BEAR'} "
              f"ema20={last_cross['ema_fast']:.2f} ema50={last_cross['ema_slow']:.2f} rsi={last_cross['rsi']:.1f}")
    else:
        print("✗ did not catch any cross (this would have been missed without session-open expansion).")

def scenario_persistent_without_new_cross():
    now = pd.Timestamp.utcnow().tz_localize("UTC")
    start = now - pd.Timedelta(hours=7)
    df = make_bars(84, start_price=25, drift=0.03, vol=0.12, start_ts=start, tf_minutes=5)
    df.loc[6:12, "c"] += 3.0
    df.loc[13:, "c"] += np.linspace(0.0, 1.5, len(df)-13)
    df = enrich(df)
    last_run_cutoff = now - pd.Timedelta(hours=3)
    recent = df[df["t"] >= last_run_cutoff]
    crosses = recent[recent["bull_cross"] | recent["bear_cross"]]
    print("\n--- Scenario 2: No new cross, but persistent bull setup ---")
    print(f"Cross in recent window? {len(crosses)} (expect 0)")
    last = df.iloc[-1]
    bull_rsi_min = 52
    adx_min = 18
    rvol = 1.5
    volume_ok = True
    bull_ctx = (last["ema_fast"] > last["ema_slow"]) and (last["rsi"] > bull_rsi_min)
    if (not bull_ctx) or (not volume_ok) or (last["adx"] < adx_min) or (rvol < 1.2) or (last["c"] <= 3):
        print("✗ persistent criteria not met (unexpected for this scenario).")
    else:
        score = 0.0
        score += min(max((rvol - 1.0), 0.0), 2.0)
        score += 1.0
        if last["c"] > last["ema_slow"]:
            score += 0.5
        if (last["ema_fast"] - last["ema_slow"]) / max(abs(last["ema_slow"]), 1e-9) > 0.0025:
            score += 0.25
        print(f"✓ persistent bull valid — latest bar: time={last['t']} "
              f"ema20={last['ema_fast']:.2f} ema50={last['ema_slow']:.2f} rsi={last['rsi']:.1f} score≈{score:.2f}")

if __name__ == "__main__":
    np.random.seed(7)
    scenario_cross_earlier_than_lookback()
    scenario_persistent_without_new_cross()
