import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from dateutil import tz
import yaml
import requests
from typing import Optional

def load_cfg():
    with open("config.yaml","r") as f:
        return yaml.safe_load(f)

def load_signals():
    p = "data/signals.csv"
    if not os.path.exists(p):
        return pd.DataFrame(columns=["ticker","last_price","timestamp","direction","session"])
    df = pd.read_csv(p)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

def ct_now():
    return datetime.utcnow().replace(tzinfo=tz.tzutc()).astimezone(tz.gettz("America/Chicago"))

EASTERN_TZ = tz.gettz("America/New_York")

def is_market_hours(now: Optional[datetime] = None) -> bool:
    now = now.astimezone(EASTERN_TZ) if now else datetime.now(EASTERN_TZ)
    if now.weekday() >= 5:
        return False
    open_t = datetime(now.year, now.month, now.day, 9, 30, tzinfo=EASTERN_TZ)
    close_t = datetime(now.year, now.month, now.day, 16, 0, tzinfo=EASTERN_TZ)
    return open_t <= now <= close_t

def plot_save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

def filter_day(df, today_ct):
    day = today_ct.date()
    start = datetime(day.year, day.month, day.day, 0,0, tzinfo=today_ct.tzinfo)
    end   = start + timedelta(days=1)
    return df[(df["timestamp"]>=start) & (df["timestamp"]<end)].copy()

def filter_week(df, today_ct):
    start = today_ct - timedelta(days=7)
    return df[(df["timestamp"]>=start) & (df["timestamp"]<=today_ct)].copy()

def charts_and_payload(dd: pd.DataFrame, title_prefix: str):
    if dd.empty:
        return None, f"{title_prefix}: No signals."
    by_dir = dd["direction"].value_counts().reindex(["bullish","bearish"]).fillna(0).astype(int)
    fig1 = plt.figure(figsize=(5,3))
    plt.bar(by_dir.index, by_dir.values)
    plt.title(f"{title_prefix} — Bulls vs Bears"); plt.xlabel("Direction"); plt.ylabel("Count")
    plot_save(fig1, "artifacts/bulls_bears.png")

    top = dd["ticker"].value_counts().head(10)
    fig2 = plt.figure(figsize=(6,4))
    plt.barh(top.index[::-1], top.values[::-1])
    plt.title(f"{title_prefix} — Top 10 Tickers by Signals"); plt.xlabel("Signals"); plt.ylabel("Ticker")
    plot_save(fig2, "artifacts/top_tickers.png")

    sector_path = "data/sectors.csv"
    sector_note = ""
    if os.path.exists(sector_path):
        m = pd.read_csv(sector_path)
        dd2 = dd.merge(m, on="ticker", how="left")
        sec = dd2["sector"].fillna("Unknown").value_counts()
        fig3 = plt.figure(figsize=(6,4))
        plt.bar(sec.index, sec.values)
        plt.title(f"{title_prefix} — Signals by Sector"); plt.xlabel("Sector"); plt.ylabel("Signals")
        plt.xticks(rotation=45, ha="right")
        plot_save(fig3, "artifacts/sectors.png")
    else:
        sector_note = "(Sector chart skipped — add data/sectors.csv to enable.)"

    bulls = int(by_dir.get("bullish",0)); bears = int(by_dir.get("bearish",0))
    tickers = ", ".join(sorted(dd["ticker"].unique()))[:500]
    text = f"{title_prefix}\nBullish: {bulls} | Bearish: {bears}\nTickers: {tickers}\n{sector_note}"
    return ["artifacts/bulls_bears.png","artifacts/top_tickers.png","artifacts/sectors.png"], text

def send_webhook(cfg, text: str):
    url = cfg.get("webhook_url","");
    if not url:
        print("No webhook configured; saved artifacts only.");
        return
    if not is_market_hours():
        return
    typ = cfg.get("webhook_type","generic").lower()
    payload = ({"text": text} if typ=="slack" else {"content": text} if typ=="discord" else {"summary": text})
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print("Webhook error:", e)

def main(kind: str):
    cfg = load_cfg()
    cfg["webhook_url"]  = os.getenv("WEBHOOK_URL",  cfg.get("webhook_url",""))
    cfg["webhook_type"] = os.getenv("WEBHOOK_TYPE", cfg.get("webhook_type","discord"))
    df = load_signals()
    now_ct = ct_now()
    if kind == "daily":
        dd = filter_day(df, now_ct)
        files, text = charts_and_payload(dd, f"Daily EMA Cross Summary — {now_ct.date()}")
    else:
        dd = filter_week(df, now_ct)
        files, text = charts_and_payload(dd, f"Weekly EMA Cross Summary — ending {now_ct.date()}")
    print(text)
    send_webhook(cfg, text)

if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv)>1 else "daily")
