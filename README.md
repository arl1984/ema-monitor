# NYSE EMA Monitor

NYSE EMA Monitor is a Python project that scans U.S. equities for 20/50 exponential moving average (EMA) crosses and related trend signals. It retrieves market data from the [Alpaca Markets](https://alpaca.markets/) API, ranks potential setups, and can send alerts via webhooks.

## Features

- **Classic EMA signals** – Detects bullish/bearish 20/50 EMA crosses.
- **Early signals** – Predictive checks such as near crosses, MACD, RSI regimes, VWAP reclaims, and squeeze breakouts.
- **Volume & price filters** – Ensures candidates meet minimum price, relative volume, and ADX requirements.
- **Ranking snapshot** – Optional ranking of top bullish and bearish symbols based on composite scores.
- **Webhook support** – Sends alerts to Discord or other services defined in configuration.

## Setup

1. Install Python 3.10+.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Provide Alpaca credentials via environment variables:
   ```bash
   export ALPACA_KEY_ID="your-key"
   export ALPACA_SECRET_KEY="your-secret"
   ```

## Configuration

Configuration lives in `config.yaml` with alternative presets in `config.loose.yaml` and `config.strict.yaml`. Key options include:

- Universe selection and size limits
- Indicator parameters (EMA lengths, RSI period, ADX thresholds)
- Signal behavior (minimum scores, maximum alerts, early-signal tuning)
- Webhook and CSV output settings

Adjust these files to match your trading preferences or alert destinations.

## Usage

Run a dry scan (no alerts or file writes) on custom tickers:

```bash
python monitor.py --dry-run --tickers AAPL,MSFT
```

For live operation against the full configured universe:

```bash
python monitor.py
```

Additional utilities:

- `sanity_sim_crosses.py` – quick simulation of EMA cross behavior.
- `summary.py` – summarises latest signals and metrics.

## License

This project is provided as-is without warranty. Review and test thoroughly before using in production trading environments.

