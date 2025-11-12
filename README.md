# stock_scanner

Personal SPY signal + put credit spread scanner with Tastytrade OAuth and DXLink snapshots.

⚠️ **Important:** This app requires OAuth authentication with a local callback server. It **will not work on Streamlit Cloud's free tier**. For cloud deployment, use Railway, Render, or Fly.io instead.

## Quick start

If you have Python 3.10+ installed, this one-liner will set up a virtualenv, install deps, and launch the UI:

```bash
./run.sh
```

Tips:

- To use a custom port: `./run.sh --server.port 8502`
- In VS Code, open the Command Palette → “Run Task” → “Run Streamlit App” for one-click launch.

## Setup

1. Python 3.10+ recommended.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` in the project root:

```
TASTYTRADE_CLIENT_ID=your_client_id
TASTYTRADE_CLIENT_SECRET=your_client_secret
TASTYTRADE_REDIRECT_URI=http://localhost:3000/callback
# Optional
# TASTYTRADE_ENV=sandbox
# Override OAuth scopes (default: "openid read trade")
# TASTYTRADE_SCOPES="openid read trade"
```

Notes:

- Local HTTPS is disabled via `OAUTHLIB_INSECURE_TRANSPORT=1` to support localhost redirect during development.
- Secrets remain on your machine.

## Run (CLI)

```bash
python3 main.py
```

- Opens a browser for OAuth.
- Prints signal diagnostics and, if valid, scans for put credit spreads.

## Run (Streamlit UI)

```bash
streamlit run streamlit_app.py
```

- Tab “Signal”: shows Close vs EMA21, MACD, RSI, and VIX metrics and a line chart.
- Tab “Spreads”: click “Authorize Tastytrade” to sign in; then it fetches and evaluates a candidate spread.

## Architecture

- `config.py`: env loading and OAuth endpoints
- `auth.py`: automated OAuth (local callback) → authorized requests.Session
- `data.py`: yfinance download + robust Close series extraction
- `indicators.py`: EMA(21), MACD diff, RSI(14)
- `market_signal.py`: signal boolean + diagnostics
- `options_scanner.py`: fetch options JSON and compute spread metrics
- `dxlink_snapshot.py`: DXLink utilities for quick snapshots (prices/Greeks)
- `main.py`: CLI orchestrator
- `streamlit_app.py`: Streamlit dashboard

### Alternative launcher

You can also use the included `run.sh`, which creates `.venv/`, installs `requirements.txt`, sets `OAUTHLIB_INSECURE_TRANSPORT=1` for localhost OAuth, and runs Streamlit. This is what the VS Code task uses under the hood.

## Customize thresholds

Modify `market_signal.py` (VIX < 20, RSI > 50, etc.) and `options_scanner.py` (DTE/delta ranges, ROR threshold) to your liking. You can also parameterize via env vars or Streamlit widgets.

## Cloud Deployment

**Note:** This app requires OAuth with a local callback server, which doesn't work on Streamlit Cloud's free tier.

### Recommended Options:

1. **Run Locally** (Easiest)

   ```bash
   git clone https://github.com/silentlayers/stock_scanner.git
   cd stock_scanner
   ./run.sh
   ```

2. **Railway** (Best for cloud)

   - Free $5/month credit
   - Supports OAuth callbacks
   - Add environment variables in dashboard
   - Deploy directly from GitHub

3. **Render**

   - Free tier with 512MB RAM
   - Configure redirect URI to your Render URL
   - Note: Sleeps after 15min inactivity

4. **Fly.io**
   - Free tier available
   - Requires Dockerfile (not included)
   - More complex setup

### For Streamlit Cloud Users:

If you want to use Streamlit Cloud, you'll need to:

1. Remove OAuth dependency
2. Use TastyTrade session tokens directly (store in secrets)
3. Modify authentication to use token-based auth instead of OAuth

## Troubleshooting

- **OAuth fails on cloud:** This app requires a local server for OAuth callbacks. Run locally or use Railway/Render.
- If OAuth fails with https error: ensure redirect URI is http://localhost:3000/callback and the app sets `OAUTHLIB_INSECURE_TRANSPORT=1` (already handled by config.py import).
- If yfinance columns seem odd, `data.get_close_series` normalizes to a 1D Series.
- If `GET /api-quote-tokens` returns 403 insufficient scopes:
  - Ensure your account is a tastytrade customer (not just a username signup).
  - Set `TASTYTRADE_SCOPES="openid read trade"` in `.env` (or as provided by tastytrade for streaming access), then delete `.tastytrade_tokens.json` and re-run to trigger a fresh OAuth with updated scopes.
