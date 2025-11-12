"""
DXLink API Quote Token management.

Fetches a quote token via GET /api-quote-tokens using the existing
OAuth-authenticated requests.Session, caches it locally for 24h,
and exposes a helper to retrieve a valid token and websocket URL.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Tuple, Optional

import requests


TOKEN_TTL_SECONDS = 24 * 60 * 60  # 24 hours
TOKEN_FILE = Path(__file__).parent / ".dxlink_token.json"


def _load_cached_token() -> Optional[Dict]:
    try:
        if not TOKEN_FILE.exists():
            return None
        with open(TOKEN_FILE, "r") as f:
            data = json.load(f)
        saved_at = float(data.get("saved_at", 0))
        # Add small safety buffer of 60s
        if (time.time() - saved_at) < (TOKEN_TTL_SECONDS - 60):
            return data
        return None
    except Exception:
        return None


def _save_cached_token(token: Dict) -> None:
    try:
        token_copy = dict(token)
        token_copy["saved_at"] = time.time()
        with open(TOKEN_FILE, "w") as f:
            json.dump(token_copy, f, indent=2)
    except Exception as e:
        print(f"⚠️ Failed to cache DXLink token: {e}")


def get_quote_token(session: requests.Session, force_refresh: bool = False) -> Tuple[str, str]:
    """Return (token, dxlink_url).

    Uses cached token when available and not expired. If force_refresh is True,
    always re-fetch from the API.
    """
    if not force_refresh:
        cached = _load_cached_token()
        if cached and isinstance(cached.get("data"), dict):
            data = cached["data"]
            tok = data.get("token")
            url = data.get("dxlink-url") or data.get("dxlink_url")
            if tok and url:
                return str(tok), str(url)

    urls = [
        "https://api.tastyworks.com/api-quote-tokens",
        "https://api.tastytrade.com/api-quote-tokens",
    ]
    resp = None
    payload = None
    for u in urls:
        r = session.get(u)
        if r.status_code == 200:
            resp = r
            payload = r.json()
            break
        last_err = f"{r.status_code} {r.text}"
    if payload is None:
        raise RuntimeError(
            f"Failed to get api quote token: {last_err}")
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), dict):
        raise RuntimeError("Unexpected api-quote-tokens response shape")
    _save_cached_token(payload)
    tok = payload["data"].get("token")
    url = payload["data"].get(
        "dxlink-url") or payload["data"].get("dxlink_url")
    if not tok or not url:
        raise RuntimeError(
            "api-quote-tokens response missing token or dxlink-url")
    return str(tok), str(url)


def clear_cached_quote_token() -> None:
    try:
        if TOKEN_FILE.exists():
            TOKEN_FILE.unlink()
            print("✅ Cleared cached DXLink token")
    except Exception as e:
        print(f"⚠️ Failed to clear cached DXLink token: {e}")
