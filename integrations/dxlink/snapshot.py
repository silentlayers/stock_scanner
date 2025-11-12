from __future__ import annotations

import threading
from typing import Dict, List, Optional

from integrations.dxlink.quote_token import get_quote_token
from integrations.dxlink.client import DXLinkClient, DEFAULT_EVENT_FIELDS


def get_option_snapshot(session, symbols: List[str], timeout: int = 6) -> Dict[str, dict]:
    """Subscribe to Greeks+Quote for the given symbols and return first events per symbol.

    Returns a dict mapping symbol -> {greeks: {...}, quote: {...}} (when available).
    """
    if not symbols:
        return {}

    qtoken, wss_url = get_quote_token(session)
    accept_fields = dict(DEFAULT_EVENT_FIELDS)
    accept_fields["Greeks"] = accept_fields.get("Greeks", [
        "eventType", "eventSymbol", "volatility", "delta", "gamma", "theta", "rho", "vega"
    ])

    client = DXLinkClient(
        wss_url, qtoken, accept_event_fields=accept_fields, keepalive_interval=30, log=False)
    client.connect()
    client.subscribe(symbols, ["Greeks", "Quote"], reset=True)

    results: Dict[str, dict] = {}
    done = threading.Event()

    def on_event(row: dict) -> None:
        et = row.get("eventType")
        sym = row.get("eventSymbol")
        if not isinstance(sym, str):
            return
        slot = results.setdefault(sym, {})
        if et == "Greeks" and "greeks" not in slot:
            slot["greeks"] = {
                "delta": row.get("delta"),
                "gamma": row.get("gamma"),
                "theta": row.get("theta"),
                "vega": row.get("vega"),
                "volatility": row.get("volatility"),
            }
        elif et == "Quote" and "quote" not in slot:
            slot["quote"] = {
                "bidPrice": row.get("bidPrice"),
                "askPrice": row.get("askPrice"),
                "bidSize": row.get("bidSize"),
                "askSize": row.get("askSize"),
            }
        if all((("greeks" in results.get(s, {})) and ("quote" in results.get(s, {}))) for s in symbols):
            done.set()

    t = threading.Thread(target=client.read_forever,
                         args=(on_event,), daemon=True)
    t.start()
    done.wait(timeout=timeout)
    client.stop()
    return results


def get_symbol_price_snapshot(session, symbol: str, timeout: int = 3) -> dict | None:
    """Return a quick price snapshot for a symbol using DXLink.

    Subscribes to Quote and Trade, waits up to timeout seconds, and returns:
      { 'symbol': str, 'price': float, 'trade': {...} | None, 'quote': {...} | None }

    Price is chosen by preference: Trade.price, else (bid+ask)/2, else ask, else bid.
    Returns None if no data arrives within timeout.
    """
    if not symbol:
        return None
    token, url = get_quote_token(session)
    accept_fields = dict(DEFAULT_EVENT_FIELDS)
    client = DXLinkClient(
        url, token, accept_event_fields=accept_fields, keepalive_interval=30, log=False)
    client.connect()
    client.subscribe([symbol], ["Quote", "Trade"], reset=True)

    data: Dict[str, Optional[dict]] = {"trade": None, "quote": None}
    done = threading.Event()

    def on_event(row: dict) -> None:
        et = row.get("eventType")
        if et == "Trade" and data.get("trade") is None:
            data["trade"] = {
                "price": row.get("price"),
                "size": row.get("size"),
                "dayVolume": row.get("dayVolume"),
            }
        elif et == "Quote" and data.get("quote") is None:
            data["quote"] = {
                "bidPrice": row.get("bidPrice"),
                "askPrice": row.get("askPrice"),
                "bidSize": row.get("bidSize"),
                "askSize": row.get("askSize"),
            }
        if data.get("trade") and data.get("quote"):
            done.set()

    t = threading.Thread(target=client.read_forever,
                         args=(on_event,), daemon=True)
    t.start()
    done.wait(timeout=timeout)
    client.stop()

    trade = data.get("trade")
    quote = data.get("quote")
    if not trade and not quote:
        return None
    price = None
    # Prefer last trade price
    if trade and isinstance(trade.get("price"), (int, float)):
        price = float(trade["price"])
    else:
        # Fall back to mid or one-sided
        bid = quote.get("bidPrice") if quote else None
        ask = quote.get("askPrice") if quote else None
        try:
            if isinstance(bid, (int, float)) and isinstance(ask, (int, float)):
                price = (float(bid) + float(ask)) / 2.0
            elif isinstance(ask, (int, float)):
                price = float(ask)
            elif isinstance(bid, (int, float)):
                price = float(bid)
        except Exception:
            price = None
    return {"symbol": symbol, "price": price, "trade": trade, "quote": quote}
