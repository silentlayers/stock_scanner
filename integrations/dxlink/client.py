"""
DXLink WebSocket client for streaming tastytrade market data.

Implements the minimal protocol steps:
  - SETUP
  - AUTH with api quote token
  - CHANNEL_REQUEST (service FEED)
  - FEED_SETUP (COMPACT format with selected fields)
  - FEED_SUBSCRIPTION (subscribe to event types for given symbols)
  - KEEPALIVE loop

Parsing FEED_DATA in COMPACT format using the requested acceptEventFields.
"""
from __future__ import annotations

import json
import threading
import time
from typing import Dict, List, Iterable, Callable, Any, Optional

from websocket import create_connection, WebSocket  # type: ignore


DEFAULT_EVENT_FIELDS: Dict[str, List[str]] = {
    # The order here must match what DxLink will send back in COMPACT mode
    "Trade": ["eventType", "eventSymbol", "price", "dayVolume", "size"],
    "TradeETH": ["eventType", "eventSymbol", "price", "dayVolume", "size"],
    "Quote": ["eventType", "eventSymbol", "bidPrice", "askPrice", "bidSize", "askSize"],
    "Greeks": ["eventType", "eventSymbol", "volatility", "delta", "gamma", "theta", "rho", "vega"],
    "Profile": [
        "eventType",
        "eventSymbol",
        "description",
        "shortSaleRestriction",
        "tradingStatus",
        "statusReason",
        "haltStartTime",
        "haltEndTime",
        "highLimitPrice",
        "lowLimitPrice",
        "high52WeekPrice",
        "low52WeekPrice",
    ],
    "Summary": [
        "eventType",
        "eventSymbol",
        "openInterest",
        "dayOpenPrice",
        "dayHighPrice",
        "dayLowPrice",
        "prevDayClosePrice",
    ],
}


class DXLinkClient:
    def __init__(
        self,
        url: str,
        token: str,
        accept_event_fields: Optional[Dict[str, List[str]]] = None,
        keepalive_interval: int = 30,
        log: bool = True,
    ) -> None:
        self.url = url
        self.token = token
        self.accept_event_fields = accept_event_fields or DEFAULT_EVENT_FIELDS
        self.keepalive_interval = keepalive_interval
        self.log = log
        self.ws: Optional[WebSocket] = None
        self._stop = threading.Event()
        self._channel_id = 3  # arbitrary channel as in docs

    # ---------------
    # Public API
    # ---------------
    def connect(self) -> None:
        self.ws = create_connection(self.url)
        if self.log:
            print(f"🌐 Connected to DXLink: {self.url}")
        self._send({
            "type": "SETUP",
            "channel": 0,
            "version": "0.1-DXF-PY/0.1.0",
            "keepaliveTimeout": 60,
            "acceptKeepaliveTimeout": 60,
        })
        # Startup handshake loop: wait for AUTH_STATE UNAUTHORIZED then AUTH
        authorized = False
        setup_seen = False
        while True:
            msg = self._recv_json(timeout=10)
            if not msg:
                raise RuntimeError("No response from DXLink during setup")
            mtype = msg.get("type")
            if self.log:
                if mtype in ("SETUP", "AUTH_STATE"):
                    print(f"⬇️  {mtype}: {msg}")
            if mtype == "SETUP":
                setup_seen = True
            elif mtype == "AUTH_STATE" and msg.get("state") == "UNAUTHORIZED":
                # Send AUTH
                self._send({"type": "AUTH", "channel": 0, "token": self.token})
            elif mtype == "AUTH_STATE" and msg.get("state") == "AUTHORIZED":
                authorized = True
                if self.log:
                    print("✅ AUTHORIZED with DXLink")
                break
            # else keep looping until authorized
        if not setup_seen or not authorized:
            raise RuntimeError("Failed to authorize with DXLink")

        # Open a FEED channel
        self._send({"type": "CHANNEL_REQUEST", "channel": self._channel_id,
                   "service": "FEED", "parameters": {"contract": "AUTO"}})
        while True:
            msg = self._recv_json(timeout=10)
            if not msg:
                raise RuntimeError("No response while opening FEED channel")
            mtype = msg.get("type")
            if mtype == "CHANNEL_OPENED" and msg.get("channel") == self._channel_id:
                if self.log:
                    print("📡 FEED channel opened")
                break

        # Configure feed: COMPACT format with selected fields
        self._send({
            "type": "FEED_SETUP",
            "channel": self._channel_id,
            "acceptAggregationPeriod": 0.1,
            "acceptDataFormat": "COMPACT",
            "acceptEventFields": self.accept_event_fields,
        })
        # Wait for FEED_CONFIG ack (optional)
        _ = self._recv_json(timeout=5)

        # Start keepalive thread
        threading.Thread(target=self._keepalive_loop, daemon=True).start()

    def subscribe(self, symbols: Iterable[str], events: Iterable[str], reset: bool = True) -> None:
        add_list = []
        for sym in symbols:
            for ev in events:
                add_list.append({"type": ev, "symbol": sym})
        self._send({
            "type": "FEED_SUBSCRIPTION",
            "channel": self._channel_id,
            "reset": bool(reset),
            "add": add_list,
        })
        if self.log:
            print(
                f"📝 Subscribed to {len(add_list)} feeds for symbols: {', '.join(symbols)}")

    def read_forever(self, on_event: Optional[Callable[[Dict[str, Any]], None]] = None) -> None:
        """Read messages until stop() is called. For FEED_DATA messages, parse and emit rows.
        Other messages are printed when logging is enabled.
        """
        while not self._stop.is_set():
            msg = self._recv_json(timeout=60)
            if not msg:
                continue
            mtype = msg.get("type")
            if mtype == "FEED_DATA":
                for row in self._parse_feed_data(msg):
                    if on_event:
                        on_event(row)
                    elif self.log:
                        print(json.dumps(row))
            elif mtype in ("KEEPALIVE", "AUTH_STATE", "CHANNEL_OPENED", "FEED_CONFIG"):
                if self.log:
                    print(f"⬇️  {mtype}: {msg}")
            else:
                if self.log:
                    print(f"🔹 {msg}")

    def stop(self) -> None:
        self._stop.set()
        try:
            if self.ws is not None:
                self.ws.close()
        except Exception:
            pass

    # ---------------
    # Internals
    # ---------------
    def _send(self, obj: Dict[str, Any]) -> None:
        assert self.ws is not None, "WebSocket not connected"
        self.ws.send(json.dumps(obj))

    def _recv_json(self, timeout: int = 10) -> Optional[Dict[str, Any]]:
        assert self.ws is not None, "WebSocket not connected"
        self.ws.settimeout(timeout)
        try:
            raw = self.ws.recv()
            return json.loads(raw)
        except Exception:
            return None

    def _keepalive_loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._send({"type": "KEEPALIVE", "channel": 0})
            except Exception:
                # If we can't send keepalive, assume connection is dead
                return
            time.sleep(self.keepalive_interval)

    def _parse_feed_data(self, msg: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse FEED_DATA messages in COMPACT format based on acceptEventFields.

        Expected shapes (examples):
          {"type":"FEED_DATA","channel":3,"data":["Trade", ["Trade","SPY",559.36, ... , "Trade","BTC/USD",58356.71, ...]]}

        Returns a list of dict rows with keys: eventType, eventSymbol, ...
        """
        out: List[Dict[str, Any]] = []
        data = msg.get("data")
        if not isinstance(data, list) or len(data) < 2:
            return out
        group_type = data[0]
        payload = data[1]
        if not isinstance(group_type, str) or not isinstance(payload, list):
            return out
        # Get field order for this group
        fields = self.accept_event_fields.get(group_type)
        if not fields:
            return out
        k = len(fields)
        # Payload might be a flat list with rows prefixed by event type
        i = 0
        while i + k <= len(payload):
            row = payload[i:i + k]
            # Validate the row starts with a matching eventType label
            if isinstance(row[0], str) and row[0] in self.accept_event_fields:
                # row[0] is the event type; it should equal group_type but accept others just in case
                mapped = {fields[j]: row[j] for j in range(k)}
                out.append(mapped)
                i += k
            else:
                # If payload doesn't include eventType at start, try to synthesize it
                synthesized = [group_type] + row[: k - 1]
                mapped = {fields[j]: synthesized[j] for j in range(k)}
                out.append(mapped)
                i += k
        return out
