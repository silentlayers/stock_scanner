"""
Options scanning via Tastytrade API.

This module is defensive about the response shape from Tastytrade. Some
environments wrap data differently or use different key conventions
(`snake_case` vs `kebab-case`). We therefore:

- Recursively extract option-like dicts from any nested JSON
- Normalize common fields (strike, expiration, delta, dte, bids/asks, ITM%)
- Return normalized items to the app and analysis functions
"""
from __future__ import annotations

from typing import Optional, Tuple, Any, Dict, List
from datetime import datetime, timezone
import requests
from integrations.tastytrade.config import get_api_base_url

# Centralized DTE target for selection logic
DTE_TARGET: int = 45


def fetch_spy_options(session: requests.Session, enrich: Optional[bool] = None, time_budget_s: float | None = None) -> dict:
    """Fetch SPY options using the direct option-chains endpoint (lean).

    Note: `enrich` and `time_budget_s` are accepted for backward compatibility
    and ignored.
    """
    base_url = get_api_base_url()
    url = f'{base_url}/option-chains/SPY'
    resp = session.get(url)
    if resp.status_code != 200:
        raise RuntimeError(
            f"Failed to fetch options: {resp.status_code} {resp.text}")
    raw = resp.json()

    data = raw.get('data') if isinstance(raw, dict) else None
    items = data.get('items') if isinstance(data, dict) else None
    if not (isinstance(items, list) and items):
        # If unexpected shape, just return raw for diagnostics
        return raw

    normalized_direct = [_normalize_option(it) for it in items]
    return {
        'items': normalized_direct,
        'meta': {'source': 'direct'}
    }


def _flatten_nested_chain(nested_json: dict) -> List[Dict[str, Any]]:
    """Flatten a nested option chain to a list of dicts with keys:
    symbol, option_type, strike-price, expiration-date, days-to-expiration.
    """
    results: List[Dict[str, Any]] = []

    # Try to locate expirations collection (list or dict of items)
    expirations = None
    # Common paths: data.expirations, expirations
    for key_path in (("data", "expirations"), ("expirations",)):
        node = nested_json
        try:
            for k in key_path:
                node = node[k]
        except Exception:
            node = None
        if node is not None:
            expirations = node
            break

    if not expirations:
        return results

    # Normalize expirations to list
    if isinstance(expirations, dict):
        # Some APIs return a dict with an 'items' array or keyed by date
        if 'items' in expirations and isinstance(expirations['items'], list):
            expirations_list = expirations['items']
        else:
            expirations_list = list(expirations.values())
    elif isinstance(expirations, list):
        expirations_list = expirations
    else:
        return results

    for exp in expirations_list:
        exp_date = exp.get(
            'expiration-date') or exp.get('expiration_date') or exp.get('expiration')
        dte = exp.get(
            'days-to-expiration') or exp.get('days_to_expiration') or exp.get('dte')
        try:
            dte_int = int(dte) if dte is not None else None
        except Exception:
            dte_int = None
        strikes = exp.get('strikes')
        if not strikes:
            # Some schemas use a list under 'strikes' or an object with 'items'
            if isinstance(exp.get('strikes'), dict) and isinstance(exp['strikes'].get('items'), list):
                strikes = exp['strikes']['items']
        if not strikes:
            continue
        # Normalize strikes to list
        if isinstance(strikes, dict) and 'items' in strikes and isinstance(strikes['items'], list):
            strike_list = strikes['items']
        elif isinstance(strikes, list):
            strike_list = strikes
        else:
            # Possibly dict keyed by strike
            strike_list = list(strikes.values()) if isinstance(
                strikes, dict) else []

        for s in strike_list:
            sp = s.get(
                'strike-price') or s.get('strike_price') or s.get('strike')
            try:
                sp_f = float(sp) if sp is not None else None
            except Exception:
                sp_f = None
            call_sym = s.get('call')
            put_sym = s.get('put')
            call_stream_sym = s.get(
                'call-streamer-symbol') or s.get('call_streamer_symbol')
            put_stream_sym = s.get(
                'put-streamer-symbol') or s.get('put_streamer_symbol')
            if put_sym:
                results.append({
                    'symbol': put_sym,
                    'streamer-symbol': put_stream_sym,
                    'option_type': 'put',
                    'strike-price': sp_f,
                    'expiration-date': exp_date,
                    'days-to-expiration': dte_int,
                })
            if call_sym:
                results.append({
                    'symbol': call_sym,
                    'streamer-symbol': call_stream_sym,
                    'option_type': 'call',
                    'strike-price': sp_f,
                    'expiration-date': exp_date,
                    'days-to-expiration': dte_int,
                })
    return results


def fetch_one_nested_slice(session: requests.Session, symbol: str = 'SPY') -> Dict[str, Any]:
    """Deprecated: previously used for documentation/inspection. No-op lean stub."""
    return {}


def build_nested_doc_view(session: requests.Session, symbol: str = 'SPY') -> Dict[str, Any]:
    """Deprecated: previously for docs. No-op lean stub."""
    return {}


def get_option_streamer_symbols(
    session: requests.Session,
    underlying: str,
    dte_min: int = 30,
    dte_max: int = 60,
    type_filter: str = "put",
    limit: int = 20,
    target_abs_delta_min: Optional[float] = None,
    target_abs_delta_max: Optional[float] = None,
) -> List[str]:
    """Return a list of option streamer-symbols for the given underlying.

    - Fetches the nested option chain and flattens it.
    - Filters by days-to-expiration within [dte_min, dte_max].
    - Filters by option type if provided ("put", "call", or "both").
    - If target_abs_delta_min/max are provided and a delta is available on the
      contract object, bias selection toward that band (in-band first, closest
      to band center), then nearest out-of-band; contracts without delta are
      appended last.
    - Returns up to `limit` streamer symbols (falls back to OCC symbol if needed).
    """
    base_url = get_api_base_url()
    url = f'{base_url}/option-chains/{underlying}/nested'
    resp = session.get(url)
    if resp.status_code != 200:
        raise RuntimeError(
            f"Failed to fetch nested chain: {resp.status_code} {resp.text}")
    nested_json = resp.json()
    flat = _flatten_nested_chain(nested_json)

    def _in_dte_window(v: Any) -> bool:
        try:
            if v is None:
                return False
            iv = int(v)
            return dte_min <= iv <= dte_max
        except Exception:
            return False

    tf = (type_filter or "").strip().lower()
    want_put = tf in ("put", "both", "")
    want_call = tf in ("call", "both") or (tf == "" and not want_put)

    filtered: List[Dict[str, Any]] = []

    if flat:
        for it in flat:
            if not _in_dte_window(it.get('days-to-expiration')):
                continue
            ot = (it.get('option_type') or it.get('option-type') or '').lower()
            if ot == 'put' and not want_put:
                continue
            if ot == 'call' and not want_call:
                continue
            filtered.append(it)
    else:
        # Fallback: direct option-chains endpoint
        url2 = f'{get_api_base_url()}/option-chains/{underlying}'
        r2 = session.get(url2)
        if r2.status_code == 200:
            raw = r2.json()
            items = (raw.get('data') or {}).get(
                'items') if isinstance(raw, dict) else None
            if isinstance(items, list):
                for it in items:
                    # Determine DTE
                    dte_val = _get_dte(it)
                    if not (isinstance(dte_val, int) and dte_min <= dte_val <= dte_max):
                        continue
                    ot = (it.get('option-type') or it.get('option_type')
                          or it.get('type') or '').lower()
                    if ot == 'put' and not want_put:
                        continue
                    if ot == 'call' and not want_call:
                        continue
                    # Try to carry over delta if present in item or nested greeks
                    delta_val = it.get('delta')
                    if delta_val is None:
                        delta_val = ((it.get('greeks') or {}).get('delta')) if isinstance(
                            it.get('greeks'), dict) else None
                    if delta_val is None:
                        og = it.get('option_greeks')
                        if isinstance(og, dict):
                            delta_val = og.get('delta')
                    filtered.append({
                        'symbol': it.get('symbol'),
                        'streamer-symbol': it.get('streamer-symbol') or it.get('streamer_symbol'),
                        'option_type': ot,
                        'days-to-expiration': dte_val,
                        'delta': delta_val,
                    })

    # Helper to pull delta if present
    def _get_abs_delta(itm: Dict[str, Any]) -> Optional[float]:
        d = itm.get('delta')
        if d is None:
            g = itm.get('greeks') or itm.get('option_greeks') or {}
            d = g.get('delta')
        try:
            return abs(float(d)) if d is not None else None
        except Exception:
            return None

    # Optionally score contracts by proximity to target abs(delta)
    scored: List[Tuple[int, float, Dict[str, Any]]] = []
    use_band = (
        isinstance(target_abs_delta_min, (int, float)) and
        isinstance(target_abs_delta_max, (int, float)) and
        target_abs_delta_min <= target_abs_delta_max
    )
    band_center: Optional[float] = None
    min_f: Optional[float] = None
    max_f: Optional[float] = None
    if use_band:
        # mypy/pylance: we proved not-None in use_band, so cast to float
        min_f = float(target_abs_delta_min)  # type: ignore[arg-type]
        max_f = float(target_abs_delta_max)  # type: ignore[arg-type]
        band_center = (min_f + max_f) / 2.0

    for it in filtered:
        if use_band:
            ad = _get_abs_delta(it)
            if ad is None:
                # Put unknown-delta items at the end with a neutral score
                scored.append((2, 1e9, it))
                continue
            # in-band first (rank 0), then out-of-band (rank 1)
            # with min_f/max_f present, determine band membership
            if min_f is not None and max_f is not None:
                in_band = 1 if (ad < min_f or ad > max_f) else 0
            else:
                in_band = 0
            # distance to band center (smaller is better)
            dist = abs(ad - float(band_center)
                       ) if band_center is not None else 0.0
            scored.append((in_band, dist, it))
        else:
            scored.append((0, 0.0, it))

    # Sort by in-band rank then distance
    scored.sort(key=lambda t: (t[0], t[1]))

    # Prefer streamer-symbol; fall back to OCC symbol if missing; preserve uniqueness
    symbols: List[str] = []
    for _rank, _dist, it in scored:
        ss = it.get('streamer-symbol') or it.get('streamer_symbol')
        sym = ss or it.get('symbol')
        if sym and str(sym) not in symbols:
            symbols.append(str(sym))
        if len(symbols) >= limit:
            break
    return symbols


def _fetch_quotes_for_symbols(session: requests.Session, symbols: List[str], deadline: Optional[float] = None) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    """Deprecated: quotes enrichment removed for lean build. Returns empty."""
    return {}, []


def _enrich_flat_with_quotes(flat: List[Dict[str, Any]], quotes: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deprecated: quotes enrichment removed."""
    return flat


# -----------------------------
# Extraction & normalization
# -----------------------------

OptionDict = Dict[str, Any]


def _looks_like_option(node: dict) -> bool:
    """Heuristic to determine if a dict resembles an option contract."""
    keys = set(node.keys())
    indicators = {
        'option_type', 'option-type', 'type',
        'strike_price', 'strike-price', 'strike',
        'expiration_date', 'expiration-date', 'expiration',
        'days_to_expiration', 'days-to-expiration', 'dte',
        'delta', 'bid', 'ask', 'mark_price', 'mark-price'
    }
    return bool(keys & indicators)


def _extract_option_items(obj: Any, path: str = "$") -> Tuple[List[OptionDict], List[str]]:
    """Recursively collect option-like dicts from any JSON structure.

    Returns (items, paths) where paths contains JSON pointer-like strings
    to where option-like dicts were found, useful for debugging.
    """
    found: List[OptionDict] = []
    locations: List[str] = []

    if isinstance(obj, dict):
        if _looks_like_option(obj):
            found.append(obj)
            locations.append(path)
        # Continue exploring values
        for k, v in obj.items():
            sub_items, sub_paths = _extract_option_items(v, f"{path}/{k}")
            if sub_items:
                found.extend(sub_items)
                locations.extend(sub_paths)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            sub_items, sub_paths = _extract_option_items(v, f"{path}[{i}]")
            if sub_items:
                found.extend(sub_items)
                locations.extend(sub_paths)
    return found, locations


def _first_non_null(*values: Any) -> Any:
    for v in values:
        if v is not None:
            return v
    return None


def _to_float(val: Any) -> Optional[float]:
    try:
        return float(val)
    except Exception:
        return None


def _normalize_option(raw: OptionDict) -> OptionDict:
    """Normalize an option dict to a common schema, while preserving
    both snake_case and kebab-case aliases for UI compatibility.
    """
    opt_type = _first_non_null(
        raw.get('option_type'), raw.get('option-type'), raw.get('type'))
    # Canonicalize to 'put'/'call' when possible
    if opt_type is not None:
        opt_l = str(opt_type).lower()
        if opt_l in ("p", "put"):
            opt_type = "put"
        elif opt_l in ("c", "call"):
            opt_type = "call"

    strike_price = _first_non_null(
        raw.get('strike_price'), raw.get('strike-price'), raw.get('strike'))
    strike_price = _to_float(strike_price)

    expiration_date = _first_non_null(
        raw.get('expiration_date'), raw.get('expiration-date'), raw.get('expiration'))

    dte = _first_non_null(
        raw.get('days_to_expiration'), raw.get('days-to-expiration'), raw.get('dte'))
    try:
        dte = int(dte) if dte is not None else None
    except Exception:
        dte = None

    # Delta may be nested under greeks
    delta = _to_float(
        _first_non_null(
            raw.get('delta'),
            (raw.get('greeks') or {}).get('delta'),
            (raw.get('option_greeks') or {}).get('delta')
        )
    )

    bid = _first_non_null(raw.get('bid'), raw.get(
        'best_bid'), raw.get('bid-price'))
    bid = _to_float(bid)
    ask = _first_non_null(raw.get('ask'), raw.get(
        'best_ask'), raw.get('ask-price'))
    ask = _to_float(ask)
    mark_price = _first_non_null(
        raw.get('mark_price'), raw.get('mark-price'), raw.get('mid'))
    mark_price = _to_float(mark_price)
    if mark_price is None and (bid is not None and ask is not None):
        mark_price = round((bid + ask) / 2, 4)

    itm_percent = _first_non_null(
        raw.get('itm_percent'),
        raw.get('probability_itm'), raw.get('probability-itm'),
        (raw.get('greeks') or {}).get('probability-itm'),
        (raw.get('greeks') or {}).get('probability_itm'),
    )
    itm_percent = _to_float(itm_percent)

    norm: OptionDict = {
        'option_type': opt_type,
        'strike_price': strike_price,
        'expiration_date': expiration_date,
        'days_to_expiration': dte,
        'delta': delta,
        'bid': bid,
        'ask': ask,
        'mark_price': mark_price,
        'itm_percent': itm_percent,
    }
    # Preserve identifiers when present for downstream quote/stream usage
    if 'symbol' in raw:
        try:
            norm['symbol'] = str(raw.get('symbol'))
        except Exception:
            pass
    if 'streamer-symbol' in raw or 'streamer_symbol' in raw:
        try:
            norm['streamer-symbol'] = raw.get(
                'streamer-symbol') or raw.get('streamer_symbol')
        except Exception:
            pass
    # Aliases for UI
    norm['strike-price'] = norm['strike_price']
    norm['expiration-date'] = norm['expiration_date']
    norm['option-type'] = norm['option_type']
    return norm


# -----------------------------
# Helpers for analysis
# -----------------------------

def _canonical_option_type(val: Any) -> Optional[str]:
    if val is None:
        return None
    tl = str(val).strip().lower()
    if tl in ('p', 'put') or tl.startswith('p'):
        return 'put'
    if tl in ('c', 'call') or tl.startswith('c'):
        return 'call'
    return tl


def _get_opt_type(opt: Dict[str, Any]) -> Optional[str]:
    t = opt.get('option_type')
    if t is None:
        t = opt.get('option-type') or opt.get('type')
    return _canonical_option_type(t)


def _get_dte(opt: Dict[str, Any]) -> int:
    v = opt.get('days_to_expiration')
    if v is None:
        v = opt.get('days-to-expiration') or opt.get('dte')
    try:
        iv = int(v) if v is not None else 0
        if iv:
            return iv
    except Exception:
        pass
    # Fallback: derive from expiration date if available
    exp = opt.get('expiration_date') or opt.get(
        'expiration-date') or opt.get('expires-at')
    if isinstance(exp, str) and exp:
        try:
            # Normalize Z suffix
            iso = exp.replace('Z', '+00:00')
            dt = datetime.fromisoformat(iso)
            # Use UTC date to avoid TZ surprises
            today = datetime.now(timezone.utc).date()
            dte_days = (dt.date() - today).days
            return max(dte_days, 0)
        except Exception:
            return 0
    return 0


def _get_normalized_items(options_json: dict) -> Tuple[List[OptionDict], Dict[str, Any]]:
    """Return normalized option items with debug metadata."""
    # Common direct paths first
    direct_items: List[OptionDict] = []
    used_path: Optional[tuple] = None
    for key_path in (
        ('items',),
        ('data', 'items'),
        ('data', 'options'),
        ('options',),
        ('contracts',),
        ('chain', 'items'),
    ):
        node = options_json
        try:
            for k in key_path:
                node = node[k]
        except Exception:
            node = None
        if isinstance(node, list) and node:
            direct_items = node
            used_path = key_path
            break

    # If not found in common paths, recursively search entire JSON
    search_items: List[OptionDict] = []
    search_paths: List[str] = []
    if not direct_items:
        search_items, search_paths = _extract_option_items(options_json)

    raw_items = direct_items if direct_items else search_items
    normalized = [_normalize_option(it) for it in raw_items]

    # Add raw field samples for diagnostics
    raw_type_samples = []
    raw_dte_samples = []
    for it in raw_items[:25]:
        raw_type_samples.append(
            it.get('option_type') or it.get('option-type') or it.get('type')
        )
        raw_dte_samples.append(
            it.get('days_to_expiration') or it.get(
                'days-to-expiration') or it.get('dte')
        )

    debug = {
        'source': 'direct' if direct_items else 'recursive',
        'direct_path_used': used_path,
        'found_count_raw': len(raw_items),
        'found_paths': search_paths[:25],
        'sample_fields': list(raw_items[0].keys()) if raw_items else [],
        'raw_type_samples': raw_type_samples,
        'raw_dte_samples': raw_dte_samples,
    }
    return normalized, debug


def analyze_option_deltas(options_json: dict) -> dict:
    """Analyze and display ITM% values from Tastytrade API."""
    items, extract_debug = _get_normalized_items(options_json)
    all_puts = [opt for opt in items if _get_opt_type(opt) == 'put']

    # Debug: Show what we have before filtering
    # Make comparable DTE list for debug (ints only)
    _dtes_raw = [_get_dte(p) for p in all_puts]
    _dtes_int = [int(v) for v in _dtes_raw if isinstance(v, (int, float))]
    # Type counts for quick diagnostics
    type_counts: Dict[str, int] = {}
    for it in items:
        t = _get_opt_type(it) or 'unknown'
        type_counts[t] = type_counts.get(t, 0) + 1

    norm_type_samples = [(_get_opt_type(it) or 'unknown') for it in items[:25]]
    debug_info = {
        "total_items": len(items),
        "total_puts": len(all_puts),
        "dte_values": sorted(list(set(_dtes_int)))[:25],
        "option_types": sorted(list({(_get_opt_type(o) or 'unknown') for o in items})),
        "extraction": extract_debug,
        "type_counts": type_counts,
        "sample_fields": (list(items[0].keys()) if items else []),
        "normalized_type_samples": norm_type_samples,
    }
    if isinstance(options_json, dict) and 'meta' in options_json:
        debug_info['source_meta'] = options_json.get('meta')

    # Enforce exactly DTE_TARGET
    puts = [opt for opt in all_puts if _get_dte(opt) == DTE_TARGET]

    if not puts:
        return {
            "error": "No puts found at DTE = 45",
            "debug_info": debug_info
        }

    # Sort puts by strike price for easier analysis
    puts.sort(key=lambda x: (x.get('strike_price')
              or x.get('strike-price') or 0))

    analysis = {
        "total_puts": len(puts),
        "delta_analysis": [],
        "summary": {},
        "sample_fields": list(puts[0].keys()) if puts else [],
        "debug_info": debug_info
    }

    for put in puts:
        delta_val = put.get('delta')
        delta = float(delta_val) if isinstance(
            delta_val, (int, float)) else None
        strike = float(put.get('strike_price') or put.get('strike-price') or 0)
        dte = _get_dte(put)

        # Use actual ITM% from Tastytrade if available, otherwise calculate from delta
        itm_val = put.get('itm_percent', put.get('probability_itm'))
        if itm_val is None and delta is not None:
            itm_val = abs(delta) * 100.0
        itm_percent = float(itm_val) if isinstance(
            itm_val, (int, float)) else None

        abs_delta = abs(delta) if delta is not None else 0.0
        moneyness = "OTM"
        if delta is not None:
            moneyness = "ITM" if delta < - \
                0.5 else ("ATM" if delta < -0.3 else "OTM")

        analysis["delta_analysis"].append({
            "strike": strike,
            "delta": delta if delta is not None else 0.0,
            "abs_delta": abs_delta,
            "itm_percent": round(itm_percent, 1) if itm_percent is not None else None,
            "dte": dte,
            "moneyness": moneyness,
        })

    # Find options in our target ranges
    target_sell = [opt for opt in analysis["delta_analysis"]
                   if 0.45 <= opt["abs_delta"] <= 0.55]
    target_buy = [opt for opt in analysis["delta_analysis"]
                  if 0.20 <= opt["abs_delta"] <= 0.30]

    # Compute delta range from computed rows to avoid None issues
    abs_deltas = [row.get("abs_delta") for row in analysis["delta_analysis"]
                  if isinstance(row.get("abs_delta"), (int, float))]
    if abs_deltas:
        delta_range_str = f"{min(abs_deltas):.3f} to {max(abs_deltas):.3f}"
    else:
        delta_range_str = "N/A"

    analysis["summary"] = {
        "target_sell_deltas": target_sell,
        "target_buy_deltas": target_buy,
        "delta_range": delta_range_str,
        "note": f"DTE fixed at {DTE_TARGET}. For puts: Delta -0.5 ≈ 50% ITM, Delta -0.3 ≈ 30% ITM, Delta -0.2 ≈ 20% ITM"
    }

    return analysis


def find_put_credit_spread(
    options_json: dict,
    session: Optional[requests.Session] = None,
    dxlink_limit: int = 200,
    dxlink_timeout: float = 10.0,
    dte_min: int = 30,
    dte_max: int = 45,
    sell_delta_target: float = 0.50,
    buy_delta_target: float = 0.25,
) -> Optional[Tuple[dict, dict, float, float, float, float, int]]:
    """Pick a put credit spread using live Greeks with DTE range support.

    Requirements:
    - Live Greeks via DXLink are required (no REST/chain fallback for delta).
    - DTE in range [dte_min, dte_max].
    - Targets abs(delta): sell ~sell_delta_target, buy ~buy_delta_target; same expiration; buy strike < sell strike.

    Returns: (sell_leg, buy_leg, spread_width, credit, max_loss, ror, dte) or None
    """
    # Normalize and filter to DTE range puts
    items, _dbg = _get_normalized_items(options_json)
    puts = [
        opt for opt in items
        if _get_opt_type(opt) == 'put' and dte_min <= _get_dte(opt) <= dte_max
    ]
    if not puts:
        return None

    # Live Greeks are mandatory
    if session is None:
        return None
    live_abs_delta: Dict[str, float] = {}
    # Store bid/ask/mark for each symbol
    live_quotes: Dict[str, Dict[str, Any]] = {}
    try:
        from integrations.dxlink.snapshot import get_option_snapshot
        # Order by strike, gather unique streamer symbols, cap request size

        def _strike(o: Dict[str, Any]) -> float:
            try:
                return float(o.get('strike_price') or o.get('strike-price') or 0)
            except Exception:
                return 0.0
        # Start from highest strikes (closest to ATM)
        puts_sorted = sorted(puts, key=_strike, reverse=True)
        seen: set[str] = set()
        syms: List[str] = []
        for p in puts_sorted:
            ss = p.get('streamer-symbol') or p.get('streamer_symbol')
            if isinstance(ss, str) and ss and ss not in seen:
                seen.add(ss)
                syms.append(ss)
            if len(syms) >= int(dxlink_limit):
                break
        if syms:
            snap = get_option_snapshot(
                session, syms, timeout=int(dxlink_timeout))
            if isinstance(snap, dict):
                for s, parts in snap.items():
                    # Extract Greeks
                    g = (parts or {}).get('greeks') or {}
                    d = g.get('delta')
                    try:
                        if d is not None:
                            live_abs_delta[s] = abs(float(d))
                    except Exception:
                        pass

                    # Extract pricing data
                    q = (parts or {}).get('quote') or {}
                    try:
                        bid = q.get('bidPrice')
                        ask = q.get('askPrice')
                        mark = q.get('mark')
                        live_quotes[s] = {
                            'bid': float(bid) if bid is not None else None,
                            'ask': float(ask) if ask is not None else None,
                            'mark': float(mark) if mark is not None else None,
                        }
                    except Exception:
                        pass
        if not live_abs_delta:
            return None
    except Exception:
        return None

    # Choose sell leg nearest to sell_delta_target abs(delta)
    candidates: List[Tuple[float, float, Dict[str, Any]]] = []
    for o in puts:
        ss = o.get('streamer-symbol') or o.get('streamer_symbol')
        ad = live_abs_delta.get(ss) if isinstance(ss, str) else None
        if ad is None:
            continue
        strike_f = float(o.get('strike_price') or o.get('strike-price') or 0)
        candidates.append((abs(ad - sell_delta_target), -strike_f, o))
    if not candidates:
        return None
    candidates.sort(key=lambda t: (t[0], t[1]))
    sell_leg = candidates[0][2]

    # Add the delta to sell_leg for display (negative for puts)
    sell_ss = sell_leg.get(
        'streamer-symbol') or sell_leg.get('streamer_symbol')
    if isinstance(sell_ss, str):
        if sell_ss in live_abs_delta:
            sell_leg['delta'] = -live_abs_delta[sell_ss]  # Negative for puts
        # Add pricing data from live quotes
        if sell_ss in live_quotes:
            quotes = live_quotes[sell_ss]
            if quotes.get('bid') is not None:
                sell_leg['bid'] = quotes['bid']
            if quotes.get('ask') is not None:
                sell_leg['ask'] = quotes['ask']
            if quotes.get('mark') is not None:
                sell_leg['mark_price'] = quotes['mark']

    # Restrict buy candidates to same expiration and lower strike than sell
    sell_exp = sell_leg.get(
        'expiration_date') or sell_leg.get('expiration-date')
    sell_strike = float(sell_leg.get('strike_price')
                        or sell_leg.get('strike-price') or 0)
    buy_candidates: List[Tuple[float, float, Dict[str, Any]]] = []
    for o in puts:
        exp = o.get('expiration_date') or o.get('expiration-date')
        strike = float(o.get('strike_price') or o.get('strike-price') or 0)
        if exp != sell_exp or strike >= sell_strike:
            continue
        ss = o.get('streamer-symbol') or o.get('streamer_symbol')
        ad = live_abs_delta.get(ss) if isinstance(ss, str) else None
        if ad is None:
            continue
        buy_candidates.append((abs(ad - buy_delta_target), -strike, o))
    if not buy_candidates:
        return None
    buy_candidates.sort(key=lambda t: (t[0], t[1]))
    buy_leg = buy_candidates[0][2]

    # Add the delta to buy_leg for display (negative for puts)
    buy_ss = buy_leg.get('streamer-symbol') or buy_leg.get('streamer_symbol')
    if isinstance(buy_ss, str):
        if buy_ss in live_abs_delta:
            buy_leg['delta'] = -live_abs_delta[buy_ss]  # Negative for puts
        # Add pricing data from live quotes
        if buy_ss in live_quotes:
            quotes = live_quotes[buy_ss]
            if quotes.get('bid') is not None:
                buy_leg['bid'] = quotes['bid']
            if quotes.get('ask') is not None:
                buy_leg['ask'] = quotes['ask']
            if quotes.get('mark') is not None:
                buy_leg['mark_price'] = quotes['mark']

    # Get DTE from sell leg
    sell_dte = _get_dte(sell_leg)

    spread_width = abs(
        (sell_leg.get('strike_price') or sell_leg.get('strike-price') or 0) -
        (buy_leg.get('strike_price') or buy_leg.get('strike-price') or 0)
    )
    # Compute credit using mark_price if available, else mid of bid/ask

    def _mark(o: Dict[str, Any]) -> float:
        mp = o.get('mark_price')
        if isinstance(mp, (int, float)):
            return float(mp)
        b = o.get('bid') or o.get('bid-price')
        a = o.get('ask') or o.get('ask-price')
        try:
            if b is not None and a is not None:
                return (float(b) + float(a)) / 2.0
        except Exception:
            pass
        return 0.0
    credit = _mark(sell_leg) - _mark(buy_leg)
    # Multiply by 100 for per-contract amount
    max_loss = (spread_width - credit) * 100
    # Return as percentage
    ror = (credit / (spread_width - credit) *
           100.0) if (spread_width - credit) > 0 else 0.0
    return sell_leg, buy_leg, spread_width, credit, max_loss, ror, sell_dte


def find_bull_call_debit_spread(
    options_json: dict,
    session: Optional[requests.Session] = None,
    dxlink_limit: int = 80,
    dxlink_timeout: float = 6.0,
    dte_min: int = 30,
    dte_max: int = 45,
    buy_delta_target: float = 0.40,
    sell_delta_target: float = 0.20,
) -> Optional[Tuple[dict, dict, float, float, float, float, int]]:
    """Pick a bull call debit spread using live Greeks.

    Strategy: Buy lower strike call (higher delta), sell higher strike call (lower delta).

    Requirements:
    - Live Greeks via DXLink are required (no REST/chain fallback for delta).
    - DTE in range [dte_min, dte_max] (default 30-45 days).
    - Targets delta: buy ~0.40, sell ~0.20; same expiration; buy strike < sell strike.

    Returns: (buy_leg, sell_leg, spread_width, debit, max_gain, avg_return_pct, dte) or None
    """
    # Normalize and filter to calls in DTE range
    items, _dbg = _get_normalized_items(options_json)
    calls = [
        opt for opt in items
        if _get_opt_type(opt) == 'call' and dte_min <= _get_dte(opt) <= dte_max
    ]
    if not calls:
        return None

    # Live Greeks are mandatory
    if session is None:
        return None
    live_delta: Dict[str, float] = {}
    try:
        from integrations.dxlink.snapshot import get_option_snapshot

        def _strike(o: Dict[str, Any]) -> float:
            try:
                return float(o.get('strike_price') or o.get('strike-price') or 0)
            except Exception:
                return 0.0

        calls_sorted = sorted(calls, key=_strike)
        seen: set[str] = set()
        syms: List[str] = []
        for c in calls_sorted:
            ss = c.get('streamer-symbol') or c.get('streamer_symbol')
            if isinstance(ss, str) and ss and ss not in seen:
                seen.add(ss)
                syms.append(ss)
            if len(syms) >= max(10, int(dxlink_limit)):
                break
        if syms:
            snap = get_option_snapshot(
                session, syms, timeout=int(dxlink_timeout))
            if isinstance(snap, dict):
                for s, parts in snap.items():
                    g = (parts or {}).get('greeks') or {}
                    d = g.get('delta')
                    try:
                        if d is not None:
                            # Keep sign (positive for calls)
                            live_delta[s] = float(d)
                    except Exception:
                        pass
        if not live_delta:
            return None
    except Exception:
        return None

    # Choose buy leg (long call) nearest to buy_delta_target (e.g., 0.40)
    buy_candidates: List[Tuple[float, float, Dict[str, Any]]] = []
    for o in calls:
        ss = o.get('streamer-symbol') or o.get('streamer_symbol')
        d = live_delta.get(ss) if isinstance(ss, str) else None
        if d is None or d <= 0:  # Calls should have positive delta
            continue
        strike_f = float(o.get('strike_price') or o.get('strike-price') or 0)
        buy_candidates.append((abs(d - buy_delta_target), strike_f, o))
    if not buy_candidates:
        return None
    # Closest delta, then lower strike
    buy_candidates.sort(key=lambda t: (t[0], t[1]))
    buy_leg = buy_candidates[0][2]

    # Restrict sell candidates to same expiration and higher strike than buy
    buy_exp = buy_leg.get('expiration_date') or buy_leg.get('expiration-date')
    buy_strike = float(buy_leg.get('strike_price')
                       or buy_leg.get('strike-price') or 0)
    buy_dte = _get_dte(buy_leg)

    sell_candidates: List[Tuple[float, float, Dict[str, Any]]] = []
    for o in calls:
        exp = o.get('expiration_date') or o.get('expiration-date')
        strike = float(o.get('strike_price') or o.get('strike-price') or 0)
        if exp != buy_exp or strike <= buy_strike:
            continue
        ss = o.get('streamer-symbol') or o.get('streamer_symbol')
        d = live_delta.get(ss) if isinstance(ss, str) else None
        if d is None or d <= 0:
            continue
        sell_candidates.append((abs(d - sell_delta_target), strike, o))
    if not sell_candidates:
        return None
    # Closest delta, then lower strike
    sell_candidates.sort(key=lambda t: (t[0], t[1]))
    sell_leg = sell_candidates[0][2]

    spread_width = abs(
        (sell_leg.get('strike_price') or sell_leg.get('strike-price') or 0) -
        (buy_leg.get('strike_price') or buy_leg.get('strike-price') or 0)
    )

    # Compute debit using mark_price if available, else mid of bid/ask
    def _mark(o: Dict[str, Any]) -> float:
        mp = o.get('mark_price')
        if isinstance(mp, (int, float)):
            return float(mp)
        b = o.get('bid') or o.get('bid-price')
        a = o.get('ask') or o.get('ask-price')
        try:
            if b is not None and a is not None:
                return (float(b) + float(a)) / 2.0
        except Exception:
            pass
        return 0.0

    debit = _mark(buy_leg) - _mark(sell_leg)  # Cost to enter (positive)
    max_gain = spread_width - debit
    avg_return_pct = (max_gain / debit * 100.0) if debit > 0 else 0.0

    return buy_leg, sell_leg, spread_width, debit, max_gain, avg_return_pct, buy_dte
