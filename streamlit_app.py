from __future__ import annotations

import time as time_module
import streamlit as st
import pandas as pd
import altair as alt
import datetime
from datetime import datetime as dt
from zoneinfo import ZoneInfo
import yfinance as yf

from integrations.tastytrade.auth import authenticate
from core.data import download_symbol, get_close_series
from core.indicators import compute_indicators
from core.market_signal import signal_diagnostics
from options.scanner import fetch_spy_options, find_put_credit_spread
from options.backtest import run_backtest, run_backtest_bull_call
from integrations.dxlink.snapshot import get_symbol_price_snapshot
from integrations.tastytrade.token_manager import clear_saved_token
from core.safety import SafetyManager, SafetyLimits, DryRunManager
from core.automation import AutomationEngine, check_signal_now, can_trade_now, get_current_market_snapshot
import os


st.set_page_config(page_title="Stock Scanner", layout="wide")

# Mobile-optimized CSS
st.markdown("""
<style>
    /* Mobile optimization */
    @media only screen and (max-width: 768px) {
        /* Reduce padding on mobile */
        .main .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
            padding-top: 1rem;
            max-width: 100%;
        }
        
        /* Make title smaller on mobile */
        h1 {
            font-size: 1.5rem !important;
        }
        
        h2 {
            font-size: 1.3rem !important;
        }
        
        h3 {
            font-size: 1.1rem !important;
        }
        
        /* Make tables scrollable horizontally */
        .dataframe {
            font-size: 0.8rem;
            overflow-x: auto;
            display: block;
        }
        
        /* Improve button sizing for mobile */
        .stButton button {
            width: 100%;
            font-size: 0.9rem;
        }
        
        /* Make metrics more compact */
        [data-testid="stMetricValue"] {
            font-size: 1.2rem;
        }
        
        [data-testid="stMetricLabel"] {
            font-size: 0.8rem;
        }
        
        /* Reduce chart heights on mobile */
        .vega-embed {
            max-height: 300px;
        }
        
        /* Faster rendering - reduce animations */
        * {
            transition: none !important;
            animation: none !important;
        }
        
        /* Hide heavy elements initially */
        .stSpinner {
            min-height: 100px;
        }
        
        /* Make tabs scrollable */
        .stTabs [data-baseweb="tab-list"] {
            overflow-x: auto;
            flex-wrap: nowrap;
        }
        
        /* Compact expanders */
        .streamlit-expanderHeader {
            font-size: 0.9rem;
        }
        
        /* Better input fields on mobile */
        .stTextInput input, .stNumberInput input {
            font-size: 16px !important; /* Prevents zoom on iOS */
        }
        
        /* Stack columns vertically on mobile */
        [data-testid="column"] {
            width: 100% !important;
            flex: 1 1 100% !important;
        }
    }
    
    /* Tablet optimization */
    @media only screen and (min-width: 769px) and (max-width: 1024px) {
        .main .block-container {
            padding-left: 2rem;
            padding-right: 2rem;
        }
    }
    
    /* General improvements for all screen sizes */
    /* Better scrolling on tables */
    .dataframe-container {
        overflow-x: auto;
    }
    
    /* Ensure charts are responsive */
    .vega-embed {
        width: 100% !important;
    }
    
    /* Prevent captions from floating into charts */
    .element-container:has(> .stCaption) {
        position: relative;
        clear: both;
        margin-top: 1rem;
    }
    
    /* Add spacing before horizontal dividers */
    hr {
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("Stock Scanner")

# Detect mobile device


def is_mobile():
    """Detect if user is on mobile based on viewport width"""
    # Use JavaScript to detect screen width
    mobile_check = st.query_params.get('mobile', 'false')
    return mobile_check == 'true'


# Auto-refresh every 30 seconds during market hours (60 seconds on mobile for better performance)
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = time_module.time()

refresh_interval = 60 if is_mobile() else 30  # Longer refresh on mobile
if time_module.time() - st.session_state.last_refresh > refresh_interval:
    st.session_state.last_refresh = time_module.time()
    st.rerun()


# cache downloads for 15 minutes, hide spinner
@st.cache_data(ttl=900, show_spinner=False)
def load_market_data():
    spy = download_symbol('SPY', period='200d',
                          interval='1d', auto_adjust=False)
    vix = download_symbol('^VIX', period='200d',
                          interval='1d', auto_adjust=False)
    spy_close = get_close_series(spy, 'SPY')
    vix_close = get_close_series(vix, '^VIX')
    inds = compute_indicators(spy_close)
    return spy, vix, spy_close, vix_close, inds


def is_running_on_cloud():
    """Detect if running on Streamlit Cloud"""
    # Allow manual override for testing
    if os.getenv('FORCE_CLOUD_AUTH', 'false').lower() in ('true', '1', 'yes'):
        return True

    # Check multiple indicators of Streamlit Cloud
    hostname = os.getenv('HOSTNAME', '')
    cloud_indicators = [
        os.getenv('STREAMLIT_SHARING_MODE') is not None,
        # Changed from 'streamlit-' to 'streamlit'
        hostname.startswith('streamlit'),
        os.getenv('IS_STREAMLIT_CLOUD') is not None,
        'streamlit.app' in hostname,
        os.getenv('STREAMLIT_RUNTIME_ENVIRONMENT') == 'cloud',
    ]

    return any(cloud_indicators)


def ensure_session():
    if 'auth_session' not in st.session_state or st.session_state['auth_session'] is None:
        # Use cloud-friendly auth if on cloud, local auth otherwise
        on_cloud = is_running_on_cloud()

        # Debug info - show environment variables
        with st.expander("🔍 Debug: Environment Detection", expanded=False):
            st.write("**Cloud Detection Result:**", on_cloud)
            st.write("**Environment Variables:**")
            st.json({
                'STREAMLIT_SHARING_MODE': os.getenv('STREAMLIT_SHARING_MODE'),
                'HOSTNAME': os.getenv('HOSTNAME'),
                'IS_STREAMLIT_CLOUD': os.getenv('IS_STREAMLIT_CLOUD'),
                'STREAMLIT_RUNTIME_ENVIRONMENT': os.getenv('STREAMLIT_RUNTIME_ENVIRONMENT'),
                'FORCE_CLOUD_AUTH': os.getenv('FORCE_CLOUD_AUTH'),
            })

        # Debug info (can remove later)
        if on_cloud:
            st.caption("🌐 Detected: Streamlit Cloud - Using URL-based OAuth")
        else:
            st.caption(
                "💻 Detected: Local environment - Using callback server OAuth")

        if on_cloud:
            from integrations.tastytrade.cloud_auth import authenticate_cloud

            # Try to get auth from URL query params or show auth link
            result = authenticate_cloud()
            if result is not None:
                session, token = result
                st.session_state['auth_session'] = session
                st.session_state['oauth_token'] = token
                st.rerun()
            else:
                # Still waiting for authorization
                return False
        else:
            # Local authentication with callback server
            st.warning(
                "You need to authorize with Tastytrade to fetch options data.")

            # Check for existing token first
            from integrations.tastytrade.token_manager import get_persistent_session
            existing = get_persistent_session()
            if existing:
                if st.button("Use Saved Token"):
                    session, token = existing
                    st.session_state['auth_session'] = session
                    st.session_state['oauth_token'] = token
                    st.success("✅ Using saved token!")
                    st.rerun()
                st.info(
                    "💡 Found a saved token. Click above to use it, or authorize again below.")

            col1, col2 = st.columns([2, 1])
            with col1:
                if st.button("Authorize Tastytrade"):
                    # Generate auth URL manually
                    from integrations.tastytrade.config import get_oauth_settings, get_oauth_scopes
                    from requests_oauthlib import OAuth2Session

                    authorization_base_url, token_url, client_id, client_secret, redirect_uri = get_oauth_settings()
                    oauth = OAuth2Session(
                        client_id, redirect_uri=redirect_uri, scope=get_oauth_scopes())
                    auth_url, state = oauth.authorization_url(
                        authorization_base_url)

                    st.markdown("### 🔐 Manual Authorization")
                    st.info("**Copy this URL and open it in your browser:**")
                    st.code(auth_url, language=None)
                    st.link_button("🌐 Or Click Here to Open", auth_url)
                    st.warning(
                        "⚠️ After authorizing, the app will attempt to catch the callback automatically. Keep this window open!")

                    # Try to start the auth flow in background
                    try:
                        with st.spinner("Waiting for authorization... (check your browser)"):
                            session, token = authenticate()
                            st.session_state['auth_session'] = session
                            st.session_state['oauth_token'] = token
                            st.success("✅ Authorized successfully!")
                            st.rerun()
                    except Exception as e:
                        st.error(f"❌ Authentication failed: {e}")
                        st.info(
                            "💡 The browser callback may have failed. Try using the cloud deployment instead.")
            with col2:
                if st.button("Clear Saved Token", help="Remove saved authentication token"):
                    clear_saved_token()
                    st.info("Saved token cleared. You'll need to re-authorize.")
        return False
    return True


@st.cache_data(ttl=20)
def get_live_spy_quote():
    """Fetch latest intraday SPY price (1m). Returns dict with timestamp and price or None."""
    try:
        df = yf.download('SPY', period='1d', interval='1m',
                         auto_adjust=False, progress=False)
        if df is None or df.empty:
            return None
        ts = df.index[-1]
        # Extract scalar value and convert to Python float
        price = float(df['Close'].values[-1].item())
        return {"timestamp": ts, "price": price}
    except Exception:
        return None


@st.cache_data(ttl=20)
def get_live_vix_quote():
    """Fetch latest intraday VIX price (1m). Returns dict with timestamp and price or None."""
    try:
        df = yf.download('^VIX', period='1d', interval='1m',
                         auto_adjust=False, progress=False)
        if df is None or df.empty:
            return None
        ts = df.index[-1]
        # Extract scalar value and convert to Python float
        price_val = df['^VIX'].values[-1] if '^VIX' in df.columns else df['Close'].values[-1]
        price = float(price_val.item())
        return {"timestamp": ts, "price": price}
    except Exception:
        return None


def is_us_equity_market_open(now_et: dt | None = None) -> bool:
    from datetime import time as dt_time
    now_et = now_et or dt.now(ZoneInfo('America/New_York'))
    # Basic hours (excludes holidays): 9:30–16:00 ET, Mon–Fri
    if now_et.weekday() >= 5:
        return False
    t = now_et.time()
    return dt_time(9, 30) <= t <= dt_time(16, 0)


@st.cache_data(ttl=86400)  # Cache for 24 hours (updates monthly)
def get_shiller_pe():
    """Fetch the Shiller PE (CAPE) ratio from FRED. Returns float or None."""
    try:
        import pandas_datareader as pdr
        # Fetch Shiller PE ratio from FRED (updates monthly)
        cape = pdr.data.DataReader(
            'MULTPL/SHILLER_PE_RATIO_MONTH', 'fred', start='2020-01-01')
        if cape is None or cape.empty:
            return None
        # Get the most recent value
        latest_value = cape.iloc[-1, 0]
        return float(latest_value)
    except Exception:
        # Fallback: try to get from a simple CSV endpoint
        try:
            import requests
            # Alternative source: Robert Shiller's data (Yale)
            url = 'https://www.multpl.com/shiller-pe/table/by-month'
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                # Parse the HTML table (first row has the latest value)
                df = pd.read_html(response.text)[0]
                if not df.empty and len(df.columns) >= 2:
                    latest_value = df.iloc[0, 1]
                    return float(str(latest_value).replace(',', ''))
        except Exception:
            pass
        return None


tab_signal, tab_spreads, tab_backtest = st.tabs(
    ["Signal", "Spreads", "Backtest"])

with tab_signal:
    # Lazy load market data only when Signal tab is active
    with st.spinner("Loading market data..."):
        spy, vix, spy_close, vix_close, inds = load_market_data()
    ema21 = inds['ema21']
    macd_diff = inds['macd_diff']
    rsi14 = inds['rsi14']

    diag = signal_diagnostics(spy_close, ema21, macd_diff, rsi14, vix_close)
    m = diag['metrics']
    c = diag['checks']
    # Price (live if available), VIX, and Price vs EMA21
    # Prefer DXLink price if user is authenticated; fallback to yfinance
    live = None
    try:
        if 'auth_session' in st.session_state and st.session_state['auth_session'] is not None:
            snap = get_symbol_price_snapshot(
                st.session_state['auth_session'], 'SPY', timeout=3)
            if isinstance(snap, dict) and isinstance(snap.get('price'), (int, float)):
                live = {"timestamp": pd.Timestamp.utcnow(
                ), "price": float(snap['price'])}
        if live is None:
            live = get_live_spy_quote()
    except Exception:
        live = get_live_spy_quote()
    # Prefer DXLink price for VIX if authenticated; try common symbol variants
    live_vix = None
    try:
        if 'auth_session' in st.session_state and st.session_state['auth_session'] is not None:
            for vix_sym in ('VIX', '^VIX'):
                snap_vix = get_symbol_price_snapshot(
                    st.session_state['auth_session'], vix_sym, timeout=3)
                if isinstance(snap_vix, dict) and isinstance(snap_vix.get('price'), (int, float)):
                    live_vix = {"timestamp": pd.Timestamp.utcnow(
                    ), "price": float(snap_vix['price'])}
                    break
        if live_vix is None:
            live_vix = get_live_vix_quote()
    except Exception:
        live_vix = get_live_vix_quote()
    price_val = live['price'] if live is not None else m['close']
    vix_val = live_vix['price'] if live_vix is not None else m['vix_close']

    # Calculate percentage changes from daily close
    spy_pct_change = (
        (price_val - m['close']) / m['close']) * 100 if price_val != m['close'] else 0
    vix_pct_change = ((vix_val - m['vix_close']) / m['vix_close']
                      ) * 100 if vix_val != m['vix_close'] else 0

    # Fetch Shiller PE ratio
    shiller_pe = get_shiller_pe()

    cols = st.columns(3)
    cols[0].metric("SPY", f"{price_val:.2f}", f"{spy_pct_change:+.2f}%")
    cols[1].metric("VIX", f"{vix_val:.2f}", f"{vix_pct_change:+.2f}%")

    # Market status with Shiller PE
    market_status = "🟢 OPEN" if is_us_equity_market_open() else "🔴 CLOSED"
    if shiller_pe is not None:
        # Color the delta: green if <25 (good), normal if 25-30, red if >30 (overvalued)
        delta_color = "normal" if 25 <= shiller_pe <= 30 else (
            "inverse" if shiller_pe > 30 else "off")
        cols[2].metric("Market Status", market_status,
                       f"Shiller PE {shiller_pe:.1f}", delta_color=delta_color)
    else:
        cols[2].metric("Market Status", market_status)

    # Signal Status - only show the filters that matter (RSI and VIX)
    st.write("#### Signal Status")
    sig_cols = st.columns(2)

    # RSI with distance from 55
    rsi_diff = m['rsi'] - 55
    sig_cols[0].metric(
        "RSI > 55",
        "✅ YES" if m['rsi'] > 55 else "❌ NO",
        f"{rsi_diff:+.1f} pts"
    )

    # VIX with distance from 20
    vix_diff = vix_val - 20
    sig_cols[1].metric(
        "VIX < 20",
        "✅ YES" if vix_val < 20 else "❌ NO",
        f"{vix_diff:+.2f}"
    )

    st.write("\n")
    # Fixed visible range: last 90 days (no slider)
    days = 90

    # Build a single interactive daily chart (no separate overview)
    try:
        idx = spy_close.index
        # If index is timezone-aware DatetimeIndex, drop tz for cleaner axis
        try:
            if isinstance(idx, pd.DatetimeIndex) and getattr(idx, 'tz', None) is not None:
                idx = idx.tz_localize(None)
        except Exception:
            pass

        # Limit to the last N days to avoid compressing scale
        # Use fewer days on mobile for better performance
        if 'HTTP_USER_AGENT' in st.context.headers:
            user_agent = st.context.headers['HTTP_USER_AGENT'].lower()
            is_mobile_device = any(x in user_agent for x in [
                                   'mobile', 'android', 'iphone', 'ipad'])
            days = 90 if is_mobile_device else days  # Reduce to 90 days on mobile

        if isinstance(idx, pd.DatetimeIndex) and len(idx) > 0:
            last_date = idx.max()
            cutoff = last_date - pd.Timedelta(days=days)
            mask = idx >= cutoff
            idx_filtered = idx[mask]
            close_filtered = spy_close[mask]
            ema_filtered = ema21[mask]
        else:
            idx_filtered = idx
            close_filtered = spy_close
            ema_filtered = ema21

        df_plot = pd.DataFrame({
            'Date': idx_filtered,
            'Close': close_filtered.values,
            'EMA21': ema_filtered.values,
        })

        # Tailor to EMA availability: drop days where EMA isn't available
        df_plot = df_plot.dropna(subset=['EMA21'])

        # On mobile, sample every other point to reduce rendering load
        if 'HTTP_USER_AGENT' in st.context.headers:
            user_agent = st.context.headers['HTTP_USER_AGENT'].lower()
            is_mobile_device = any(x in user_agent for x in [
                                   'mobile', 'android', 'iphone', 'ipad'])
            if is_mobile_device and len(df_plot) > 60:
                # Sample every 2nd point on mobile for faster rendering
                df_plot = df_plot.iloc[::2].copy()

        # Enrich with MACD, RSI, and VIX aligned by date for per-day checks
        if not df_plot.empty:
            date_index = pd.DatetimeIndex(df_plot['Date'])
            df_plot['MACD'] = macd_diff.reindex(date_index).values
            df_plot['RSI'] = rsi14.reindex(date_index).values
            df_plot['VIX'] = vix_close.reindex(date_index).values

            # Create ordinal index for continuous X-axis (removes gaps for non-trading days)
            df_plot = df_plot.reset_index(drop=True)
            df_plot['Index'] = df_plot.index
            df_plot['DateStr'] = df_plot['Date'].dt.strftime('%Y-%m-%d')

            # Compute daily checks - only the filters that matter (RSI > 55 and VIX < 20)
            df_plot['AllOK'] = (
                (df_plot['RSI'] > 55) &
                (df_plot['VIX'] < 20)
            )
            # Precompute rectangle bounds to span full chart height per day
            y_min_const = float(
                min(df_plot['Close'].min(), df_plot['EMA21'].min()))
            y_max_const = float(
                max(df_plot['Close'].max(), df_plot['EMA21'].max()))
            df_plot['y_min'] = y_min_const
            df_plot['y_max'] = y_max_const
            # Center the bar around each day: from Index-0.5 to Index+0.5
            df_plot['IndexStart'] = df_plot['Index'] - 0.5
            df_plot['IndexEnd'] = df_plot['Index'] + 0.5

        if df_plot.empty:
            st.info("Not enough data to plot EMA for the last 90 days.")
        else:
            # Single tooltip for the day: respond anywhere along the x-axis
            nearest = alt.selection_point(
                nearest=True, on='mousemove', encodings=['x'], empty=False)

            # Invisible rule across full height for each date to capture hover anywhere
            hover_selector = alt.Chart(df_plot).mark_rule(color='transparent', strokeWidth=50).encode(
                x='Index:Q',
                tooltip=[
                    alt.Tooltip('DateStr:N', title='Date'),
                    alt.Tooltip('Close:Q', format=',.2f', title='Close'),
                    alt.Tooltip('EMA21:Q', format=',.2f', title='EMA21'),
                ]
            ).add_params(nearest)

            # Background shading for days where signal is active (centered on each day)
            bg = alt.Chart(df_plot).mark_rect(color='#66bb6a', opacity=0.18).encode(
                x='IndexStart:Q', x2='IndexEnd:Q', y='y_min:Q', y2='y_max:Q', tooltip=[]
            ).transform_filter('datum.AllOK == true')

            close_line = alt.Chart(df_plot).mark_line(color='#4e79a7', strokeWidth=2).encode(
                x=alt.X('Index:Q', title='', axis=alt.Axis(
                    labels=False, ticks=False)),
                y=alt.Y('Close:Q', title='', scale=alt.Scale(zero=False)),
                tooltip=[],
            )
            ema_line = alt.Chart(df_plot).mark_line(color='#f28e2c', strokeDash=[6, 3]).encode(
                x='Index:Q', y=alt.Y('EMA21:Q', title='', scale=alt.Scale(zero=False)), tooltip=[]
            )

            # nearest selection already defined above

            points_close = alt.Chart(df_plot).mark_circle(color='#4e79a7', size=40).encode(
                x='Index:Q',
                y='Close:Q',
                opacity=alt.condition(nearest, alt.value(1), alt.value(0)),
                tooltip=[],
            )
            points_ema = alt.Chart(df_plot).mark_circle(color='#f28e2c', size=40).encode(
                x='Index:Q',
                y='EMA21:Q',
                opacity=alt.condition(nearest, alt.value(1), alt.value(0)),
                tooltip=[],
            )
            hover_rule = alt.Chart(df_plot).mark_rule(color='#999', strokeWidth=2).encode(
                x='Index:Q', tooltip=[]
            ).transform_filter(nearest)

            # Order: background shading and lines first, then rule tooltip, then invisible hitbox on top
            layers = [bg, close_line, ema_line,
                      points_close, points_ema, hover_rule, hover_selector]

            # Overlay latest intraday price as a red dot if available
            if live is not None:
                live_ts = pd.Timestamp(live['timestamp'])
                try:
                    # Drop timezone for consistency with df_plot dates
                    if getattr(live_ts, 'tzinfo', None) is not None:
                        live_ts = live_ts.tz_convert(None)
                except Exception:
                    try:
                        live_ts = live_ts.tz_localize(None)
                    except Exception:
                        pass
                live_df = pd.DataFrame({
                    'Date': [live_ts],
                    'LivePrice': [live['price']]
                })
                live_point = alt.Chart(live_df).mark_circle(color='red', size=80).encode(
                    x='Date:T',
                    y=alt.Y('LivePrice:Q', title='Price'),
                    tooltip=[],
                )
                layers.append(live_point)

            # Draw rule on top so it's always visible
            price_chart = alt.layer(
                *([bg, close_line, ema_line, points_close, points_ema, hover_selector, hover_rule])
            ).properties(height=320, title='SPY Price & EMA21')

            # MACD bar chart
            macd_base = alt.Chart(df_plot).encode(
                x=alt.X('Index:Q', title='', axis=alt.Axis(
                    labels=False, ticks=False))
            )

            # Color bars based on positive/negative
            macd_bars = macd_base.mark_bar().encode(
                y=alt.Y('MACD:Q', title='MACD', scale=alt.Scale(zero=True)),
                color=alt.condition(
                    alt.datum.MACD > 0,
                    alt.value('#66bb6a'),  # Green for positive
                    alt.value('#f44336')   # Red for negative
                ),
                tooltip=[
                    alt.Tooltip('DateStr:N', title='Date'),
                    alt.Tooltip('MACD:Q', format=',.4f', title='MACD'),
                ]
            )

            # Zero line for reference
            macd_zero = macd_base.mark_rule(color='#999', strokeDash=[3, 3]).encode(
                y=alt.datum(0)
            )

            macd_chart = alt.layer(macd_bars, macd_zero).properties(
                height=100,
                title='MACD Histogram'
            )

            # RSI chart
            # Sample every Nth date for X-axis labels to avoid crowding
            label_indices = df_plot.index[::10].tolist()  # Every 10th point
            df_plot['ShowLabel'] = df_plot['Index'].isin(label_indices)

            rsi_base = alt.Chart(df_plot).encode(
                x=alt.X('Index:Q', title='', axis=alt.Axis(
                    labels=False, ticks=False))
            )

            # RSI line
            rsi_line = rsi_base.mark_line(color='#9467bd', strokeWidth=2).encode(
                y=alt.Y('RSI:Q', title='RSI',
                        scale=alt.Scale(domain=[20, 80])),
                tooltip=[
                    alt.Tooltip('DateStr:N', title='Date'),
                    alt.Tooltip('RSI:Q', format=',.1f', title='RSI'),
                ]
            )

            # Horizontal reference line at 55 (signal threshold)
            rsi_ref_55 = alt.Chart(pd.DataFrame({'y': [55]})).mark_rule(
                color='#2196f3',
                strokeWidth=2,
                strokeDash=[5, 5]
            ).encode(
                y=alt.Y('y:Q', scale=alt.Scale(domain=[20, 80]))
            )

            # Horizontal reference line at 70 (overbought)
            rsi_ref_70 = alt.Chart(pd.DataFrame({'y': [70]})).mark_rule(
                color='#4caf50',
                strokeWidth=1,
                strokeDash=[3, 3],
                opacity=0.5
            ).encode(
                y=alt.Y('y:Q', scale=alt.Scale(domain=[20, 80]))
            )

            rsi_chart = alt.layer(
                rsi_ref_55, rsi_ref_70, rsi_line
            ).properties(
                height=150,
                title='RSI (Signal: > 55)'
            )

            # VIX Chart
            vix_base = alt.Chart(df_plot).encode(
                x=alt.X('Index:Q', title='', axis=alt.Axis(
                    labels=False, ticks=False))
            )

            # VIX line
            vix_line = vix_base.mark_line(color='#e74c3c', strokeWidth=2).encode(
                y=alt.Y('VIX:Q', title='VIX', scale=alt.Scale(zero=False)),
                tooltip=[
                    alt.Tooltip('DateStr:N', title='Date'),
                    alt.Tooltip('VIX:Q', format=',.2f', title='VIX'),
                ]
            )

            # Horizontal reference line at 20 (signal threshold)
            vix_ref_20 = alt.Chart(pd.DataFrame({'y': [20]})).mark_rule(
                color='#2196f3',
                strokeWidth=2,
                strokeDash=[5, 5]
            ).encode(
                y=alt.Y('y:Q')
            )

            # Horizontal reference line at 18 (optimal threshold)
            vix_ref_18 = alt.Chart(pd.DataFrame({'y': [18]})).mark_rule(
                color='#4caf50',
                strokeWidth=1,
                strokeDash=[3, 3],
                opacity=0.5
            ).encode(
                y=alt.Y('y:Q')
            )

            vix_chart = alt.layer(
                vix_ref_18, vix_ref_20, vix_line
            ).properties(
                height=150,
                title='VIX (Signal: < 20)'
            )

            # Combine price, RSI, and VIX charts vertically
            combined_chart = alt.vconcat(price_chart, rsi_chart, vix_chart).resolve_scale(
                x='shared'
            )

            st.altair_chart(combined_chart, use_container_width=True)

            # Add spacing and info caption after charts
            st.write("")
            st.caption(
                "📊 Data cached 15 minutes. OAuth tokens saved locally; secrets stay on your machine. Tokens auto-refresh when possible."
            )
    except Exception as e:
        st.error(f"Failed to render chart: {e}")

with tab_spreads:
    st.subheader("Options Spreads from Tastytrade")

    # Automated Signal Detection Section
    st.write("---")
    st.subheader("🤖 Automated Signal Detection")

    # Initialize automation engine in session state
    if 'automation_engine' not in st.session_state:
        st.session_state.automation_engine = AutomationEngine(
            rsi_threshold=55.0,
            vix_threshold=20.0,
            execution_window=("10:30", "10:45"),
            check_interval_seconds=300
        )

    auto_engine = st.session_state.automation_engine

    # Display current configuration (read-only)
    st.write("**Configuration:**")
    config_cols = st.columns(4)
    config_cols[0].metric("RSI Threshold", "≥ 55")
    config_cols[1].metric("VIX Threshold", "≤ 20")
    config_cols[2].metric("Window Start", "10:30 AM ET")
    config_cols[3].metric("Window End", "10:45 AM ET")

    # Check Signal Now button
    if st.button("🔍 Check Signal Now", use_container_width=True):
        with st.spinner("Checking market conditions..."):
            ready_to_trade, conditions = auto_engine.check_and_log_conditions()

            # Display results
            if ready_to_trade:
                st.success("✅ **SIGNAL CONFIRMED** - Ready to execute trade!")
            else:
                st.info(f"⏳ {conditions.get('reason', 'Waiting for signal')}")

            # Show detailed conditions
            cond_cols = st.columns(4)
            cond_cols[0].metric("RSI", f"{conditions.get('rsi', 0):.2f}",
                                f"Target: ≥55")
            cond_cols[1].metric("VIX", f"{conditions.get('vix', 0):.2f}",
                                f"Target: ≤20")
            cond_cols[2].metric("Market Hours",
                                "✅ Open" if conditions.get('can_trade_now') else "❌ Closed")
            cond_cols[3].metric("Execution Window",
                                "✅ Active" if conditions.get('can_trade_now') else "⏳ Waiting")

            # Show schedule reason
            if not conditions.get('can_trade_now'):
                st.caption(f"📅 {conditions.get('schedule_reason', '')}")

    # Show last check status
    if auto_engine.last_check_time:
        st.caption(
            f"🕐 Last checked: {auto_engine.last_check_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")

        if auto_engine.last_signal_conditions:
            with st.expander("📊 Last Signal Details", expanded=False):
                st.json(auto_engine.last_signal_conditions)

    st.write("---")

    # TODO: Fetch live account balance when automating
    # Example API call: account_balance = get_account_balance(session)
    # For now, position sizing uses hardcoded ACCOUNT_SIZE constant in the spread display section

    ok = ensure_session()
    if ok:
        try:
            with st.spinner("Fetching SPY options..."):
                options_json = fetch_spy_options(
                    st.session_state['auth_session'])
        except Exception as e:
            st.error(f"❌ Failed to fetch options data: {e}")
            st.write("This could be due to:")
            st.write("- API authentication issues")
            st.write("- Network connectivity problems")
            st.write("- Tastytrade API being unavailable")
            st.write("Try re-authorizing or check your connection.")
        else:
            meta = options_json.get('meta', {}) if isinstance(
                options_json, dict) else {}
            if meta:
                st.caption(f"Source: {meta.get('source')}")

            candidate = find_put_credit_spread(
                options_json,
                session=st.session_state['auth_session'],
                dte_min=30,
                dte_max=45,
                sell_delta_target=0.50,
                buy_delta_target=0.25
            )

            if candidate:
                sell_leg, buy_leg, spread_width, credit, max_loss, avg_return_pct, dte = candidate

                # Store spread details in session state for OCO orders
                st.session_state['current_spread'] = {
                    'sell_leg': sell_leg,
                    'buy_leg': buy_leg,
                    'credit': credit
                }

                # Display spread details in organized layout
                st.success("✅ Found candidate put credit spread")
                st.caption(
                    "🎯 Optimized strategy: Collect credit with defined max risk")

                # Main spread metrics
                met_cols = st.columns(4)
                met_cols[0].metric("Credit Received", f"${credit:.2f}")
                met_cols[1].metric("Max Risk", f"${max_loss:.2f}")
                met_cols[2].metric("Return on Risk %", f"{avg_return_pct:.1f}%",
                                   "✅ Excellent" if avg_return_pct >= 20 else "⚠️ Low")
                met_cols[3].metric("DTE", f"{dte} days")

                st.write("#### Spread Details")

                # Sell leg (short put)
                st.write("**📤 SELL (Short Put)**")
                sell_cols = st.columns(4)
                sell_cols[0].write(
                    f"Strike: **${sell_leg.get('strike-price', 'N/A')}**")
                sell_cols[1].write(
                    f"Exp: **{sell_leg.get('expiration-date', 'N/A')}**")
                # Delta (safe formatting)
                _sd = sell_leg.get('delta')
                try:
                    _sd_str = f"{float(_sd):.3f}" if _sd is not None else "N/A"
                except Exception:
                    _sd_str = "N/A"
                sell_cols[2].write(f"Delta: **{_sd_str}**")
                # Show mid price (average of bid and ask)
                _sell_bid = sell_leg.get('bid')
                _sell_ask = sell_leg.get('ask')
                try:
                    if _sell_bid is not None and _sell_ask is not None:
                        _sell_mid = (float(_sell_bid) + float(_sell_ask)) / 2.0
                        sell_cols[3].write(f"Mid: **${_sell_mid:.2f}**")
                    else:
                        sell_cols[3].write(f"Mid: **N/A**")
                except Exception:
                    sell_cols[3].write(f"Mid: **N/A**")

                # Buy leg (long put)
                st.write("**📥 BUY (Long Put)**")
                buy_cols = st.columns(4)
                buy_cols[0].write(
                    f"Strike: **${buy_leg.get('strike-price', 'N/A')}**")
                buy_cols[1].write(
                    f"Exp: **{buy_leg.get('expiration-date', 'N/A')}**")
                _bd = buy_leg.get('delta')
                try:
                    _bd_str = f"{float(_bd):.3f}" if _bd is not None else "N/A"
                except Exception:
                    _bd_str = "N/A"
                buy_cols[2].write(f"Delta: **{_bd_str}**")
                # Show mid price (average of bid and ask)
                _buy_bid = buy_leg.get('bid')
                _buy_ask = buy_leg.get('ask')
                try:
                    if _buy_bid is not None and _buy_ask is not None:
                        _buy_mid = (float(_buy_bid) + float(_buy_ask)) / 2.0
                        buy_cols[3].write(f"Mid: **${_buy_mid:.2f}**")
                    else:
                        buy_cols[3].write(f"Mid: **N/A**")
                except Exception:
                    buy_cols[3].write(f"Mid: **N/A**")

                # Show Deltas
                st.write("#### Deltas")
                delta_cols = st.columns(2)
                delta_cols[0].write(f"**Sell Put δ:** {_sd_str}")
                delta_cols[1].write(f"**Buy Put δ:** {_bd_str}")

                st.info("💡 Targets: Sell δ ≈ -0.50, Buy δ ≈ -0.25 | DTE 30-45 days")

                # Risk profile
                st.write("#### Risk Profile")
                risk_cols = st.columns(2)
                risk_cols[0].write(f"**Spread Width:** ${spread_width:.2f}")
                risk_cols[1].write(
                    f"**Break-even:** ${sell_leg.get('strike-price', 0) - credit:.2f}")

                # Fetch Live Account Balance
                st.write("#### 💰 Account Balance")
                try:
                    from integrations.tastytrade.account import get_account_numbers, get_account_summary
                    session = st.session_state.get('auth_session')

                    if session:
                        account_numbers = get_account_numbers(session)
                        if account_numbers:
                            account_number = account_numbers[0]
                            account_summary = get_account_summary(
                                session, account_number)

                            ACCOUNT_SIZE = account_summary['net_liquidating_value']

                            # Display account info
                            acc_cols = st.columns(4)
                            acc_cols[0].metric(
                                "Account", account_summary['account_number'])
                            acc_cols[1].metric(
                                "Net Liq Value", f"${ACCOUNT_SIZE:,.2f}")
                            acc_cols[2].metric(
                                "OPT BP (Options)", f"${account_summary['buying_power']:,.2f}")
                            acc_cols[3].metric(
                                "Open Positions", account_summary['open_positions_count'])

                            # Check for zero or negative balance
                            if ACCOUNT_SIZE <= 0:
                                st.error(
                                    f"⚠️ Account balance is ${ACCOUNT_SIZE:,.2f}. Cannot calculate position sizing or execute trades.")
                                st.warning(
                                    "💡 Please fund your account or switch to sandbox mode for testing.")
                                st.stop()

                            # Check for zero buying power
                            if account_summary['buying_power'] <= 0:
                                buying_power = account_summary['buying_power']
                                st.error(
                                    f"🚨 OPTIONS BUYING POWER IS ${buying_power:,.2f} - Cannot place trades!")
                                st.stop()
                            else:
                                st.success(
                                    f"✅ Using live account balance: ${ACCOUNT_SIZE:,.2f} | Options BP: ${account_summary['buying_power']:,.2f}")
                        else:
                            st.error("⚠️ No accounts found")
                            st.stop()
                    else:
                        st.error(
                            "⚠️ Not authenticated - please authorize with TastyTrade")
                        st.stop()

                except Exception as e:
                    st.error(f"❌ Failed to fetch account balance: {e}")
                    st.stop()

                # Position Sizing Recommendations
                RISK_PER_POSITION_PCT = 1.7  # Fixed rule: risk 1.7% per trade
                MAX_PORTFOLIO_DEPLOYMENT_PCT = 10.0  # Fixed rule: max 10% total deployment

                # Execution Timing Rules for Automation
                EXECUTION_TIME_ET = "10:30"  # Execute at 10:30 AM Eastern Time
                EXECUTION_WINDOW_MINUTES = 15  # Execution window: 10:30-10:45 AM ET
                # Rationale: Volatility settled, tight spreads, good liquidity, clear market direction

                # Exit Strategy Rules for Automation
                # Close at 50% profit (if sold $1.00, close at $0.50)
                TAKE_PROFIT_PCT_OF_CREDIT = 0.5
                # Close at 100% loss (if sold $1.00, close at $2.00)
                STOP_LOSS_PCT_OF_CREDIT = 1.0
                # Rationale: Capture majority of profit early, limit losses to 1:1 risk/reward

                st.write("#### 📊 Position Sizing (Auto-Calculated)")

                # Calculate position size based on fixed rules
                max_risk_per_trade = ACCOUNT_SIZE * \
                    (RISK_PER_POSITION_PCT / 100.0)

                # Calculate number of contracts based on risk
                num_contracts = int(max_risk_per_trade /
                                    max_loss) if max_loss > 0 else 1
                num_contracts = max(1, num_contracts)  # At least 1 contract

                # Get the safety limit from the safety controls section
                # We'll need to cap at max_position_size, but that's defined later
                # So we'll use a default of 5 here and apply the actual limit later
                safety_position_limit = 5  # Default, will be overridden by user setting below

                # Calculate actual capital deployed and risk
                actual_capital_at_risk = num_contracts * max_loss
                actual_credit_received = num_contracts * credit
                actual_return_if_win = actual_credit_received
                actual_risk_pct = (actual_capital_at_risk /
                                   ACCOUNT_SIZE) * 100.0

                # Show recommendations
                size_cols = st.columns(4)
                size_cols[0].metric(
                    "Recommended Contracts",
                    f"{num_contracts}",
                    help=f"Auto-calculated: ${max_risk_per_trade:.2f} max risk ÷ ${max_loss:.2f} per contract"
                )
                size_cols[1].metric(
                    "Total Capital at Risk",
                    f"${actual_capital_at_risk:.2f}",
                    help=f"{num_contracts} contracts × ${max_loss:.2f} max loss"
                )
                size_cols[2].metric(
                    "Total Credit Received",
                    f"${actual_credit_received:.2f}",
                    help=f"{num_contracts} contracts × ${credit:.2f} credit"
                )
                size_cols[3].metric(
                    "Account Risk %",
                    f"{actual_risk_pct:.2f}%",
                    help=f"${actual_capital_at_risk:.2f} ÷ ${ACCOUNT_SIZE:,.0f} account"
                )

                if actual_risk_pct <= RISK_PER_POSITION_PCT:
                    st.success(
                        f"✅ Position fits within {RISK_PER_POSITION_PCT}% risk rule (using {actual_risk_pct:.2f}%)")
                else:
                    st.warning(
                        f"⚠️ Position risks {actual_risk_pct:.2f}% (target: {RISK_PER_POSITION_PCT}%). Reducing to 1 contract recommended.")

                st.caption(f"🤖 **Auto-Trade Setup:** Sell {num_contracts} × ${sell_leg.get('strike-price', 0)} put, Buy {num_contracts} × ${buy_leg.get('strike-price', 0)} put → ${actual_credit_received:.2f} credit | Rules: {RISK_PER_POSITION_PCT}% risk per trade, {MAX_PORTFOLIO_DEPLOYMENT_PCT}% max total deployment")

                # Manual contract override
                st.write("\n")
                st.write("**Override Position Size (Optional):**")
                use_manual_contracts = st.checkbox(
                    "Manually set number of contracts",
                    value=False,
                    help="Override the auto-calculated position size"
                )

                if use_manual_contracts:
                    manual_contracts = st.number_input(
                        "Number of Contracts",
                        min_value=1,
                        max_value=20,
                        value=min(num_contracts, 5),
                        step=1,
                        help="Manually specify how many contracts to trade"
                    )
                    # Override the calculated value
                    num_contracts = manual_contracts

                    # Recalculate metrics with manual value
                    actual_capital_at_risk = num_contracts * max_loss
                    actual_credit_received = num_contracts * credit
                    actual_risk_pct = (
                        actual_capital_at_risk / ACCOUNT_SIZE) * 100.0

                    st.info(
                        f"📝 Using manual override: {num_contracts} contracts = ${actual_capital_at_risk:.2f} at risk ({actual_risk_pct:.2f}%)")

                st.write("\n")
                st.subheader("📤 Execute Trade in Sandbox")

                # Profit/Loss targets
                exec_col1, exec_col2 = st.columns(2)
                with exec_col1:
                    take_profit_pct = st.slider(
                        "Take Profit %",
                        min_value=25,
                        max_value=75,
                        value=50,
                        step=5,
                        help="Close position when profit reaches this % of max credit"
                    )
                with exec_col2:
                    stop_loss_pct = st.slider(
                        "Stop Loss %",
                        min_value=50,
                        max_value=200,
                        value=100,
                        step=10,
                        help="Close position when loss reaches this % of max loss"
                    )

                # Show what the exit prices will be
                take_profit_debit = credit * (1 - take_profit_pct / 100.0)
                stop_loss_debit = credit * (1 + stop_loss_pct / 100.0)

                exit_info = st.columns(3)
                exit_info[0].metric("Entry Credit", f"${credit:.2f}")
                exit_info[1].metric(
                    "Take Profit At",
                    f"${take_profit_debit:.2f}",
                    help=f"Close when spread drops to ${take_profit_debit:.2f} ({take_profit_pct}% profit)"
                )
                exit_info[2].metric(
                    "Stop Loss At",
                    f"${stop_loss_debit:.2f}",
                    help=f"Close when spread rises to ${stop_loss_debit:.2f} ({stop_loss_pct}% loss)"
                )

                # Production Safety Controls
                st.write("\n")
                st.divider()
                st.subheader("🛡️ Production Safety Controls")

                # Initialize safety manager based on environment
                is_production = os.getenv(
                    'TASTYTRADE_USE_PRODUCTION', 'true').lower() in ('true', '1', 'yes')

                if is_production:
                    st.warning("⚠️ **PRODUCTION MODE** - Real money at risk!")
                else:
                    st.info("🧪 **SANDBOX MODE** - Safe testing environment")

                # Calculate dynamic max position size based on account balance
                # Rule: Allow enough contracts to deploy up to 10% of account
                # Assuming average max loss of $100 per contract (conservative estimate)
                avg_max_loss_per_contract = 100.0
                max_deployment = ACCOUNT_SIZE * 0.10  # 10% of account
                calculated_max_position = int(
                    max_deployment / avg_max_loss_per_contract)
                calculated_max_position = max(
                    1, min(calculated_max_position, 20))  # Between 1-20

                # Safety configuration
                safety_col1, safety_col2, safety_col3 = st.columns(3)

                with safety_col1:
                    max_position_size = st.number_input(
                        "Max Position Size (contracts)",
                        min_value=1,
                        max_value=20,
                        value=calculated_max_position,
                        help=f"Maximum contracts per trade (auto-calculated: {calculated_max_position} based on ${ACCOUNT_SIZE:,.0f} account)"
                    )

                with safety_col2:
                    max_daily_trades = st.number_input(
                        "Max Daily Trades",
                        min_value=1,
                        max_value=10,
                        value=3,
                        help="Maximum trades per day"
                    )

                with safety_col3:
                    dry_run_mode = st.checkbox(
                        "🧪 Dry Run (Simulate Only)",
                        value=not is_production,
                        help="Simulate orders without placing them"
                    )

                # Initialize safety manager with custom limits
                safety_limits = SafetyLimits(
                    max_position_size=max_position_size,
                    max_daily_trades=max_daily_trades,
                    require_confirmation=is_production,
                    dry_run_mode=dry_run_mode
                )
                safety_manager = SafetyManager(limits=safety_limits)

                # Apply safety position limit to auto-calculated contracts
                # Cap the recommended contracts at the max_position_size
                if num_contracts > max_position_size:
                    original_num_contracts = num_contracts
                    num_contracts = max_position_size
                    st.info(
                        f"ℹ️ Auto-calculated {original_num_contracts} contracts, capped to safety limit of {max_position_size}")

                    # Recalculate metrics with capped value
                    actual_capital_at_risk = num_contracts * max_loss
                    actual_credit_received = num_contracts * credit
                    actual_risk_pct = (
                        actual_capital_at_risk / ACCOUNT_SIZE) * 100.0

                # Get current positions count (you'll implement this)
                from integrations.tastytrade.account import get_account_numbers
                try:
                    session = st.session_state.get('auth_session')
                    if session:
                        account_numbers = get_account_numbers(session)
                        if account_numbers:
                            # TODO: Implement get_positions_count function
                            current_positions = 0  # Placeholder
                        else:
                            current_positions = 0
                    else:
                        current_positions = 0
                except:
                    current_positions = 0

                # Validate trade
                is_valid, violations = safety_manager.validate_trade(
                    quantity=num_contracts,
                    spread_price=credit,
                    account_balance=ACCOUNT_SIZE,
                    current_positions=current_positions,
                    daily_trades_count=len(safety_manager.get_today_trades()),
                    max_loss_per_contract=max_loss
                )

                # Display safety report
                daily_stats = safety_manager.get_daily_stats()
                safety_status_cols = st.columns(4)
                safety_status_cols[0].metric(
                    "Trades Today", f"{daily_stats['trades_count']}/{max_daily_trades}")
                safety_status_cols[1].metric(
                    "Open Positions", f"{current_positions}/5")
                safety_status_cols[2].metric(
                    "Position Size", f"{num_contracts} contracts")
                safety_status_cols[3].metric(
                    "Capital at Risk", f"${actual_capital_at_risk:,.2f}")

                if not is_valid:
                    st.error("❌ Trade BLOCKED by safety controls:")
                    for violation in violations:
                        st.warning(f"⚠️ {violation}")
                else:
                    st.success("✅ Trade passes all safety checks")

                # Execute button - disabled if safety checks fail
                button_disabled = not is_valid
                button_label = "🧪 SIMULATE Trade (Dry Run)" if dry_run_mode else "🚀 Execute Trade with Bracket Order"

                if st.button(button_label, type="primary", key="execute_bracket_btn", use_container_width=True, disabled=button_disabled):

                    # Handle dry run mode
                    if dry_run_mode:
                        st.info(
                            "🧪 **DRY RUN MODE** - Simulating order without execution")
                        dry_run_mgr = DryRunManager()

                        spread_details = {
                            'symbol': 'SPY',
                            'quantity': num_contracts,
                            'spread_price': credit,
                            'take_profit': take_profit_debit,
                            'stop_loss': stop_loss_debit,
                            'sell_strike': sell_leg.get('strike-price', 0),
                            'buy_strike': buy_leg.get('strike-price', 0)
                        }

                        simulated_result = dry_run_mgr.simulate_otoco_order(
                            spread_details, num_contracts)

                        st.success("✅ Order simulated successfully!")
                        st.json(simulated_result)

                        # Log the simulated trade
                        safety_manager.log_trade(
                            trade_type="PUT_CREDIT_SPREAD_OTOCO",
                            spread_details=spread_details,
                            order_result=simulated_result,
                            account_balance=ACCOUNT_SIZE,
                            dry_run=True
                        )

                        st.info(
                            "💡 Enable real execution by unchecking 'Dry Run' mode")
                        st.session_state['last_bracket_result'] = simulated_result

                    else:
                        # Real execution
                        with st.spinner("Submitting bracket order..."):
                            try:
                                from integrations.tastytrade.orders import open_spread_with_bracket
                                from integrations.tastytrade.account import get_account_numbers

                                # Get session from state
                                session = st.session_state.get('auth_session')
                                if not session:
                                    st.error(
                                        "❌ No active session. Please reauthorize.")
                                    st.stop()

                                # Get account number
                                account_numbers = get_account_numbers(session)
                                if not account_numbers:
                                    st.error("❌ No accounts found")
                                    st.stop()

                                account_number = account_numbers[0]

                                # Submit the bracket order
                                result = open_spread_with_bracket(
                                    session=session,
                                    account_number=account_number,
                                    underlying="SPY",
                                    sell_leg=sell_leg,
                                    buy_leg=buy_leg,
                                    num_contracts=num_contracts,
                                    entry_credit=credit,
                                    take_profit_pct=take_profit_pct / 100.0,
                                    stop_loss_pct=stop_loss_pct / 100.0,
                                    dry_run_first=True
                                )

                                # Store result in session state FIRST (before any displays)
                                st.session_state['last_bracket_result'] = result

                                # Show prominent success message
                                st.balloons()
                                st.success(
                                    f"✅ **BRACKET ORDER SUBMITTED SUCCESSFULLY!**")
                                st.success(
                                    f"📋 Complex Order ID: **{result.get('id', 'N/A')}**")

                                # Show order details
                                with st.expander("📋 Full Order Details", expanded=True):
                                    st.json(result)

                                # Display order IDs for tracking
                                if 'orders' in result:
                                    st.write("**Order Structure:**")
                                    order_cols = st.columns(3)
                                    orders = result['orders']
                                    if len(orders) >= 1:
                                        order_cols[0].info(
                                            f"🎯 Trigger Order\n\nID: {orders[0].get('id', 'N/A')}")
                                    if len(orders) >= 2:
                                        order_cols[1].success(
                                            f"� Take Profit\n\nID: {orders[1].get('id', 'N/A')}")
                                    if len(orders) >= 3:
                                        order_cols[2].error(
                                            f"🛑 Stop Loss\n\nID: {orders[2].get('id', 'N/A')}")

                                is_production = os.getenv(
                                    'TASTYTRADE_USE_PRODUCTION', 'true').lower() in ('true', '1', 'yes')
                                if not is_production:
                                    st.write(
                                        "**Reminder:** This is sandbox - orders use simulated fills:")
                                    st.write("- Market orders fill at $1")
                                    st.write(
                                        "- Limit orders ≤ $3 fill immediately")
                                    st.write("- Limit orders ≥ $3 stay Live")
                                else:
                                    st.warning(
                                        "⚠️ **PRODUCTION ORDER PLACED** - Real money at risk!")

                                # Log the real trade
                                spread_details = {
                                    'symbol': 'SPY',
                                    'quantity': num_contracts,
                                    'spread_price': credit,
                                    'take_profit': take_profit_debit,
                                    'stop_loss': stop_loss_debit,
                                    'sell_strike': sell_leg.get('strike-price', 0),
                                    'buy_strike': buy_leg.get('strike-price', 0)
                                }
                                safety_manager.log_trade(
                                    trade_type="PUT_CREDIT_SPREAD_OTOCO",
                                    spread_details=spread_details,
                                    order_result=result,
                                    account_balance=ACCOUNT_SIZE,
                                    dry_run=False
                                )

                            except Exception as e:
                                error_msg = str(e)

                                # Show prominent error message
                                st.error("# ❌ ORDER SUBMISSION FAILED")
                                st.error(f"**Error:** {error_msg}")

                                # Provide helpful context for OTOCO errors
                                if "illegal_buy_and_sell_on_same_symbol" in error_msg or "preflight_check_failure" in error_msg:
                                    st.warning(
                                        "⚠️ **OTOCO Bracket Order Issue:** TastyTrade sandbox may not support OTOCO complex orders.")
                                    st.info(
                                        "💡 **Workaround:** Use the 'Submit Test Order' button below instead - it submits a simple opening order without the bracket.")

                                with st.expander("📋 Full Error Details"):
                                    import traceback
                                    st.code(traceback.format_exc())

                st.write("\n")
                st.divider()

                # Show last bracket order result if it exists
                if 'last_bracket_result' in st.session_state:
                    with st.expander("📋 Last Bracket Order Details (Persisted)", expanded=True):
                        st.caption(
                            "✅ This data persists across page refreshes")
                        st.json(st.session_state['last_bracket_result'])
                        if st.button("Clear", key="clear_last_bracket"):
                            del st.session_state['last_bracket_result']
                            st.rerun()

                # Order Management Section
                st.subheader("📋 Live Orders & Testing")

                # Show last refresh time
                import datetime as dt_module
                current_time = dt_module.datetime.now().strftime("%H:%M:%S")
                st.caption(f"🕐 Last refreshed: {current_time}")

                st.write("**🧪 Sandbox Fill Testing:**")
                st.caption(
                    "Limit orders > $3 won't auto-fill in sandbox and will stay Live until manually filled or cancelled.")

                col_test1, col_test2 = st.columns(2)

                with col_test1:
                    if st.button("🔬 Submit Test Order (≤ $3 for instant fill)", key="test_order_btn", use_container_width=True):
                        with st.spinner("Submitting simple test order that will auto-fill..."):
                            try:
                                from integrations.tastytrade.orders import open_spread_simple
                                from integrations.tastytrade.account import get_account_numbers

                                session = st.session_state.get('auth_session')
                                if session:
                                    account_numbers = get_account_numbers(
                                        session)
                                    if account_numbers:
                                        account_number = account_numbers[0]

                                        # Submit simple order (no OCO) with reduced credit ≤ $3 for instant sandbox fill
                                        test_credit = 2.99
                                        result = open_spread_simple(
                                            session=session,
                                            account_number=account_number,
                                            underlying="SPY",
                                            sell_leg=sell_leg,
                                            buy_leg=buy_leg,
                                            num_contracts=1,  # Use 1 contract for testing
                                            entry_credit=test_credit,
                                            dry_run_first=False  # Skip dry run for test
                                        )

                                        st.success(
                                            "✅ Test order submitted at $2.99 - should fill immediately!")
                                        st.json(result)
                                        st.info(
                                            "💡 This is a simple order (no OCO). Once filled, click 'Refresh Live Orders' below to see the filled position.")

                                        # Store result in session state so it persists
                                        st.session_state['last_order_result'] = result

                            except Exception as e:
                                st.error(f"❌ Test order failed: {e}")
                                import traceback
                                st.code(traceback.format_exc())

                with col_test2:
                    if st.button("🔄 Refresh Live Orders", key="refresh_orders_btn", use_container_width=True):
                        st.rerun()

                # Show last submitted order result if it exists
                if 'last_order_result' in st.session_state:
                    with st.expander("📋 Last Submitted Order Details", expanded=True):
                        st.json(st.session_state['last_order_result'])
                        if st.button("Clear", key="clear_last_order"):
                            del st.session_state['last_order_result']
                            st.rerun()

                st.write("\n")

                try:
                    from integrations.tastytrade.orders import get_live_orders
                    from integrations.tastytrade.account import get_account_numbers

                    session = st.session_state.get('auth_session')
                    if session:
                        account_numbers = get_account_numbers(session)
                        if account_numbers:
                            account_number = account_numbers[0]
                            live_orders = get_live_orders(
                                session, account_number)

                            if live_orders:
                                # Group orders by complex-order-id
                                complex_orders = {}
                                simple_orders = []

                                for order in live_orders:
                                    complex_id = order.get('complex-order-id')
                                    if complex_id:
                                        if complex_id not in complex_orders:
                                            complex_orders[complex_id] = {
                                                'trigger': None,
                                                'oco': [],
                                                'type': None
                                            }

                                        tag = order.get(
                                            'complex-order-tag', '')
                                        if 'trigger' in tag.lower():
                                            complex_orders[complex_id]['trigger'] = order
                                            # Extract type from tag (e.g., "OTOCO::trigger-order" -> "OTOCO")
                                            complex_orders[complex_id]['type'] = tag.split(
                                                '::')[0] if '::' in tag else 'Complex'
                                        else:
                                            complex_orders[complex_id]['oco'].append(
                                                order)
                                    else:
                                        simple_orders.append(order)

                                total_order_count = len(
                                    complex_orders) + len(simple_orders)
                                st.write(
                                    f"**Found {total_order_count} order group(s):**")

                                # Display complex orders (OTOCO, OCO, etc.)
                                for complex_id, group in complex_orders.items():
                                    order_type = group['type'] or 'Complex'
                                    trigger = group['trigger']
                                    oco_orders = group['oco']

                                    if trigger:
                                        trigger_status = trigger.get(
                                            'status', 'Unknown')
                                        trigger_price = trigger.get(
                                            'price', 'N/A')
                                        trigger_effect = trigger.get(
                                            'price-effect', '')

                                        with st.expander(f"🎯 {order_type} Bracket Order #{complex_id} - Trigger: {trigger_status}"):
                                            st.write(
                                                f"**📦 Complex Order Type:** {order_type}")
                                            st.write(
                                                f"**Complex Order ID:** {complex_id}")
                                            st.divider()

                                            # Trigger order section
                                            st.write(
                                                "**1️⃣ TRIGGER ORDER (Entry)**")
                                            col1, col2, col3 = st.columns(3)
                                            col1.metric(
                                                "Order ID", f"#{trigger.get('id')}")
                                            col2.metric(
                                                "Status", trigger_status)
                                            col3.metric(
                                                "Price", f"${trigger_price} {trigger_effect}")

                                            if trigger.get('legs'):
                                                st.write("**Legs:**")
                                                for i, leg in enumerate(trigger['legs']):
                                                    st.write(
                                                        f"  {i+1}. {leg.get('action')} {leg.get('quantity')}x {leg.get('symbol')}")

                                            # OCO orders section
                                            if oco_orders:
                                                st.divider()
                                                st.write(
                                                    f"**2️⃣ OCO EXIT ORDERS** ({len(oco_orders)} orders - One Cancels Other)")
                                                st.caption(
                                                    "These activate when the trigger order fills. Whichever executes first will automatically cancel the other.")

                                                for idx, oco_order in enumerate(oco_orders):
                                                    oco_type = oco_order.get(
                                                        'order-type', '')
                                                    oco_status = oco_order.get(
                                                        'status', 'Unknown')
                                                    oco_price = oco_order.get(
                                                        'price', 'N/A')

                                                    # Determine if it's take profit or stop loss based on order type
                                                    if 'Stop' in oco_type:
                                                        label = "🛑 Stop Loss"
                                                    else:
                                                        label = "💰 Take Profit"

                                                    st.write(
                                                        f"\n**{label}** (Order #{oco_order.get('id')})")
                                                    oco_cols = st.columns(3)
                                                    oco_cols[0].write(
                                                        f"Type: {oco_type}")
                                                    oco_cols[1].write(
                                                        f"Status: {oco_status}")
                                                    oco_cols[2].write(
                                                        f"Price: ${oco_price}")

                                                    if oco_order.get('stop-trigger'):
                                                        st.write(
                                                            f"Stop Trigger: ${oco_order.get('stop-trigger')}")

                                                    if oco_order.get('legs'):
                                                        st.caption("Legs: " + ", ".join(
                                                            [f"{leg.get('action')} {leg.get('quantity')}x" for leg in oco_order['legs']]))

                                            # Cancel button for entire complex order
                                            st.divider()
                                            if st.button(f"❌ Cancel Entire Bracket Order #{complex_id}", key=f"cancel_complex_{complex_id}", type="secondary"):
                                                try:
                                                    from integrations.tastytrade.orders import cancel_order
                                                    cancelled_count = 0

                                                    # Cancel trigger
                                                    if trigger and trigger.get('id'):
                                                        cancel_order(
                                                            session, account_number, int(trigger.get('id')))
                                                        cancelled_count += 1

                                                    # Cancel OCO orders
                                                    for oco in oco_orders:
                                                        if oco.get('id'):
                                                            cancel_order(
                                                                session, account_number, int(oco.get('id')))
                                                            cancelled_count += 1

                                                    st.success(
                                                        f"✅ Cancelled {cancelled_count} orders in bracket #{complex_id}. Click 'Refresh' to update.")
                                                except Exception as e:
                                                    st.error(
                                                        f"❌ Cancel failed: {e}")

                                # Display simple orders
                                for order in simple_orders:
                                    with st.expander(f"Order #{order.get('id')} - {order.get('status', 'Unknown')} - {order.get('order-type', '')}"):
                                        col1, col2, col3 = st.columns(3)

                                        col1.write(
                                            f"**Status:** {order.get('status')}")
                                        col2.write(
                                            f"**Type:** {order.get('order-type')}")
                                        col3.write(
                                            f"**Size:** {order.get('size')}")

                                        if order.get('price'):
                                            st.write(
                                                f"**Price:** ${order.get('price')} ({order.get('price-effect')})")

                                        if order.get('underlying-symbol'):
                                            st.write(
                                                f"**Underlying:** {order.get('underlying-symbol')}")

                                        # Show legs
                                        if order.get('legs'):
                                            st.write("**Legs:**")
                                            for i, leg in enumerate(order['legs']):
                                                st.write(
                                                    f"  {i+1}. {leg.get('action')} {leg.get('quantity')}x {leg.get('symbol')}")

                                        # Sandbox fill simulation button
                                        if order.get('status') in ['Received', 'Live', 'Contingent']:
                                            st.write(
                                                "\n**🧪 Sandbox Testing:**")
                                            st.caption(
                                                "In sandbox, you need to manually simulate fills since limit orders > $3 don't auto-fill")

                                            if st.button(f"✅ Simulate Fill for Order #{order.get('id')}", key=f"fill_{order.get('id')}"):
                                                st.info(
                                                    "📝 In sandbox, contact TastyTrade support to manually fill orders, or:")
                                                st.write(
                                                    "1. Cancel this order")
                                                st.write(
                                                    "2. Submit a new order with price ≤ $3 for instant fill")
                                                st.write(
                                                    "3. Or use TastyTrade's sandbox UI to manually fill")

                                            # Cancel button
                                            if st.button(f"❌ Cancel Order #{order.get('id')}", key=f"cancel_{order.get('id')}", type="secondary"):
                                                try:
                                                    from integrations.tastytrade.orders import cancel_order
                                                    order_id = order.get('id')
                                                    if order_id is not None:
                                                        result = cancel_order(
                                                            session, account_number, int(order_id))
                                                        st.success(
                                                            f"✅ Order #{order_id} cancelled! Click 'Refresh Live Orders' to update the list.")
                                                        # Store cancelled order ID to hide it
                                                        if 'cancelled_orders' not in st.session_state:
                                                            st.session_state['cancelled_orders'] = [
                                                            ]
                                                        st.session_state['cancelled_orders'].append(
                                                            order_id)
                                                    else:
                                                        st.error(
                                                            "Order ID not found")
                                                except Exception as e:
                                                    st.error(
                                                        f"❌ Cancel failed: {e}")

                            else:
                                st.info("No live orders found")

                except Exception as e:
                    st.error(f"Error loading orders: {e}")

                # Show current positions
                st.write("\n")
                st.subheader("📊 Current Positions & P&L")

                try:
                    from integrations.tastytrade.account import get_options_positions, get_spread_pnl

                    session = st.session_state.get('auth_session')
                    if session:
                        account_numbers = get_account_numbers(session)
                        if account_numbers:
                            account_number = account_numbers[0]
                            positions = get_options_positions(
                                session, account_number)

                            if positions:
                                st.write(
                                    f"**Found {len(positions)} option positions:**")

                                # Calculate total P&L summary
                                total_unrealized_pnl = 0.0
                                spread_count = 0

                                # Group positions by expiration/underlying for spread identification
                                from collections import defaultdict
                                spread_groups = defaultdict(list)

                                for pos in positions:
                                    underlying = pos.get(
                                        'underlying-symbol', 'Unknown')
                                    exp_date = pos.get(
                                        'expires-at', '')[:10]  # YYYY-MM-DD
                                    key = f"{underlying}_{exp_date}"
                                    spread_groups[key].append(pos)

                                st.info(
                                    f"💡 Detected {len(spread_groups)} spread(s) across {len(positions)} positions")

                                # Display each spread group
                                for group_key, spread_positions in spread_groups.items():
                                    underlying, exp_date = group_key.split('_')

                                    with st.expander(f"📈 {underlying} Spread - Expires {exp_date} ({len(spread_positions)} legs)", expanded=True):

                                        # If 2-leg spread, calculate P&L
                                        if len(spread_positions) == 2:
                                            spread_count += 1

                                            # Get current mark prices for each leg
                                            current_marks = {}
                                            entry_credit = 0.0

                                            spread_cols = st.columns(2)

                                            for idx, pos in enumerate(spread_positions):
                                                symbol = pos.get(
                                                    'symbol', 'Unknown')
                                                qty = float(
                                                    pos.get('quantity', 0))
                                                qty_direction = pos.get(
                                                    'quantity-direction', 'Unknown')
                                                avg_price = float(
                                                    pos.get('average-open-price', 0))
                                                close_price = float(
                                                    pos.get('close-price', avg_price))
                                                multiplier = int(
                                                    pos.get('multiplier', 100))

                                                # Use close price as current mark
                                                current_marks[symbol] = close_price

                                                # Calculate entry credit component
                                                if qty_direction == 'Short':
                                                    entry_credit += avg_price * \
                                                        abs(qty) * multiplier
                                                else:
                                                    entry_credit -= avg_price * \
                                                        abs(qty) * multiplier

                                                # Display leg details
                                                position_type = "🔴 SHORT" if qty_direction == "Short" else "🟢 LONG"
                                                strike = pos.get(
                                                    'strike-price', 0)

                                                with spread_cols[idx]:
                                                    st.metric(
                                                        f"{position_type} Leg", f"${strike:.2f}")
                                                    st.caption(
                                                        f"Qty: {abs(qty):.0f} | Entry: ${avg_price:.2f} | Mark: ${close_price:.2f}")

                                            # Calculate spread P&L
                                            try:
                                                pnl_info = get_spread_pnl(
                                                    spread_positions, current_marks)

                                                unrealized_pnl = pnl_info['unrealized_pnl']
                                                pnl_pct = pnl_info['pnl_pct_of_credit'] * 100
                                                current_debit = pnl_info['current_debit']

                                                total_unrealized_pnl += unrealized_pnl

                                                # P&L display
                                                st.write("---")
                                                pnl_cols = st.columns(4)
                                                pnl_cols[0].metric(
                                                    "Entry Credit", f"${entry_credit:.2f}")
                                                pnl_cols[1].metric(
                                                    "Current Cost", f"${current_debit:.2f}")
                                                pnl_cols[2].metric("Unrealized P&L",
                                                                   f"${unrealized_pnl:.2f}",
                                                                   delta=f"{pnl_pct:.1f}%")
                                                pnl_cols[3].metric("P&L %",
                                                                   f"{pnl_pct:.1f}%",
                                                                   delta="Profit" if unrealized_pnl > 0 else "Loss")

                                                # Exit recommendations
                                                if pnl_pct >= 50:
                                                    st.success(
                                                        f"✅ **TAKE PROFIT ZONE** - Consider closing at {pnl_pct:.1f}% profit")
                                                elif pnl_pct <= -100:
                                                    st.error(
                                                        f"⚠️ **STOP LOSS ZONE** - Consider closing to limit loss at {pnl_pct:.1f}%")
                                                elif pnl_pct >= 25:
                                                    st.info(
                                                        f"💰 Approaching profit target ({pnl_pct:.1f}%)")
                                                else:
                                                    st.caption(
                                                        f"� Current P&L: {pnl_pct:.1f}%")

                                            except Exception as e:
                                                st.warning(
                                                    f"Could not calculate P&L: {e}")

                                        else:
                                            # Single leg or complex structure
                                            for pos in spread_positions:
                                                symbol = pos.get(
                                                    'symbol', 'Unknown')
                                                qty = pos.get('quantity', 0)
                                                qty_direction = pos.get(
                                                    'quantity-direction', 'Unknown')
                                                avg_price = float(
                                                    pos.get('average-open-price', 0))
                                                close_price = float(
                                                    pos.get('close-price', avg_price))

                                                position_type = "🟢 LONG" if qty_direction == "Long" else "🔴 SHORT"

                                                col1, col2, col3 = st.columns(
                                                    3)
                                                col1.metric(
                                                    "Type", position_type)
                                                col2.metric("Quantity", qty)
                                                col3.metric(
                                                    "Avg Open", f"${avg_price:.2f}")

                                                with st.expander("📋 Full Details"):
                                                    st.json(pos)

                                # Overall summary
                                if spread_count > 0:
                                    st.write("---")
                                    st.write("### 💼 Portfolio Summary")
                                    summary_cols = st.columns(3)
                                    summary_cols[0].metric(
                                        "Total Spreads", spread_count)
                                    summary_cols[1].metric(
                                        "Total Positions", len(positions))
                                    summary_cols[2].metric("Total Unrealized P&L",
                                                           f"${total_unrealized_pnl:.2f}",
                                                           delta="Profit" if total_unrealized_pnl > 0 else "Loss")
                            else:
                                st.info("No open positions")

                            # Add OCO exit button if positions exist
                            if positions and len(positions) == 2:
                                st.write("\n")
                                st.subheader("🎯 Place OCO Exit Orders")
                                st.write(
                                    "Set profit target and stop loss for your open spread:")

                                oco_col1, oco_col2 = st.columns(2)
                                with oco_col1:
                                    oco_take_profit = st.slider(
                                        "Take Profit %",
                                        min_value=25,
                                        max_value=75,
                                        value=50,
                                        step=5,
                                        key="oco_tp",
                                        help="Close when profit reaches this % of credit"
                                    )
                                with oco_col2:
                                    oco_stop_loss = st.slider(
                                        "Stop Loss %",
                                        min_value=50,
                                        max_value=200,
                                        value=100,
                                        step=10,
                                        key="oco_sl",
                                        help="Close when loss reaches this % of credit"
                                    )

                                # Calculate exit prices - use stored spread data if available
                                if 'current_spread' in st.session_state:
                                    entry_credit = st.session_state['current_spread']['credit']
                                    stored_sell_leg = st.session_state['current_spread']['sell_leg']
                                    stored_buy_leg = st.session_state['current_spread']['buy_leg']
                                else:
                                    # Fallback to $2.99 if no stored data
                                    entry_credit = 2.99
                                    stored_sell_leg = None
                                    stored_buy_leg = None

                                tp_price = entry_credit * \
                                    (1 - oco_take_profit / 100.0)
                                sl_price = entry_credit * \
                                    (1 + oco_stop_loss / 100.0)

                                oco_info = st.columns(3)
                                oco_info[0].metric(
                                    "Entry Credit", f"${entry_credit:.2f}")
                                oco_info[1].metric("Take Profit At", f"${tp_price:.2f}",
                                                   help=f"Close at {oco_take_profit}% profit")
                                oco_info[2].metric("Stop Loss At", f"${sl_price:.2f}",
                                                   help=f"Close at {oco_stop_loss}% loss")

                                st.warning(
                                    "⚠️ **Sandbox Limitation**: OCO orders don't work in sandbox. This will place TWO SEPARATE orders instead.")
                                st.caption(
                                    "You'll need to manually cancel one order when the other fills.")

                                if st.button("🚀 Place Exit Orders (Separate)", type="primary", key="place_oco_btn", use_container_width=True):
                                    with st.spinner("Submitting exit orders..."):
                                        try:
                                            from integrations.tastytrade.orders import place_separate_exit_orders
                                            from integrations.tastytrade.account import get_account_numbers

                                            session = st.session_state.get(
                                                'auth_session')
                                            if session:
                                                account_numbers = get_account_numbers(
                                                    session)
                                                if account_numbers:
                                                    account_number = account_numbers[0]

                                                    # Check if we have stored spread data
                                                    if stored_sell_leg and stored_buy_leg:
                                                        # Use stored spread details - place separate orders
                                                        result = place_separate_exit_orders(
                                                            session=session,
                                                            account_number=account_number,
                                                            underlying="SPY",
                                                            sell_leg=stored_sell_leg,
                                                            buy_leg=stored_buy_leg,
                                                            num_contracts=3,  # Your current position size
                                                            entry_credit=entry_credit,
                                                            take_profit_pct=oco_take_profit / 100.0,
                                                            stop_loss_pct=oco_stop_loss / 100.0
                                                        )

                                                        st.success(
                                                            "✅ Exit orders placed successfully!")
                                                        st.json(result)
                                                        st.warning(
                                                            "⚠️ **Important**: These are TWO INDEPENDENT orders (not OCO). When one fills, you MUST manually cancel the other!")

                                                        # Store in session
                                                        st.session_state['last_oco_result'] = result
                                                    else:
                                                        st.error(
                                                            "❌ No spread data found. Please find a spread first.")

                                        except Exception as e:
                                            st.error(
                                                f"❌ OCO order failed: {e}")
                                            with st.expander("📋 Error Details"):
                                                import traceback
                                                st.code(traceback.format_exc())

                except Exception as e:
                    st.error(f"Error loading positions: {e}")

                st.write("\n")
                if avg_return_pct >= 20:
                    st.success(
                        "🎯 This spread meets the 20%+ return on risk threshold!")
                else:
                    st.warning(
                        "⚠️ Return below 20% - consider different strikes or expiration.")
            else:
                st.info(
                    "🔍 No suitable put credit spreads found with target criteria:")
                st.write("- **DTE:** 30-45 days")
                st.write("- **Sell Put δ:** ~-0.50")
                st.write("- **Buy Put δ:** ~-0.25")
                st.write("- **Target:** 20%+ return on risk")

    else:
        st.info("👆 Please authorize with Tastytrade above to view options data.")

with tab_backtest:
    st.subheader("Historical Backtest")
    st.caption("Uses daily SPY and VIX with Black-Scholes to approximate 45D vertical spreads on signal days. Indicative only.")

    # Always have a cached baseline history available for fallback
    with st.spinner("Loading data…"):
        spy_base, vix_base, spy_close_base, vix_close_base, _inds = load_market_data()

    # Backtest can benefit from a longer history than the main dashboard
    hist = st.selectbox("History window", options=[
                        "1y", "2y", "3y", "5y"], index=1, help="Download range for backtest only")

    # Strategy selection and core params
    strat = st.selectbox("Strategy", options=[
                         "Put credit spread", "Bull call debit spread"], index=0)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        use_dte_range = st.checkbox(
            "Use DTE range", value=False, help="Pick first expiry in range instead of exact DTE")
        if use_dte_range:
            dte_min = st.number_input(
                "DTE min", min_value=7, max_value=120, value=30, step=1)
            dte_max = st.number_input(
                "DTE max", min_value=7, max_value=120, value=45, step=1)
            dte = None
        else:
            dte = st.number_input("DTE (days)", min_value=7,
                                  max_value=120, value=45, step=1)
            dte_min, dte_max = None, None
    with c2:
        if strat == "Put credit spread":
            sell_delta = st.number_input(
                "Sell put delta", min_value=-0.95, max_value=-0.05, value=-0.50, step=0.05)
        else:
            buy_call_delta = st.number_input(
                "Buy call delta", min_value=0.05, max_value=0.95, value=0.40, step=0.05)
    with c3:
        if strat == "Put credit spread":
            buy_delta = st.number_input(
                "Buy put delta", min_value=-0.95, max_value=-0.05, value=-0.25, step=0.05)
        else:
            sell_call_delta = st.number_input(
                "Sell call delta", min_value=0.05, max_value=0.95, value=0.20, step=0.05)
    with c4:
        iv_scale = st.number_input(
            "IV scale (x VIX)", min_value=0.1, max_value=2.0, value=1.0, step=0.1)

    with st.expander("⚙️ Filters & Exit Strategy", expanded=False):
        st.write("**VIX regime**")
        vc1, vc2, vc3, vc4 = st.columns(4)
        with vc1:
            vix_threshold = st.number_input(
                "VIX <", min_value=5.0, max_value=80.0, value=20.0, step=0.5)
        with vc2:
            vix_ma_window = st.number_input(
                "VIX MA (days)", min_value=1, max_value=20, value=5)
        with vc3:
            require_vix_ma_down = st.checkbox("MA trending down", value=False)
        with vc4:
            avoid_vix_spike = st.checkbox("Avoid VIX spike", value=False)
        vs1, _ = st.columns(2)
        with vs1:
            vix_spike_pct = st.number_input(
                "Spike threshold %", min_value=0.01, max_value=0.20, value=0.05, step=0.01, help="VIX > (1+X) * MA")

        st.write("Entry timing")
        t1, t2, t3, t4 = st.columns(4)
        with t1:
            rsi_threshold = st.number_input(
                "RSI >", min_value=40.0, max_value=80.0, value=55.0, step=1.0)
        with t2:
            require_ema21_slope_up = st.checkbox("EMA21 slope up", value=False)
        with t3:
            require_price_above_ema21 = st.checkbox("Price > EMA21", value=False,
                                                    help="Require price above EMA21 (not in backtest by default)")
        with t4:
            require_macd_positive = st.checkbox("MACD > 0", value=False,
                                                help="Require positive MACD (not in backtest by default)")

        # Weekday selector
        weekday_map = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4}
        allowed_weekdays_labels = st.multiselect("Allowed weekdays", options=list(
            weekday_map.keys()), default=list(weekday_map.keys()))
        allowed_weekdays = [weekday_map[k] for k in allowed_weekdays_labels]

        st.write("Exit strategy (optional)")
        e1, e2, e3, e4 = st.columns(4)
        with e1:
            if strat == "Put credit spread":
                tp_pct = st.number_input("Take profit % of credit", min_value=0.0, max_value=2.0,
                                         value=0.0, step=0.05, help="0 disables; e.g., 0.5 means +50% of credit")
            else:
                tp_pct_debit = st.number_input("Take profit % of debit", min_value=0.0, max_value=2.0,
                                               value=0.0, step=0.05, help="0 disables; e.g., 0.5 means +50% of debit")
        with e2:
            if strat == "Put credit spread":
                sl_pct = st.number_input("Stop loss % of credit", min_value=0.0, max_value=3.0,
                                         value=0.0, step=0.1, help="0 disables; e.g., 1.0 means -100% of credit")
            else:
                sl_pct_debit = st.number_input("Stop loss % of debit", min_value=0.0, max_value=3.0,
                                               value=0.0, step=0.1, help="0 disables; e.g., 0.5 means -50% of debit")
        with e3:
            exit_after_days = st.number_input(
                "Exit after N trading days", min_value=0, max_value=120, value=0, step=1)
        with e4:
            exit_at_dte = st.number_input(
                "Exit at DTE", min_value=0, max_value=120, value=0, step=1)

    # Position sizing inputs
    st.write("#### 💰 Position Sizing")
    ps1, ps2 = st.columns(2)
    with ps1:
        account_size_input = st.number_input(
            "Account Size ($)",
            min_value=1000.0,
            value=10000.0,
            step=1000.0,
            help="Your total trading capital"
        )
    with ps2:
        risk_per_position_input = st.number_input(
            "Risk per Position (%)",
            min_value=0.5,
            max_value=10.0,
            value=1.7,
            step=0.5,
            help="What % of account to risk on each trade"
        )

    if st.button("Run backtest"):
        with st.spinner("Simulating trades…"):
            # Download dedicated history for backtest to avoid too-short windows
            spy_b = yf.download('SPY', period=hist, interval='1d',
                                auto_adjust=False, progress=False)
            vix_b = yf.download('^VIX', period=hist, interval='1d',
                                auto_adjust=False, progress=False)
            # Guard against None/empty
            if spy_b is None or getattr(spy_b, 'empty', True):
                st.warning(
                    "Could not download SPY for selected window; falling back to cached baseline (200d).")
                spy_close_bt = spy_close_base
            else:
                try:
                    spy_close_bt = get_close_series(spy_b, 'SPY')
                except Exception:
                    spy_close_bt = spy_close_base
            if vix_b is None or getattr(vix_b, 'empty', True):
                st.warning(
                    "Could not download VIX for selected window; falling back to cached baseline (200d).")
                vix_close_bt = vix_close_base
            else:
                try:
                    vix_close_bt = get_close_series(vix_b, '^VIX')
                except Exception:
                    vix_close_bt = vix_close_base

            # Debug: Show filter values being used
            st.info(
                f"🔍 Filter Debug: RSI>{rsi_threshold}, EMA21_slope={require_ema21_slope_up}, Price>EMA21={require_price_above_ema21}, MACD>0={require_macd_positive}")

            if strat == "Put credit spread":
                trades_df, equity, summary = run_backtest(
                    spy_close_bt,
                    vix_close_bt,
                    dte_days=int(dte) if dte is not None else 45,
                    dte_min_days=int(dte_min) if dte_min is not None else None,
                    dte_max_days=int(dte_max) if dte_max is not None else None,
                    sell_delta=float(sell_delta),
                    buy_delta=float(buy_delta),
                    iv_scale=float(iv_scale),
                    vix_threshold=float(vix_threshold),
                    vix_ma_window=int(vix_ma_window),
                    require_vix_ma_down=bool(require_vix_ma_down),
                    avoid_vix_spike=bool(avoid_vix_spike),
                    rsi_threshold=float(rsi_threshold),
                    require_ema21_slope_up=bool(require_ema21_slope_up),
                    require_price_above_ema21=bool(require_price_above_ema21),
                    require_macd_positive=bool(require_macd_positive),
                    allowed_weekdays=allowed_weekdays,
                    take_profit_pct_of_credit=(
                        float(tp_pct) if 'tp_pct' in locals() and tp_pct > 0 else None),
                    stop_loss_pct_of_credit=(
                        float(sl_pct) if 'sl_pct' in locals() and sl_pct > 0 else None),
                    exit_after_days_in_trade=(
                        int(exit_after_days) if exit_after_days > 0 else None),
                    exit_at_days_to_expiration=(
                        int(exit_at_dte) if exit_at_dte > 0 else None),
                )
            else:
                trades_df, equity, summary = run_backtest_bull_call(
                    spy_close_bt,
                    vix_close_bt,
                    dte_days=int(dte) if dte is not None else 45,
                    dte_min_days=int(dte_min) if dte_min is not None else None,
                    dte_max_days=int(dte_max) if dte_max is not None else None,
                    buy_delta=float(buy_call_delta),
                    sell_delta=float(sell_call_delta),
                    iv_scale=float(iv_scale),
                    vix_threshold=float(vix_threshold),
                    vix_ma_window=int(vix_ma_window),
                    require_vix_ma_down=bool(require_vix_ma_down),
                    avoid_vix_spike=bool(avoid_vix_spike),
                    rsi_threshold=float(rsi_threshold),
                    require_ema21_slope_up=bool(require_ema21_slope_up),
                    require_price_above_ema21=bool(require_price_above_ema21),
                    require_macd_positive=bool(require_macd_positive),
                    allowed_weekdays=allowed_weekdays,
                    take_profit_pct_of_debit=(float(
                        tp_pct_debit) if 'tp_pct_debit' in locals() and tp_pct_debit > 0 else None),
                    stop_loss_pct_of_debit=(float(
                        sl_pct_debit) if 'sl_pct_debit' in locals() and sl_pct_debit > 0 else None),
                    exit_after_days_in_trade=(
                        int(exit_after_days) if exit_after_days > 0 else None),
                    exit_at_days_to_expiration=(
                        int(exit_at_dte) if exit_at_dte > 0 else None),
                )

        if trades_df.empty:
            st.info("No trades generated. Try relaxing filters: allow all weekdays, RSI > 55, disable VIX MA down/spike, or extend history.")
        else:
            mcols = st.columns(5)
            mcols[0].metric("Trades", f"{summary['trades']}")
            mcols[1].metric("Win rate", f"{summary['win_rate']:.0%}")
            mcols[2].metric("Avg PnL", f"${summary['avg_pnl']:.2f}")
            mcols[3].metric("Total PnL", f"${summary['total_pnl']:.2f}")
            mcols[4].metric("Avg ROR", f"{summary['avg_ror']:.1%}")

            # Capital efficiency metrics
            st.write("#### 💰 Capital Efficiency")
            cap_cols = st.columns(3)
            cap_cols[0].metric("Avg Capital/Trade", f"${summary.get('avg_capital_per_trade', 0):.2f}",
                               help="Average max loss per trade (collateral for credit spread, debit paid for debit spread)")
            cap_cols[1].metric("Total Capital Required", f"${summary.get('total_capital_required', 0):.2f}",
                               help="Sum of all max losses if running all trades")
            cap_cols[2].metric("Return on Capital", f"{summary.get('return_on_capital', 0):.1%}",
                               help="Total PnL / Total Capital Required")

            # Compounding projection
            st.write("#### 📈 Compounding Potential")
            comp_cols = st.columns(2)
            comp_cols[0].metric("Avg Return %", f"{summary.get('avg_return_pct', 0):.2f}%",
                                help="Average return per trade as % of capital required")
            comp_cols[1].metric("Total Growth", f"{summary.get('compound_growth_pct', 0):.1f}%",
                                help="Total account growth if starting with avg capital and reinvesting all gains")

            st.caption("💡 Higher 'Avg Return %' compounds faster. A strategy with lower capital per trade and consistent returns will grow your account faster over time.")

            # Position sizing calculator
            st.write("#### 📊 Position Sizing Analysis")

            max_concurrent = summary.get('max_concurrent_positions', 0)
            st.info(
                f"📌 Peak Concurrency: At maximum, **{max_concurrent}** positions would be open simultaneously across the backtest period.")

            # Use the inputs from above
            account_size = account_size_input
            risk_per_position = risk_per_position_input

            # Calculate recommended position sizing
            avg_capital = summary.get('avg_capital_per_trade', 0)
            if avg_capital > 0:
                # Calculate how much capital to allocate per position based on risk %
                capital_per_position = account_size * \
                    (risk_per_position / 100.0)

                # Get total signals from summary
                total_signals = summary['trades']

                # Calculate what's needed to take ALL signals (minimum capital)
                capital_needed_for_all = avg_capital * max_concurrent
                max_total_deployment = account_size * 0.10

                # Check if we can take all signals within the 10% cap
                can_take_all_signals = capital_needed_for_all <= max_total_deployment

                if can_take_all_signals:
                    # Use minimum capital needed to take all signals
                    recommended_concurrent = max_concurrent
                    capital_per_position = avg_capital  # Use actual minimum needed
                    take_every_n = 1
                    positions_to_run = total_signals
                    total_deployed = capital_needed_for_all
                else:
                    # Can't take all - use risk % to determine how many we can take
                    max_positions_by_cap = int(
                        max_total_deployment / capital_per_position)
                    recommended_concurrent = min(
                        max_positions_by_cap, max_concurrent)
                    recommended_concurrent = max(1, recommended_concurrent)

                    if recommended_concurrent >= max_concurrent:
                        take_every_n = 1
                        positions_to_run = total_signals
                    else:
                        take_every_n = max(
                            1, int(total_signals / recommended_concurrent))
                        positions_to_run = int(total_signals / take_every_n)

                    total_deployed = capital_per_position * recommended_concurrent

                # Calculate actual deployment percentage
                deployment_pct = (total_deployed / account_size) * 100.0
                risk_pct_for_all = (
                    capital_needed_for_all / account_size) * 100.0

                # Show key metric
                st.metric(
                    "Recommended Concurrent Positions",
                    f"{recommended_concurrent}",
                    help=f"Based on ${capital_per_position:.2f} per position / ${avg_capital:.2f} avg capital needed"
                )

                # Show message about taking all signals
                if can_take_all_signals:
                    st.success(
                        f"✅ **Taking all signals!** You can capture all {summary['trades']} opportunities with {max_concurrent} concurrent positions using only {deployment_pct:.1f}% of your account (${total_deployed:.2f}).")
                else:
                    st.warning(
                        f"⚠️ **Can't take all signals with 10% cap.** Taking all {summary['trades']} signals needs {max_concurrent} positions = ${capital_needed_for_all:.2f} ({risk_pct_for_all:.1f}%). Increase account size or raise 10% cap to capture all opportunities.")

                st.write("##### Strategy Recommendations")
                rec_cols = st.columns(3)
                rec_cols[0].metric(
                    "Capital per Position",
                    f"${capital_per_position:.2f}",
                    help=f"Actual capital allocated per position"
                )
                rec_cols[1].metric(
                    "Take Every Nth Signal",
                    f"{take_every_n}",
                    help="To maintain target position count with available opportunities"
                )
                rec_cols[2].metric(
                    "Est. Positions per Year",
                    f"{positions_to_run}",
                    help=f"Taking {positions_to_run} out of {total_signals} total signals"
                )

                # Project returns
                scaled_pnl = summary['total_pnl'] * \
                    (positions_to_run / total_signals) if total_signals > 0 else 0
                # Use YOUR capital allocation, not the backtest's minimum requirement
                scaled_capital = capital_per_position * recommended_concurrent
                scaled_return = (
                    scaled_pnl / scaled_capital) if scaled_capital > 0 else 0

                st.write("##### 📈 Projected Performance")
                proj_cols = st.columns(4)
                proj_cols[0].metric(
                    "Total Capital Deployed",
                    f"${scaled_capital:.2f}",
                    help=f"{recommended_concurrent} positions × ${capital_per_position:.2f} (your allocation)"
                )
                proj_cols[1].metric(
                    "Account Utilization",
                    f"{deployment_pct:.1f}%",
                    help=f"${scaled_capital:.2f} / ${account_size:,.0f} (capped at 10%)"
                )
                proj_cols[2].metric(
                    "Projected Total PnL",
                    f"${scaled_pnl:.2f}",
                    help=f"Based on taking {positions_to_run} positions"
                )
                proj_cols[3].metric(
                    "Projected ROC",
                    f"{scaled_return:.1%}",
                    help="Projected PnL / Total Capital Deployed"
                )

                st.caption(f"💡 With a ${account_size:,.0f} account and {risk_per_position}% risk per trade, you would run up to {recommended_concurrent} concurrent positions (capped at 10% total deployment), taking every {take_every_n}{'st' if take_every_n == 1 else 'nd' if take_every_n == 2 else 'rd' if take_every_n == 3 else 'th'} signal.")

            # Strategy equity curve
            st.write("### Strategy Equity Curve")
            eq_df = equity.rename(
                "Cumulative P&L ($)").to_frame().reset_index()
            eq_df.columns = ['Date', 'Cumulative P&L ($)']

            chart = alt.Chart(eq_df).mark_line().encode(
                x=alt.X('Date:T', title='Date'),
                y=alt.Y('Cumulative P&L ($):Q', title='Cumulative P&L ($)',
                        scale=alt.Scale(zero=False)),
                tooltip=['Date:T', 'Cumulative P&L ($):Q']
            ).properties(height=400)
            st.altair_chart(chart, use_container_width=True)
            st.caption("📊 Cumulative profit/loss per spread over time")

            # SPY comparison chart
            if not trades_df.empty:
                st.write("### SPY Performance (Same Period)")
                backtest_start = trades_df['entry_date'].min()
                backtest_end = trades_df['expiry_date'].max()

                # Get SPY prices for the backtest period
                spy_backtest = spy_close_bt[(spy_close_bt.index >= backtest_start) &
                                            (spy_close_bt.index <= backtest_end)]

                if not spy_backtest.empty:
                    spy_df = spy_backtest.rename(
                        "SPY Price ($)").to_frame().reset_index()
                    spy_df.columns = ['Date', 'SPY Price ($)']

                    spy_chart = alt.Chart(spy_df).mark_line(color='green').encode(
                        x=alt.X('Date:T', title='Date'),
                        y=alt.Y('SPY Price ($):Q', title='SPY Price ($)',
                                scale=alt.Scale(zero=False)),
                        tooltip=['Date:T', 'SPY Price ($):Q']
                    ).properties(height=400)
                    st.altair_chart(spy_chart, use_container_width=True)

                    # Calculate SPY stats
                    spy_start = float(spy_backtest.iloc[0])
                    spy_end = float(spy_backtest.iloc[-1])
                    spy_return_pct = ((spy_end / spy_start) - 1.0) * 100.0

                    st.caption(
                        f"� SPY: ${spy_start:.2f} → ${spy_end:.2f} ({spy_return_pct:+.1f}% return)")

            with st.expander("Trades", expanded=False):
                df_show = trades_df.copy()
                # Format some columns
                for col in ("S_entry", "VIX_entry", "K_sell", "K_buy", "credit", "max_loss", "S_expiry", "pnl"):
                    if col in df_show.columns:
                        df_show[col] = df_show[col].map(
                            lambda x: round(float(x), 2))
                st.dataframe(df_show, use_container_width=True)
