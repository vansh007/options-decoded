import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime, timedelta
import time

from utils import (
    get_stock_info, get_historical_volatility, get_options_chain,
    fetch_option_chain_for_expiry, black_scholes_price, simulate_gbm_paths,
    monte_carlo_price, compute_greeks, implied_volatility,
    generate_signal, greek_explanation,
)

# ── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Options Intelligence",
    page_icon="◬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Design System — Refined Ink + Slate aesthetic ───────────────────────────
# Warm off-white paper tone, deep slate-navy, electric teal accents.
# Typeset with Syne (editorial display) + Source Code Pro (data/mono).
# Feels like a well-designed financial research report, not a dashboard.

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Source+Code+Pro:wght@300;400;500&family=Lora:ital,wght@0,400;0,500;1,400&display=swap');

:root {
    --ink:      #0F1923;
    --slate:    #1C2B3A;
    --slate2:   #243447;
    --teal:     #00C9A7;
    --teal-dim: #007A65;
    --teal-glow:#3DFFD9;
    --amber:    #FFB347;
    --red:      #FF6B6B;
    --paper:    #F4F1EC;
    --paper2:   #EAE6DF;
    --muted:    #8A9BB0;
    --border:   #2E3F52;
    --border-l: #D6CFC4;
    --text:     #F0EDE8;
    --text-dim: #7A8EA0;
}

*, html, body { box-sizing: border-box; }
body, [class*="css"], .stApp {
    font-family: 'Source Code Pro', monospace;
    background: var(--ink);
    color: var(--text);
}

/* ── Main container ── */
.block-container {
    padding: 2rem 3rem !important;
    max-width: 1400px !important;
}

/* ── Hero header ── */
.hero {
    text-align: center;
    padding: 3rem 0 2rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2.5rem;
}
.hero-eyebrow {
    font-family: 'Source Code Pro', monospace;
    font-size: 11px;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: var(--teal);
    margin-bottom: 0.75rem;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2.2rem, 5vw, 3.8rem);
    font-weight: 800;
    color: var(--text);
    line-height: 1.05;
    letter-spacing: -0.03em;
    margin: 0 0 0.5rem;
}
.hero-title span { color: var(--teal); }
.hero-sub {
    font-family: 'Lora', serif;
    font-size: 1rem;
    color: var(--text-dim);
    font-style: italic;
    margin: 0;
}

/* ── Search bar ── */
.stTextInput > div > div > input {
    background: var(--slate) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 4px !important;
    color: var(--text) !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 1.3rem !important;
    font-weight: 700 !important;
    padding: 0.75rem 1.2rem !important;
    letter-spacing: 0.05em !important;
    transition: border-color 0.2s !important;
}
.stTextInput > div > div > input:focus {
    border-color: var(--teal) !important;
    box-shadow: 0 0 0 3px rgba(0,201,167,0.12) !important;
}
.stTextInput > label {
    font-family: 'Source Code Pro', monospace !important;
    font-size: 10px !important;
    letter-spacing: 0.2em !important;
    text-transform: uppercase !important;
    color: var(--text-dim) !important;
}

/* ── Stock info bar ── */
.stock-bar {
    background: var(--slate);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 1.2rem 1.8rem;
    margin-bottom: 2rem;
    display: flex;
    align-items: center;
    gap: 2.5rem;
    flex-wrap: wrap;
}
.stock-name {
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--text);
}
.stock-price {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: var(--teal);
}
.stock-meta {
    font-size: 10px;
    color: var(--text-dim);
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
.stock-meta strong { color: var(--text); font-size: 12px; }

/* ── Section headers ── */
.section-head {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--text);
    letter-spacing: -0.01em;
    margin: 0 0 0.3rem;
}
.section-label {
    font-family: 'Source Code Pro', monospace;
    font-size: 9px;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: var(--teal-dim);
    margin-bottom: 1rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid var(--border);
}

/* ── Signal cards ── */
.signal-card {
    background: var(--slate);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
    border-left: 3px solid var(--teal);
}
.signal-card.warning { border-left-color: var(--amber); }
.signal-card.good { border-left-color: var(--teal); }
.signal-card.danger { border-left-color: var(--red); }
.signal-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.9rem;
    font-weight: 700;
    color: var(--text);
    margin-bottom: 0.4rem;
}
.signal-plain {
    font-size: 12px;
    color: var(--text-dim);
    line-height: 1.6;
    margin-bottom: 0.6rem;
}
.signal-tip {
    font-size: 11px;
    color: var(--teal);
    background: rgba(0,201,167,0.06);
    border-radius: 3px;
    padding: 0.5rem 0.8rem;
    line-height: 1.5;
}

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: var(--slate2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
    padding: 1rem 1.2rem !important;
}
[data-testid="metric-container"] label {
    font-family: 'Source Code Pro', monospace !important;
    font-size: 9px !important;
    letter-spacing: 0.18em !important;
    text-transform: uppercase !important;
    color: var(--text-dim) !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Syne', sans-serif !important;
    font-size: 1.5rem !important;
    font-weight: 700 !important;
    color: var(--teal-glow) !important;
}
[data-testid="stMetricDelta"] {
    font-family: 'Source Code Pro', monospace !important;
    font-size: 11px !important;
}

/* ── Greek cards ── */
.greek-card {
    background: var(--slate);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 1.2rem;
    margin-bottom: 1rem;
}
.greek-name {
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    color: var(--teal);
    margin-bottom: 0.2rem;
}
.greek-value {
    font-family: 'Source Code Pro', monospace;
    font-size: 1.4rem;
    color: var(--text);
    font-weight: 500;
    margin-bottom: 0.6rem;
}
.greek-what { font-size: 12px; color: var(--text); line-height: 1.5; margin-bottom: 0.4rem; }
.greek-analogy { font-size: 11px; color: var(--text-dim); line-height: 1.5; font-style: italic; margin-bottom: 0.4rem; }
.greek-watch {
    font-size: 10px;
    color: var(--amber);
    background: rgba(255,179,71,0.07);
    border-radius: 3px;
    padding: 0.4rem 0.6rem;
    line-height: 1.5;
}

/* ── Verdict badge ── */
.verdict {
    display: inline-block;
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 800;
    letter-spacing: 0.08em;
    padding: 0.3rem 1rem;
    border-radius: 3px;
    text-transform: uppercase;
}
.verdict-expensive { background: rgba(255,107,107,0.15); color: var(--red); border: 1px solid var(--red); }
.verdict-cheap { background: rgba(0,201,167,0.12); color: var(--teal); border: 1px solid var(--teal-dim); }
.verdict-neutral { background: rgba(138,155,176,0.15); color: var(--muted); border: 1px solid var(--border); }

/* ── Tabs ── */
[data-testid="stTab"] button {
    font-family: 'Source Code Pro', monospace !important;
    font-size: 10px !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    color: var(--text-dim) !important;
}
[data-testid="stTab"] button[aria-selected="true"] {
    color: var(--teal) !important;
    border-bottom-color: var(--teal) !important;
}

/* ── Selectbox / sliders ── */
.stSelectbox > div > div {
    background: var(--slate2) !important;
    border-color: var(--border) !important;
    border-radius: 4px !important;
    font-family: 'Source Code Pro', monospace !important;
    font-size: 12px !important;
}
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background: var(--teal) !important;
}

/* ── Info boxes ── */
.info-box {
    background: rgba(0,201,167,0.05);
    border: 1px solid rgba(0,201,167,0.2);
    border-radius: 6px;
    padding: 1rem 1.2rem;
    margin: 1rem 0;
    font-size: 12px;
    color: var(--text-dim);
    line-height: 1.6;
}
.info-box strong { color: var(--teal); }

/* ── Dividers ── */
hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--ink); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

/* ── Button ── */
.stButton > button {
    background: var(--teal) !important;
    color: var(--ink) !important;
    border: none !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 13px !important;
    letter-spacing: 0.05em !important;
    padding: 0.6rem 2rem !important;
    border-radius: 3px !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: var(--teal-glow) !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(0,201,167,0.3) !important;
}

/* ── Toggle ── */
.stToggle label {
    font-family: 'Source Code Pro', monospace !important;
    font-size: 11px !important;
    color: var(--text-dim) !important;
    letter-spacing: 0.1em !important;
}

/* No top padding for first element */
.main > div:first-child { padding-top: 0 !important; }

/* ── Mode banner ── */
.mode-banner {
    background: linear-gradient(135deg, rgba(0,201,167,0.08), rgba(0,201,167,0.02));
    border: 1px solid rgba(0,201,167,0.2);
    border-radius: 6px;
    padding: 0.7rem 1.2rem;
    margin-bottom: 1.5rem;
    font-size: 11px;
    color: var(--teal);
    letter-spacing: 0.05em;
}

/* Hide streamlit header/footer */
#MainMenu { visibility: hidden; }
header { visibility: hidden; }
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Plotly theme ─────────────────────────────────────────────────────────────
PLOT_BG = "#0F1923"
PLOT_SURFACE = "#1C2B3A"
TEAL = "#00C9A7"
TEAL_DIM = "#007A65"
AMBER = "#FFB347"
RED = "#FF6B6B"
MUTED = "#3D5470"
TEXT_DIM = "#7A8EA0"
BORDER = "#2E3F52"

def plot_layout(**kwargs):
    base = dict(
        paper_bgcolor=PLOT_BG,
        plot_bgcolor=PLOT_SURFACE,
        font=dict(family="Source Code Pro, monospace", color=TEXT_DIM, size=10),
        xaxis=dict(gridcolor=MUTED, gridwidth=0.5, linecolor=BORDER, tickcolor=TEXT_DIM, showgrid=True),
        yaxis=dict(gridcolor=MUTED, gridwidth=0.5, linecolor=BORDER, tickcolor=TEXT_DIM, showgrid=True),
        margin=dict(t=30, b=40, l=10, r=10),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=BORDER, borderwidth=1, font=dict(size=10)),
    )
    base.update(kwargs)
    return base


# ── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-eyebrow">◬ Real-Time Options Intelligence</div>
  <h1 class="hero-title">Options<span> Decoded</span></h1>
  <p class="hero-sub">Understand exactly what the market is pricing — and what it means for you.</p>
</div>
""", unsafe_allow_html=True)


# ── Beginner mode toggle ──────────────────────────────────────────────────────
col_mode, col_spacer = st.columns([1, 3])
with col_mode:
    beginner_mode = st.toggle("📖 Plain-English Mode", value=True)

if beginner_mode:
    st.markdown('<div class="mode-banner">📖 Plain-English Mode is ON — every number will be explained in simple terms.</div>', unsafe_allow_html=True)


# ── Ticker search ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Enter Any Stock Ticker</div>', unsafe_allow_html=True)
col_search, col_btn, col_ex = st.columns([3, 1, 3])
with col_search:
    ticker_input = st.text_input("", value="AAPL", placeholder="e.g. AAPL, TSLA, MSFT, GOOGL", label_visibility="collapsed")
with col_btn:
    search_btn = st.button("ANALYSE →", use_container_width=True)
with col_ex:
    st.markdown('<div style="padding:0.6rem 0; font-size:11px; color:#3D5470;">Try: AAPL · TSLA · MSFT · NVDA · SPY · META</div>', unsafe_allow_html=True)

ticker = ticker_input.strip().upper()

# ── Fetch data ────────────────────────────────────────────────────────────────
if ticker:
    with st.spinner(f"Fetching live data for {ticker}..."):
        stock_info = get_stock_info(ticker)
        hv_value, hv_history = get_historical_volatility(ticker)
        ticker_obj, expirations = get_options_chain(ticker)

    if not stock_info or not stock_info.get("price"):
        st.error(f"Could not find data for **{ticker}**. Check the ticker symbol and try again.")
        st.stop()

    S0 = stock_info["price"]

    # ── Stock info bar ────────────────────────────────────────────────────────
    price_52_pct = ""
    if stock_info.get("week_52_high") and stock_info.get("week_52_low"):
        rng = stock_info["week_52_high"] - stock_info["week_52_low"]
        pos = (S0 - stock_info["week_52_low"]) / rng * 100 if rng > 0 else 50
        price_52_pct = f"<div class='stock-meta'>52W RANGE<br><strong>{pos:.0f}% of range</strong></div>"

    mc_str = ""
    if stock_info.get("market_cap"):
        mc = stock_info["market_cap"]
        mc_str = f"{'${:,.0f}B'.format(mc/1e9) if mc >= 1e9 else '${:,.0f}M'.format(mc/1e6)}"

    st.markdown(f"""
    <div class="stock-bar">
        <div>
            <div class="stock-name">{stock_info['name']}</div>
            <div class="stock-meta">NYSE/NASDAQ · {stock_info.get('sector','—')}</div>
        </div>
        <div class="stock-price">${S0:,.2f}</div>
        <div class="stock-meta">HV (30d)<br><strong>{f"{hv_value*100:.1f}%" if hv_value else "—"}</strong></div>
        <div class="stock-meta">MKT CAP<br><strong>{mc_str if mc_str else "—"}</strong></div>
        {price_52_pct}
        <div class="stock-meta">UPDATED<br><strong>{datetime.now().strftime("%H:%M:%S")}</strong></div>
    </div>
    """, unsafe_allow_html=True)

    if beginner_mode:
        st.markdown(f"""
        <div class="info-box">
            <strong>What am I looking at?</strong> This is the live price of {stock_info['name']} stock.
            The <strong>HV (30d)</strong> is Historical Volatility — how much the stock has actually been moving over the past 30 days,
            expressed as an annualized percentage. We'll compare this to what the options market is <em>implying</em> will happen — and that gap is where the real insight lives.
        </div>
        """, unsafe_allow_html=True)

    # ── Option controls ───────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-label">Configure Your Option</div>', unsafe_allow_html=True)

    if expirations:
        exp_days = []
        for e in expirations:
            d = (datetime.strptime(e, "%Y-%m-%d") - datetime.today()).days
            exp_days.append(f"{e}  ({d}d)")
        col1, col2, col3, col4 = st.columns([2, 1.5, 1.5, 1.5])
        with col1:
            exp_choice_idx = st.selectbox("Expiry Date", range(len(expirations)),
                format_func=lambda i: exp_days[i], key="expiry")
            chosen_expiry = expirations[exp_choice_idx]
            T_days = (datetime.strptime(chosen_expiry, "%Y-%m-%d") - datetime.today()).days
            T = T_days / 365
        with col2:
            option_type = st.selectbox("Option Type", ["call", "put"],
                format_func=lambda x: "📈 Call (bet on UP)" if x == "call" else "📉 Put (bet on DOWN)")
        with col3:
            r = st.slider("Risk-free Rate", 0.0, 0.10, 0.05, 0.005, format="%.3f",
                help="Current interest rate (US Fed rate ~5%)")
        with col4:
            n_sim = st.select_slider("Simulation Paths", [5000, 10000, 25000, 50000], 25000,
                help="More paths = more accurate Monte Carlo, but slower")
    else:
        st.warning("No options chain available for this ticker. Using manual parameters.")
        col1, col2, col3 = st.columns(3)
        with col1:
            T = st.slider("Time to Maturity (years)", 0.05, 2.0, 0.25, 0.05)
            T_days = T * 365
        with col2:
            option_type = st.selectbox("Option Type", ["call", "put"])
        with col3:
            r = st.slider("Risk-free Rate", 0.0, 0.10, 0.05, 0.005, format="%.3f")
        n_sim = 25000
        chosen_expiry = None

    if beginner_mode:
        call_put_explain = (
            "A **Call** option gives you the right to BUY the stock at the strike price. You profit if the stock goes UP."
            if option_type == "call" else
            "A **Put** option gives you the right to SELL the stock at the strike price. You profit if the stock goes DOWN."
        )
        st.markdown(f'<div class="info-box">📘 {call_put_explain}</div>', unsafe_allow_html=True)

    # ── Fetch real options chain ──────────────────────────────────────────────
    calls_df, puts_df = None, None
    if ticker_obj and chosen_expiry:
        with st.spinner("Loading options chain..."):
            calls_df, puts_df = fetch_option_chain_for_expiry(ticker_obj, chosen_expiry, S0)

    chain_df = calls_df if option_type == "call" else puts_df

    # Strike selector
    st.markdown('<div class="section-label" style="margin-top:1rem">Select Strike Price</div>', unsafe_allow_html=True)
    if chain_df is not None and not chain_df.empty:
        available_strikes = sorted(chain_df['strike'].tolist())
        atm_idx = min(range(len(available_strikes)), key=lambda i: abs(available_strikes[i] - S0))
        strike_col, info_col = st.columns([2, 3])
        with strike_col:
            K = st.select_slider("Strike Price", available_strikes, value=available_strikes[atm_idx],
                format_func=lambda x: f"${x:.2f}")
        with info_col:
            moneyness = (S0 - K) / K * 100 if option_type == "call" else (K - S0) / K * 100
            status = "IN-THE-MONEY 💰" if moneyness > 0 else ("AT-THE-MONEY ⚖️" if abs(moneyness) < 1 else "OUT-OF-THE-MONEY 📭")
            color = TEAL if moneyness > 0 else (AMBER if abs(moneyness) < 1 else RED)
            st.markdown(f"""
            <div style="padding:0.8rem 1rem; background:var(--slate2); border-radius:4px; border:1px solid var(--border); margin-top:1.5rem">
                <div style="font-size:9px; letter-spacing:0.15em; text-transform:uppercase; color:{color};">{status}</div>
                <div style="font-family:'Syne',sans-serif; font-size:1rem; font-weight:700; color:var(--text); margin-top:0.2rem;">
                    Stock is {abs(moneyness):.1f}% {'above' if moneyness > 0 else 'below'} strike
                </div>
                {'<div style="font-size:11px;color:var(--text-dim);margin-top:0.3rem;">This option has intrinsic value right now.</div>' if moneyness > 0 else ''}
            </div>
            """, unsafe_allow_html=True)

        # Get real market price for this strike
        row = chain_df[chain_df['strike'] == K]
        market_price = float(row['lastPrice'].iloc[0]) if not row.empty else None
        market_iv = float(row['impliedVolatility'].iloc[0]) if not row.empty and 'impliedVolatility' in row.columns else None
    else:
        K = st.slider("Strike Price", float(S0 * 0.7), float(S0 * 1.3), float(S0), float(S0 * 0.01))
        market_price = None
        market_iv = None

    # ── RUN PRICING ───────────────────────────────────────────────────────────
    st.markdown("---")

    sigma_for_pricing = market_iv if market_iv and market_iv > 0 else (hv_value if hv_value else 0.25)
    iv_to_use = market_iv if market_iv and market_iv > 0 else None

    bs_price = black_scholes_price(S0, K, r, sigma_for_pricing, T, option_type)
    mc_price_val, mc_se = monte_carlo_price(S0, K, r, sigma_for_pricing, T, n_sim, option_type)
    greeks = compute_greeks(S0, K, r, sigma_for_pricing, T, option_type)

    if market_price and not iv_to_use:
        iv_to_use = implied_volatility(market_price, S0, K, r, T, option_type)

    signals, verdict, verdict_color, iv_hv_ratio = generate_signal(
        bs_price, market_price, iv_to_use, hv_value,
        greeks["delta"], greeks["theta"], T_days, option_type
    )

    # ── TABS ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "  ◈ SIGNAL & PRICING  ",
        "  ◈ GREEKS EXPLAINED  ",
        "  ◈ PRICE PATHS  ",
        "  ◈ FULL OPTIONS CHAIN  ",
    ])

    # ════════════════════════════════════════════════════════════════════════
    # TAB 1 — Signal & Pricing
    # ════════════════════════════════════════════════════════════════════════
    with tab1:
        # Verdict
        verdict_class = f"verdict-{verdict.lower()}"
        st.markdown(f"""
        <div style="display:flex; align-items:center; gap:1rem; margin-bottom:1.5rem;">
            <div class="section-head">Overall Verdict</div>
            <div class="verdict {verdict_class}">{verdict}</div>
        </div>
        """, unsafe_allow_html=True)

        if beginner_mode:
            verdict_explain = {
                "EXPENSIVE": f"The market is charging more for this option than historical price moves would justify. As a buyer, you're paying a premium. As a seller, you may be getting a good price.",
                "CHEAP": f"This option is priced below what historical volatility would suggest. Buyers may be getting a bargain. Sellers are giving away more than they might realize.",
                "NEUTRAL": "Pricing looks fair based on historical data. No strong edge in either direction from valuation alone."
            }
            st.markdown(f'<div class="info-box">🎯 <strong>What this means for you:</strong> {verdict_explain.get(verdict, "")}</div>', unsafe_allow_html=True)

        # Key metrics
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Model Price (BS)", f"${bs_price:.3f}",
            help="Black-Scholes analytical fair value")
        c2.metric("Monte Carlo Price", f"${mc_price_val:.3f}", f"±{mc_se:.3f} error",
            help="Simulation-based price using 30,000 random paths")
        if market_price:
            diff = market_price - bs_price
            c3.metric("Market Price", f"${market_price:.3f}", f"{diff:+.3f} vs model",
                delta_color="inverse" if diff > 0 else "normal")
        if iv_to_use:
            c4.metric("Implied Volatility", f"{iv_to_use*100:.1f}%",
                f"{'↑' if hv_value and iv_to_use > hv_value else '↓'} vs {hv_value*100:.1f}% HV" if hv_value else None)
        if hv_value:
            c5.metric("Historical Vol (30d)", f"{hv_value*100:.1f}%")

        if beginner_mode:
            st.markdown("""
            <div class="info-box">
                <strong>How to read these numbers:</strong><br>
                • <strong>Model Price (BS)</strong> — What a mathematical formula says this option is "worth" based on inputs.<br>
                • <strong>Monte Carlo Price</strong> — Same idea, but computed by simulating thousands of possible futures. The ±error shows how precise it is.<br>
                • <strong>Market Price</strong> — What the market is actually charging right now. The gap between market and model = opportunity or trap.<br>
                • <strong>Implied Volatility (IV)</strong> — The market's forecast of future volatility, baked into the option price.<br>
                • <strong>Historical Vol (HV)</strong> — How much the stock has <em>actually</em> moved. IV vs HV is the most important comparison here.
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # IV vs HV visual
        if iv_to_use and hv_value:
            st.markdown('<div class="section-label">Implied vs Historical Volatility</div>', unsafe_allow_html=True)
            fig_vol = go.Figure()
            if hv_history is not None and len(hv_history) > 0:
                fig_vol.add_trace(go.Scatter(
                    x=list(range(len(hv_history))), y=(hv_history * 100).values,
                    mode="lines", name="Historical Vol (30d rolling)",
                    line=dict(color=TEAL, width=1.5),
                    fill="tozeroy", fillcolor="rgba(0,201,167,0.06)"
                ))
            fig_vol.add_hline(y=iv_to_use * 100, line=dict(color=AMBER, width=2, dash="dash"),
                annotation_text=f"  IV = {iv_to_use*100:.1f}%", annotation_font=dict(color=AMBER, size=11))
            fig_vol.update_layout(**plot_layout(
                height=220, xaxis_title="Days Ago (approx)", yaxis_title="Volatility (%)",
                title=dict(text="Is the market pricing in MORE or LESS volatility than history?",
                           font=dict(size=11, color=TEXT_DIM))
            ))
            st.plotly_chart(fig_vol, use_container_width=True)

            if beginner_mode:
                if iv_hv_ratio and iv_hv_ratio > 1.15:
                    st.markdown(f'<div class="info-box">📈 The orange line (IV = {iv_to_use*100:.1f}%) sits <strong>above</strong> the historical volatility. The market is forecasting bigger moves than have actually happened. Option buyers are paying a <strong>fear premium</strong>.</div>', unsafe_allow_html=True)
                elif iv_hv_ratio and iv_hv_ratio < 0.85:
                    st.markdown(f'<div class="info-box">📉 The orange line (IV = {iv_to_use*100:.1f}%) sits <strong>below</strong> historical volatility. The market is unusually calm — options are cheap relative to how the stock has been moving.</div>', unsafe_allow_html=True)

        st.markdown("---")

        # Signals
        st.markdown('<div class="section-label">Intelligence Signals</div>', unsafe_allow_html=True)
        for sig in signals:
            level_class = sig.get("level", "neutral")
            st.markdown(f"""
            <div class="signal-card {level_class}">
                <div class="signal-title">{sig['icon']} {sig['title']}</div>
                <div class="signal-plain">{sig['plain']}</div>
                <div class="signal-tip">💡 Trader Tip: {sig['trader_tip']}</div>
            </div>
            """, unsafe_allow_html=True)

        # Payoff diagram
        st.markdown("---")
        st.markdown('<div class="section-label">Payoff at Expiry</div>', unsafe_allow_html=True)
        S_range = np.linspace(max(1, S0 * 0.4), S0 * 1.8, 400)
        intrinsic = np.maximum(S_range - K, 0) if option_type == "call" else np.maximum(K - S_range, 0)
        bs_curve = [black_scholes_price(s, K, r, sigma_for_pricing, T, option_type) for s in S_range]
        premium_paid = market_price or bs_price
        pnl = intrinsic - premium_paid

        fig_payoff = go.Figure()
        fig_payoff.add_trace(go.Scatter(x=S_range, y=pnl,
            mode="lines", name="P&L if held to expiry",
            line=dict(color=TEAL, width=2),
            fill="tozeroy", fillcolor="rgba(0,201,167,0.05)"))
        fig_payoff.add_trace(go.Scatter(x=S_range, y=bs_curve,
            mode="lines", name=f"Current option value (T={T_days:.0f}d left)",
            line=dict(color=AMBER, width=1.5, dash="dot")))
        fig_payoff.add_hline(y=0, line=dict(color=BORDER, width=1))
        fig_payoff.add_vline(x=S0, line=dict(color=TEAL, width=1.5, dash="dash"),
            annotation_text=f"  Current: ${S0:.2f}", annotation_font=dict(color=TEAL, size=10))
        fig_payoff.add_vline(x=K, line=dict(color=RED, width=1, dash="dash"),
            annotation_text=f"  Strike: ${K:.2f}", annotation_font=dict(color=RED, size=10))

        breakeven = K + premium_paid if option_type == "call" else K - premium_paid
        fig_payoff.add_vline(x=breakeven, line=dict(color=AMBER, width=1, dash="dot"),
            annotation_text=f"  Breakeven: ${breakeven:.2f}", annotation_font=dict(color=AMBER, size=10))

        fig_payoff.update_layout(**plot_layout(
            height=320, xaxis_title="Stock Price at Expiry", yaxis_title="Profit / Loss ($)",
            title=dict(text=f"You break even if {ticker} {'rises above' if option_type=='call' else 'falls below'} ${breakeven:.2f} by {chosen_expiry or 'expiry'}",
                       font=dict(size=11, color=TEXT_DIM))
        ))
        st.plotly_chart(fig_payoff, use_container_width=True)

        if beginner_mode:
            st.markdown(f"""
            <div class="info-box">
                📊 <strong>Reading the payoff chart:</strong><br>
                • The <strong>teal line</strong> shows your profit/loss <em>at expiry</em> for every possible stock price.<br>
                • You start losing money below <strong>${breakeven:.2f}</strong> (breakeven) — that's the premium you paid working against you.<br>
                • The <strong>dotted amber line</strong> shows the option's value <em>today</em> with {T_days:.0f} days left — you could sell before expiry.<br>
                • Maximum loss = ${premium_paid:.2f} per share (the premium you paid). Maximum gain = {'unlimited upside' if option_type=='call' else f'up to ${K:.2f} if stock goes to $0'}.
            </div>
            """, unsafe_allow_html=True)


    # ════════════════════════════════════════════════════════════════════════
    # TAB 2 — Greeks Explained
    # ════════════════════════════════════════════════════════════════════════
    with tab2:
        st.markdown("""
        <div class="info-box" style="margin-bottom:1.5rem">
            <strong>What are the Greeks?</strong> Options don't just have a price — they have <em>sensitivities</em>.
            The Greeks measure how your option's value changes in response to different market forces:
            stock price moves, volatility changes, time passing, and interest rate shifts.
            Understanding them lets you know <em>exactly what you're betting on</em>.
        </div>
        """, unsafe_allow_html=True)

        greek_data = [
            ("Delta  Δ", "delta", greeks["delta"]),
            ("Gamma  Γ", "gamma", greeks["gamma"]),
            ("Vega  ν", "vega", greeks["vega"]),
            ("Theta  Θ", "theta", greeks["theta"]),
            ("Rho  ρ", "rho", greeks["rho"]),
        ]

        col_left, col_right = st.columns(2)
        for i, (display_name, key, value) in enumerate(greek_data):
            exp = greek_explanation(key, value, option_type, S0, K)
            card_html = f"""
            <div class="greek-card">
                <div class="greek-name">{display_name}</div>
                <div class="greek-value">{value:+.4f}</div>
                <div class="greek-what">📌 {exp.get('what', '')}</div>
                {'<div class="greek-analogy">💭 ' + exp.get('analogy','') + '</div>' if beginner_mode else ''}
                <div class="greek-watch">⚡ {exp.get('watch','')}</div>
            </div>
            """
            if i % 2 == 0:
                col_left.markdown(card_html, unsafe_allow_html=True)
            else:
                col_right.markdown(card_html, unsafe_allow_html=True)

        st.markdown("---")

        # Greeks sensitivity plot
        st.markdown('<div class="section-label">How Greeks Change as Stock Price Moves</div>', unsafe_allow_html=True)
        S_vals = np.linspace(max(1, S0 * 0.5), S0 * 1.5, 100)
        delta_vals = [compute_greeks(s, K, r, sigma_for_pricing, T, option_type)["delta"] for s in S_vals]
        gamma_vals = [compute_greeks(s, K, r, sigma_for_pricing, T, option_type)["gamma"] for s in S_vals]
        vega_vals = [compute_greeks(s, K, r, sigma_for_pricing, T, option_type)["vega"] for s in S_vals]
        theta_vals = [compute_greeks(s, K, r, sigma_for_pricing, T, option_type)["theta"] for s in S_vals]

        fig_greeks = make_subplots(rows=2, cols=2,
            subplot_titles=["Delta (Δ) — Direction sensitivity",
                            "Gamma (Γ) — Acceleration",
                            "Vega (ν) — Vol sensitivity",
                            "Theta (Θ) — Daily time decay"],
            vertical_spacing=0.18, horizontal_spacing=0.08)

        pairs = [(delta_vals, TEAL, 1, 1), (gamma_vals, AMBER, 1, 2),
                 (vega_vals, "#6E9EFF", 2, 1), (theta_vals, RED, 2, 2)]
        for data, color, row, col in pairs:
            fig_greeks.add_trace(go.Scatter(
                x=S_vals, y=data, mode="lines",
                line=dict(color=color, width=2),
                fill="tozeroy", fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.06)",
                showlegend=False
            ), row=row, col=col)
            fig_greeks.add_vline(x=S0, row=row, col=col,
                line=dict(color=TEXT_DIM, width=1, dash="dot"))

        fig_greeks.update_layout(**plot_layout(height=500, showlegend=False))
        fig_greeks.update_annotations(font=dict(family="Source Code Pro, monospace", color=TEXT_DIM, size=10))
        for i in range(1, 5):
            ax = "xaxis" if i == 1 else f"xaxis{i}"
            ay = "yaxis" if i == 1 else f"yaxis{i}"
            fig_greeks.update_layout(**{
                ax: dict(gridcolor=MUTED, gridwidth=0.5, linecolor=BORDER, tickcolor=TEXT_DIM),
                ay: dict(gridcolor=MUTED, gridwidth=0.5, linecolor=BORDER, tickcolor=TEXT_DIM),
            })
        st.plotly_chart(fig_greeks, use_container_width=True)

        if beginner_mode:
            st.markdown("""
            <div class="info-box">
                The dotted vertical line = current stock price. Notice how Delta (top-left) is an S-curve —
                it goes from 0 (deep out-of-the-money) to 1 (deep in-the-money). The steepest point of Delta
                is where Gamma (top-right) peaks — right at the strike. This is where options are most explosive.
            </div>
            """, unsafe_allow_html=True)


    # ════════════════════════════════════════════════════════════════════════
    # TAB 3 — Price Paths
    # ════════════════════════════════════════════════════════════════════════
    with tab3:
        st.markdown(f"""
        <div class="info-box">
            <strong>What is this?</strong> Monte Carlo simulation — we generate {n_sim:,} possible futures for {ticker}'s stock price
            using real volatility data ({sigma_for_pricing*100:.1f}%). Each line is one possible path the stock could take.
            The teal paths end above the strike (your option would be profitable). Red paths end below.
        </div>
        """, unsafe_allow_html=True)

        with st.spinner("Running simulation..."):
            t_axis, paths = simulate_gbm_paths(S0, r, sigma_for_pricing, T, 252, 300)

        n_show = st.slider("Paths to display", 20, 200, 60, 10)
        S_T_all = paths[:, -1]
        pct_profitable = np.mean(S_T_all > K) * 100 if option_type == "call" else np.mean(S_T_all < K) * 100

        c1, c2, c3 = st.columns(3)
        c1.metric("Paths ending profitable", f"{pct_profitable:.1f}%",
            help=f"% of simulated paths where stock is {'above' if option_type=='call' else 'below'} strike at expiry")
        c2.metric("Expected stock price", f"${np.mean(S_T_all):,.2f}")
        c3.metric("Simulated paths", f"{n_sim:,}")

        fig_paths = go.Figure()
        for i in range(min(n_show, paths.shape[0])):
            profitable = paths[i, -1] > K if option_type == "call" else paths[i, -1] < K
            color = "rgba(0,201,167,0.2)" if profitable else "rgba(255,107,107,0.15)"
            fig_paths.add_trace(go.Scatter(
                x=t_axis, y=paths[i],
                mode="lines", line=dict(color=color, width=0.7),
                showlegend=False
            ))

        mean_path = np.mean(paths[:n_show], axis=0)
        fig_paths.add_trace(go.Scatter(x=t_axis, y=mean_path,
            mode="lines", line=dict(color=AMBER, width=2.5), name="Average path"))
        fig_paths.add_hline(y=K, line=dict(color=RED, width=1.5, dash="dash"),
            annotation_text=f"  Strike ${K:.2f}", annotation_font=dict(color=RED, size=10))
        fig_paths.add_hline(y=S0, line=dict(color=TEXT_DIM, width=1, dash="dot"),
            annotation_text=f"  Start ${S0:.2f}", annotation_font=dict(color=TEXT_DIM, size=10))

        fig_paths.update_layout(**plot_layout(
            height=430, xaxis_title="Time (years)", yaxis_title=f"{ticker} Stock Price",
            title=dict(text=f"Teal = profitable paths | Red = loss paths | {pct_profitable:.1f}% end profitable",
                       font=dict(size=11, color=TEXT_DIM))
        ))
        st.plotly_chart(fig_paths, use_container_width=True)

        # Terminal distribution
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(x=S_T_all, nbinsx=80,
            marker=dict(color=TEAL, line=dict(color=PLOT_BG, width=0.5)),
            opacity=0.7, name="Simulated final prices"))
        fig_dist.add_vline(x=K, line=dict(color=RED, width=2, dash="dash"),
            annotation_text=f"  Strike ${K:.2f}", annotation_font=dict(color=RED, size=10))
        fig_dist.add_vline(x=S0, line=dict(color=AMBER, width=1.5, dash="dot"),
            annotation_text=f"  Today ${S0:.2f}", annotation_font=dict(color=AMBER, size=10))
        fig_dist.update_layout(**plot_layout(
            height=250, xaxis_title=f"{ticker} Price at Expiry ({chosen_expiry})",
            yaxis_title="Number of simulated paths",
            title=dict(text="Distribution of all possible stock prices at expiry", font=dict(size=11, color=TEXT_DIM))
        ))
        st.plotly_chart(fig_dist, use_container_width=True)

        if beginner_mode:
            st.markdown(f"""
            <div class="info-box">
                The histogram shows every possible outcome across {n_sim:,} simulated futures.
                The area to the {'right' if option_type=='call' else 'left'} of the red strike line
                represents profitable outcomes — that's {pct_profitable:.1f}% of all paths.
                This is the Monte Carlo estimate of the option's probability of profit.
                Compare this to Delta (~{abs(greeks['delta'])*100:.0f}%) — they should be similar!
            </div>
            """, unsafe_allow_html=True)


    # ════════════════════════════════════════════════════════════════════════
    # TAB 4 — Full Options Chain
    # ════════════════════════════════════════════════════════════════════════
    with tab4:
        if chain_df is not None and not chain_df.empty:
            st.markdown(f'<div class="section-label">Live {option_type.upper()}S Chain — {ticker} · Expiry {chosen_expiry}</div>', unsafe_allow_html=True)

            if beginner_mode:
                st.markdown("""
                <div class="info-box">
                    <strong>Reading the options chain:</strong><br>
                    • <strong>Strike</strong> — The price you'd have the right to buy/sell at.<br>
                    • <strong>Last Price</strong> — What the option last traded for (per share, multiply by 100 for 1 contract).<br>
                    • <strong>IV</strong> — Implied Volatility. Higher IV = more expensive option = market expects bigger moves.<br>
                    • <strong>Volume</strong> — How many contracts traded today. Higher = more liquid = easier to buy/sell.<br>
                    • <strong>Open Interest</strong> — Total open contracts. Large OI at a strike = that's a key level traders watch.<br>
                    • <strong>Model Fair Value</strong> — What our Black-Scholes model says it's worth. Compare to Last Price to find mispricing.
                </div>
                """, unsafe_allow_html=True)

            display_cols = ['strike', 'lastPrice', 'bid', 'ask', 'impliedVolatility', 'volume', 'openInterest']
            display_cols = [c for c in display_cols if c in chain_df.columns]
            chain_display = chain_df[display_cols].copy()

            # Add model price column
            chain_display['modelPrice'] = chain_display['strike'].apply(
                lambda k: round(black_scholes_price(S0, k, r, sigma_for_pricing, T, option_type), 3)
            )
            chain_display['mispricing%'] = ((chain_display['lastPrice'] - chain_display['modelPrice'])
                                             / chain_display['modelPrice'] * 100).round(1)

            if 'impliedVolatility' in chain_display.columns:
                chain_display['impliedVolatility'] = (chain_display['impliedVolatility'] * 100).round(1).astype(str) + '%'
            chain_display.columns = [c.replace('impliedVolatility','IV %').replace('lastPrice','Last $')
                                      .replace('openInterest','OI').replace('modelPrice','Model $')
                                      .replace('mispricing%','Edge %') for c in chain_display.columns]
            chain_display['strike'] = chain_display['strike'].apply(lambda x: f"${x:.2f}")

            st.dataframe(chain_display.set_index('strike'), use_container_width=True, height=420)

            # IV Smile
            st.markdown("---")
            st.markdown('<div class="section-label">Volatility Smile — Real Market Data</div>', unsafe_allow_html=True)
            if 'impliedVolatility' in chain_df.columns:
                iv_clean = chain_df[['strike', 'impliedVolatility']].dropna()
                iv_clean = iv_clean[iv_clean['impliedVolatility'] > 0]
                if not iv_clean.empty:
                    fig_smile = go.Figure()
                    fig_smile.add_trace(go.Scatter(
                        x=iv_clean['strike'], y=iv_clean['impliedVolatility'] * 100,
                        mode="lines+markers",
                        line=dict(color=TEAL, width=2),
                        marker=dict(color=AMBER, size=6, line=dict(color=PLOT_BG, width=1)),
                        name="Implied Volatility"
                    ))
                    fig_smile.add_vline(x=S0, line=dict(color=TEAL, width=1.5, dash="dash"),
                        annotation_text=f"  Current ${S0:.2f}", annotation_font=dict(color=TEAL, size=10))
                    fig_smile.update_layout(**plot_layout(
                        height=280, xaxis_title="Strike Price",
                        yaxis_title="Implied Volatility (%)",
                        title=dict(text="Real IV across strikes — the 'smile' shows fear is priced differently at different levels",
                                   font=dict(size=11, color=TEXT_DIM))
                    ))
                    st.plotly_chart(fig_smile, use_container_width=True)

                    if beginner_mode:
                        st.markdown("""
                        <div class="info-box">
                            <strong>The Volatility Smile/Skew</strong> is one of the most important real-world patterns in options markets.
                            In theory (Black-Scholes), IV should be flat across strikes. In reality, it isn't.
                            Out-of-the-money puts typically have higher IV (fear of crashes). This "skew" tells you
                            where the market is most worried. Strikes with unusually high IV are expensive;
                            those with low IV are cheap relative to market consensus.
                        </div>
                        """, unsafe_allow_html=True)
        else:
            st.info("Options chain data not available for this ticker. Try AAPL, TSLA, MSFT, or NVDA.")


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; padding:1rem 0;">
    <div style="font-family:'Source Code Pro',monospace; font-size:9px; letter-spacing:0.2em; color:#2E3F52; text-transform:uppercase;">
        Options Decoded · Real-time data via Yahoo Finance · For educational purposes only · Not financial advice
    </div>
</div>
""", unsafe_allow_html=True)



# import streamlit as st
# import numpy as np
# import plotly.graph_objects as go
# import plotly.express as px
# from plotly.subplots import make_subplots
# import pandas as pd
# from utils import (
#     simulate_gbm_paths,
#     monte_carlo_option_price,
#     black_scholes_price,
#     compute_greeks,
#     convergence_data,
#     asian_option_mc,
#     barrier_option_mc,
#     implied_volatility,
# )

# # ── Page Config ────────────────────────────────────────────────────────────
# st.set_page_config(
#     page_title="Options Pricing Lab",
#     page_icon="◈",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )

# # ── Custom CSS — Terminal Amber Aesthetic ─────────────────────────────────
# st.markdown("""
# <style>
# @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=Instrument+Serif:ital@0;1&display=swap');

# :root {
#     --amber: #E8A838;
#     --amber-dim: #B07A1A;
#     --amber-glow: #F5C842;
#     --bg: #0A0A08;
#     --surface: #111110;
#     --surface2: #1A1A16;
#     --border: #2A2A22;
#     --text: #D4C4A0;
#     --text-dim: #7A6E58;
#     --green: #5E9E6E;
#     --red: #C05050;
# }

# html, body, [class*="css"] {
#     font-family: 'IBM Plex Mono', monospace;
#     background-color: var(--bg);
#     color: var(--text);
# }

# /* Sidebar */
# [data-testid="stSidebar"] {
#     background-color: var(--surface);
#     border-right: 1px solid var(--border);
# }
# [data-testid="stSidebar"] * {
#     font-family: 'IBM Plex Mono', monospace !important;
#     color: var(--text) !important;
# }

# /* Headers */
# h1, h2, h3 {
#     font-family: 'Instrument Serif', serif !important;
#     color: var(--amber) !important;
#     letter-spacing: -0.02em;
# }

# /* Metric cards */
# [data-testid="metric-container"] {
#     background: var(--surface2);
#     border: 1px solid var(--border);
#     border-left: 3px solid var(--amber-dim);
#     padding: 16px 20px;
#     border-radius: 2px;
# }
# [data-testid="metric-container"] label {
#     color: var(--text-dim) !important;
#     font-size: 10px !important;
#     letter-spacing: 0.15em !important;
#     text-transform: uppercase;
# }
# [data-testid="metric-container"] [data-testid="stMetricValue"] {
#     font-family: 'IBM Plex Mono', monospace !important;
#     color: var(--amber-glow) !important;
#     font-size: 22px !important;
#     font-weight: 500 !important;
# }

# /* Sliders */
# [data-testid="stSlider"] > div > div > div {
#     background: var(--amber-dim) !important;
# }
# [data-testid="stSlider"] [role="slider"] {
#     background: var(--amber) !important;
#     border: 2px solid var(--amber-glow) !important;
# }

# /* Buttons */
# .stButton > button {
#     background: transparent;
#     border: 1px solid var(--amber-dim);
#     color: var(--amber);
#     font-family: 'IBM Plex Mono', monospace;
#     font-size: 11px;
#     letter-spacing: 0.12em;
#     text-transform: uppercase;
#     padding: 8px 20px;
#     border-radius: 1px;
#     transition: all 0.2s;
# }
# .stButton > button:hover {
#     background: var(--amber-dim);
#     color: var(--bg);
#     border-color: var(--amber);
# }

# /* Select boxes */
# .stSelectbox > div > div {
#     background: var(--surface2) !important;
#     border-color: var(--border) !important;
#     color: var(--text) !important;
#     font-family: 'IBM Plex Mono', monospace !important;
#     border-radius: 2px !important;
# }

# /* Tabs */
# [data-testid="stTab"] {
#     font-family: 'IBM Plex Mono', monospace !important;
#     font-size: 11px !important;
#     letter-spacing: 0.1em !important;
#     text-transform: uppercase !important;
# }

# /* Dividers */
# hr {
#     border-color: var(--border) !important;
# }

# /* Scrollbar */
# ::-webkit-scrollbar { width: 4px; }
# ::-webkit-scrollbar-track { background: var(--bg); }
# ::-webkit-scrollbar-thumb { background: var(--border); }

# /* Remove Streamlit default padding artifacts */
# .block-container { padding-top: 2rem !important; }

# /* Section labels */
# .section-label {
#     font-size: 10px;
#     letter-spacing: 0.2em;
#     text-transform: uppercase;
#     color: var(--text-dim);
#     border-bottom: 1px solid var(--border);
#     padding-bottom: 6px;
#     margin-bottom: 16px;
# }

# /* Ticker-style header */
# .ticker-header {
#     font-family: 'IBM Plex Mono', monospace;
#     font-size: 10px;
#     letter-spacing: 0.15em;
#     color: var(--text-dim);
#     text-transform: uppercase;
# }

# .amber { color: var(--amber); }
# .green { color: var(--green); }
# .red { color: var(--red); }
# </style>
# """, unsafe_allow_html=True)

# # ── Plotly dark theme matching our aesthetic ───────────────────────────────
# PLOT_TEMPLATE = dict(
#     layout=dict(
#         paper_bgcolor="#0A0A08",
#         plot_bgcolor="#111110",
#         font=dict(family="IBM Plex Mono", color="#D4C4A0", size=11),
#         xaxis=dict(gridcolor="#1A1A16", linecolor="#2A2A22", tickcolor="#7A6E58"),
#         yaxis=dict(gridcolor="#1A1A16", linecolor="#2A2A22", tickcolor="#7A6E58"),
#         colorway=["#E8A838", "#5E9E6E", "#C05050", "#6E8EC0", "#9E6E9E"],
#     )
# )

# AMBER = "#E8A838"
# AMBER_DIM = "#B07A1A"
# GREEN = "#5E9E6E"
# RED = "#C05050"
# BLUE = "#6E8EC0"
# DIM = "#7A6E58"


# # ── Header ─────────────────────────────────────────────────────────────────
# st.markdown('<p class="ticker-header">◈ Quantitative Finance Lab — Options Pricing Engine v2</p>', unsafe_allow_html=True)
# st.markdown("# Options Pricing Lab")
# st.markdown('<p style="color:#7A6E58; font-size:12px; margin-top:-12px;">Monte Carlo · Black-Scholes · Greeks · Exotics · Implied Volatility</p>', unsafe_allow_html=True)
# st.markdown("---")


# # ── Sidebar Controls ───────────────────────────────────────────────────────
# with st.sidebar:
#     st.markdown("## Parameters")
#     st.markdown('<div class="section-label">Underlying</div>', unsafe_allow_html=True)
#     S0 = st.slider("Spot Price  S₀", 50.0, 300.0, 100.0, 1.0)
#     K = st.slider("Strike Price  K", 50.0, 300.0, 100.0, 1.0)

#     st.markdown('<div class="section-label">Market</div>', unsafe_allow_html=True)
#     r = st.slider("Risk-free Rate  r", 0.00, 0.15, 0.05, 0.001, format="%.3f")
#     sigma = st.slider("Volatility  σ", 0.05, 1.00, 0.20, 0.01, format="%.2f")
#     T = st.slider("Time to Maturity  T (years)", 0.1, 3.0, 1.0, 0.05)

#     st.markdown('<div class="section-label">Simulation</div>', unsafe_allow_html=True)
#     n_paths = st.select_slider("MC Paths", [1000, 5000, 10000, 25000, 50000, 100000], 25000)
#     n_display = st.slider("Paths to Display", 10, 200, 50, 10)
#     option_type = st.selectbox("Option Type", ["call", "put"])

#     st.markdown('<div class="section-label">Exotic — Barrier</div>', unsafe_allow_html=True)
#     barrier = st.slider("Barrier Level  B", 20.0, float(S0) * 0.98, min(80.0, float(S0) * 0.8), 1.0)

#     st.markdown("---")
#     run = st.button("▶  RUN SIMULATION", use_container_width=True)


# # ── Run on load or button press ────────────────────────────────────────────
# if "results" not in st.session_state or run:
#     with st.spinner("Simulating paths..."):
#         t_axis, paths = simulate_gbm_paths(S0, r, sigma, T, 252, min(n_paths, 500))
#         mc_price, mc_se = monte_carlo_option_price(S0, r, sigma, T, K, n_paths, option_type)
#         bs_price = black_scholes_price(S0, K, r, sigma, T, option_type)
#         greeks = compute_greeks(S0, K, r, sigma, T, option_type)
#         conv_n, conv_mc, conv_err, conv_bs = convergence_data(S0, K, r, sigma, T, option_type)
#         asian = asian_option_mc(S0, K, r, sigma, T, 252, 20000, option_type)
#         barrier_price = barrier_option_mc(S0, K, barrier, r, sigma, T, 252, 20000)
#         st.session_state.results = dict(
#             t_axis=t_axis, paths=paths, mc_price=mc_price, mc_se=mc_se,
#             bs_price=bs_price, greeks=greeks,
#             conv_n=conv_n, conv_mc=conv_mc, conv_err=conv_err, conv_bs=conv_bs,
#             asian=asian, barrier_price=barrier_price,
#         )

# res = st.session_state.results

# # ── Main Tabs ──────────────────────────────────────────────────────────────
# tab1, tab2, tab3, tab4 = st.tabs(["  Pricing  ", "  Greeks  ", "  Paths  ", "  Exotics & IV  "])


# # ══════════════════════════════════════════════════════════════════════════
# # TAB 1 — Pricing
# # ══════════════════════════════════════════════════════════════════════════
# with tab1:
#     mc = res["mc_price"]
#     bs = res["bs_price"]
#     se = res["mc_se"]
#     diff = mc - bs
#     diff_pct = (diff / bs * 100) if bs != 0 else 0

#     col1, col2, col3, col4 = st.columns(4)
#     col1.metric("MC Price", f"${mc:.4f}", f"±{se:.4f} SE")
#     col2.metric("BS Price", f"${bs:.4f}")
#     col3.metric("Difference", f"${diff:+.4f}", f"{diff_pct:+.2f}%")
#     col4.metric("Moneyness", f"{S0/K:.3f}", "ITM" if S0 > K else ("ATM" if S0 == K else "OTM"))

#     st.markdown("---")

#     # Convergence plot
#     conv_n = res["conv_n"]
#     conv_mc = res["conv_mc"]
#     conv_err = res["conv_err"]
#     conv_bs = res["conv_bs"]

#     fig = make_subplots(
#         rows=1, cols=2,
#         subplot_titles=["MC Price Convergence to Black-Scholes", "Absolute Error vs Path Count"],
#         horizontal_spacing=0.08,
#     )

#     fig.add_trace(go.Scatter(
#         x=conv_n, y=conv_mc, mode="lines", name="MC Price",
#         line=dict(color=AMBER, width=1.5),
#     ), row=1, col=1)
#     fig.add_trace(go.Scatter(
#         x=conv_n, y=[conv_bs] * len(conv_n), mode="lines", name="BS Analytical",
#         line=dict(color=GREEN, width=1.5, dash="dash"),
#     ), row=1, col=1)

#     fig.add_trace(go.Scatter(
#         x=conv_n, y=conv_err, mode="lines", name="|MC − BS|",
#         line=dict(color=RED, width=1.5),
#         fill="tozeroy", fillcolor="rgba(192,80,80,0.08)",
#     ), row=1, col=2)

#     fig.update_layout(
#         **PLOT_TEMPLATE["layout"],
#         height=380,
#         showlegend=True,
#         legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#2A2A22", borderwidth=1),
#         margin=dict(t=40, b=30, l=10, r=10),
#     )
#     fig.update_xaxes(**PLOT_TEMPLATE["layout"]["xaxis"], title_text="Number of Paths")
#     fig.update_yaxes(**PLOT_TEMPLATE["layout"]["yaxis"])
#     fig.update_annotations(font=dict(family="IBM Plex Mono", color=DIM, size=11))
#     st.plotly_chart(fig, width='stretch')

#     # Payoff diagram
#     st.markdown("#### Payoff at Expiry")
#     S_range = np.linspace(max(1, S0 * 0.4), S0 * 1.8, 300)
#     if option_type == "call":
#         payoff_vals = np.maximum(S_range - K, 0)
#         bs_vals = [black_scholes_price(s, K, r, sigma, T, "call") for s in S_range]
#     else:
#         payoff_vals = np.maximum(K - S_range, 0)
#         bs_vals = [black_scholes_price(s, K, r, sigma, T, "put") for s in S_range]

#     fig2 = go.Figure()
#     fig2.add_trace(go.Scatter(x=S_range, y=payoff_vals, mode="lines", name="Intrinsic (Expiry)",
#         line=dict(color=DIM, width=1.5, dash="dot")))
#     fig2.add_trace(go.Scatter(x=S_range, y=bs_vals, mode="lines", name="BS Time Value",
#         line=dict(color=AMBER, width=2),
#         fill="tozeroy", fillcolor="rgba(232,168,56,0.07)"))
#     fig2.add_vline(x=S0, line=dict(color=GREEN, width=1, dash="dash"),
#         annotation_text=f"S₀={S0}", annotation_font=dict(color=GREEN, size=10))
#     fig2.add_vline(x=K, line=dict(color=RED, width=1, dash="dash"),
#         annotation_text=f"K={K}", annotation_font=dict(color=RED, size=10))
#     fig2.update_layout(**PLOT_TEMPLATE["layout"], height=280,
#         margin=dict(t=10, b=30, l=10, r=10),
#         legend=dict(bgcolor="rgba(0,0,0,0)"))
#     fig2.update_xaxes(**PLOT_TEMPLATE["layout"]["xaxis"], title_text="Stock Price at Expiry")
#     fig2.update_yaxes(**PLOT_TEMPLATE["layout"]["yaxis"], title_text="Option Value")
#     st.plotly_chart(fig2, width='stretch')


# # ══════════════════════════════════════════════════════════════════════════
# # TAB 2 — Greeks
# # ══════════════════════════════════════════════════════════════════════════
# with tab2:
#     g = res["greeks"]

#     col1, col2, col3, col4, col5 = st.columns(5)
#     col1.metric("Delta  Δ", f"{g['delta']:.4f}", help="Rate of change of option price w.r.t. spot")
#     col2.metric("Gamma  Γ", f"{g['gamma']:.4f}", help="Rate of change of Delta w.r.t. spot")
#     col3.metric("Vega  ν", f"{g['vega']:.4f}", help="Sensitivity to 1% change in volatility")
#     col4.metric("Theta  Θ", f"{g['theta']:.4f}", help="Daily time decay")
#     col5.metric("Rho  ρ", f"{g['rho']:.4f}", help="Sensitivity to 1% change in interest rate")

#     st.markdown("---")

#     # Greeks surface across S and sigma
#     st.markdown("#### Delta & Gamma Surface Across Spot Price")
#     S_vals = np.linspace(max(1, S0 * 0.4), S0 * 1.8, 120)
#     deltas = [compute_greeks(s, K, r, sigma, T, option_type)["delta"] for s in S_vals]
#     gammas = [compute_greeks(s, K, r, sigma, T, option_type)["gamma"] for s in S_vals]
#     vegas = [compute_greeks(s, K, r, sigma, T, option_type)["vega"] for s in S_vals]
#     thetas = [compute_greeks(s, K, r, sigma, T, option_type)["theta"] for s in S_vals]

#     fig3 = make_subplots(rows=2, cols=2,
#         subplot_titles=["Delta (Δ)", "Gamma (Γ)", "Vega (ν)", "Theta (Θ)"],
#         vertical_spacing=0.15, horizontal_spacing=0.08)

#     colors = [AMBER, GREEN, BLUE, RED]
#     data_sets = [deltas, gammas, vegas, thetas]
#     positions = [(1,1),(1,2),(2,1),(2,2)]

#     fill_colors = ["rgba(232,168,56,0.07)", "rgba(94,158,110,0.07)", "rgba(110,142,192,0.07)", "rgba(192,80,80,0.07)"]
#     for (row, col), color, fill_color, data, name in zip(positions, colors, fill_colors, data_sets, ["Δ","Γ","ν","Θ"]):
#         fig3.add_trace(go.Scatter(x=S_vals, y=data, mode="lines", name=name,
#             line=dict(color=color, width=2),
#             fill="tozeroy", fillcolor=fill_color), row=row, col=col)
#         fig3.add_vline(x=S0, row=row, col=col,
#             line=dict(color=DIM, width=1, dash="dot"))

#     fig3.update_layout(**PLOT_TEMPLATE["layout"], height=520,
#         showlegend=False, margin=dict(t=40, b=20, l=10, r=10))
#     for ax in ["xaxis", "xaxis2", "xaxis3", "xaxis4"]:
#         fig3.update_layout({ax: PLOT_TEMPLATE["layout"]["xaxis"]})
#     for ax in ["yaxis", "yaxis2", "yaxis3", "yaxis4"]:
#         fig3.update_layout({ax: PLOT_TEMPLATE["layout"]["yaxis"]})
#     fig3.update_annotations(font=dict(family="IBM Plex Mono", color=DIM, size=11))
#     st.plotly_chart(fig3, width='stretch')


# # ══════════════════════════════════════════════════════════════════════════
# # TAB 3 — Price Paths
# # ══════════════════════════════════════════════════════════════════════════
# with tab3:
#     paths = res["paths"]
#     t_axis = res["t_axis"]
#     n_show = min(n_display, paths.shape[0])

#     S_T = paths[:, -1]
#     mc_price_shown = res["mc_price"]

#     fig4 = go.Figure()

#     # Background paths (dim)
#     for i in range(n_show):
#         end_val = paths[i, -1]
#         color = GREEN if end_val > K else RED
#         fig4.add_trace(go.Scatter(
#             x=t_axis, y=paths[i],
#             mode="lines", line=dict(color=color, width=0.6),
#             opacity=0.25, showlegend=False,
#         ))

#     # Mean path
#     mean_path = np.mean(paths, axis=0)
#     fig4.add_trace(go.Scatter(x=t_axis, y=mean_path, mode="lines",
#         line=dict(color=AMBER, width=2.5), name="Mean Path"))

#     # Strike line
#     fig4.add_hline(y=K, line=dict(color=DIM, width=1, dash="dash"),
#         annotation_text=f"Strike K={K}", annotation_font=dict(color=DIM, size=10))

#     fig4.update_layout(
#         **PLOT_TEMPLATE["layout"], height=480,
#         title=dict(text=f"GBM Simulated Paths  (n={n_show} shown)", font=dict(color=DIM, size=12)),
#         xaxis_title="Time (years)", yaxis_title="Stock Price",
#         legend=dict(bgcolor="rgba(0,0,0,0)"),
#         margin=dict(t=50, b=40, l=10, r=10),
#     )
#     st.plotly_chart(fig4, width='stretch')

#     # Terminal distribution
#     st.markdown("#### Terminal Price Distribution")
#     fig5 = go.Figure()
#     fig5.add_trace(go.Histogram(x=S_T, nbinsx=80, name="S_T",
#         marker=dict(color=AMBER, line=dict(color=AMBER_DIM, width=0.3)),
#         opacity=0.75))
#     fig5.add_vline(x=S0, line=dict(color=GREEN, dash="dash"),
#         annotation_text="S₀", annotation_font=dict(color=GREEN, size=10))
#     fig5.add_vline(x=K, line=dict(color=RED, dash="dash"),
#         annotation_text="K", annotation_font=dict(color=RED, size=10))
#     fig5.add_vline(x=np.mean(S_T), line=dict(color=AMBER, dash="dot"),
#         annotation_text=f"E[S_T]={np.mean(S_T):.1f}", annotation_font=dict(color=AMBER, size=10))
#     fig5.update_layout(**PLOT_TEMPLATE["layout"], height=280,
#         xaxis_title="Stock Price at Maturity", yaxis_title="Count",
#         margin=dict(t=10, b=30, l=10, r=10), showlegend=False)
#     st.plotly_chart(fig5, width='stretch')


# # ══════════════════════════════════════════════════════════════════════════
# # TAB 4 — Exotics & IV
# # ══════════════════════════════════════════════════════════════════════════
# with tab4:
#     col_e, col_iv = st.columns([1, 1])

#     with col_e:
#         st.markdown("#### Exotic Options")
#         st.markdown('<div class="section-label">Pricing Summary</div>', unsafe_allow_html=True)

#         vanilla_bs = res["bs_price"]
#         asian = res["asian"]
#         bar = res["barrier_price"]

#         rows = [
#             {"Type": "European (BS)", "Price": f"${vanilla_bs:.4f}", "Notes": "Analytical"},
#             {"Type": f"European (MC)", "Price": f"${res['mc_price']:.4f}", "Notes": "Monte Carlo"},
#             {"Type": "Asian (Arith.)", "Price": f"${asian:.4f}", "Notes": "Average payoff"},
#             {"Type": f"Barrier D&O (B={barrier:.0f})", "Price": f"${bar:.4f}", "Notes": "Knock-out below B"},
#         ]
#         df = pd.DataFrame(rows)
#         st.dataframe(df, width='stretch', hide_index=True)

#         # Bar chart comparison
#         fig6 = go.Figure(go.Bar(
#             x=[r["Type"] for r in rows],
#             y=[float(r["Price"].replace("$", "")) for r in rows],
#             marker=dict(
#                 color=[AMBER, AMBER_DIM, GREEN, BLUE],
#                 line=dict(color="#0A0A08", width=1),
#             ),
#         ))
#         fig6.update_layout(**PLOT_TEMPLATE["layout"], height=280,
#             yaxis_title="Option Price", margin=dict(t=10, b=60, l=10, r=10),
#             showlegend=False)
#         st.plotly_chart(fig6, width='stretch')

#     with col_iv:
#         st.markdown("#### Implied Volatility Solver")
#         st.markdown('<div class="section-label">Back-solve σ from Market Price</div>', unsafe_allow_html=True)
#         market_price = st.number_input(
#             "Market Option Price ($)",
#             min_value=0.01,
#             max_value=float(S0),
#             value=float(round(res["bs_price"], 2)),
#             step=0.01,
#         )
#         iv = implied_volatility(market_price, S0, K, r, T, option_type)
#         if iv is not None:
#             st.metric("Implied Volatility", f"{iv*100:.2f}%",
#                 delta=f"{(iv - sigma)*100:+.2f}% vs input σ")

#             # IV smile simulation
#             st.markdown("#### Volatility Smile (Simulated)")
#             strikes = np.linspace(K * 0.7, K * 1.3, 25)
#             # Simulate a gentle smile around ATM
#             atm_iv = iv
#             smile_ivs = atm_iv + 0.08 * ((strikes - K) / K) ** 2 + 0.02 * abs((strikes - K) / K)
#             fig7 = go.Figure()
#             fig7.add_trace(go.Scatter(x=strikes, y=smile_ivs * 100, mode="lines+markers",
#                 line=dict(color=AMBER, width=2),
#                 marker=dict(color=AMBER_DIM, size=5),
#                 name="IV Smile"))
#             fig7.add_vline(x=K, line=dict(color=DIM, dash="dash"),
#                 annotation_text="ATM", annotation_font=dict(color=DIM, size=9))
#             fig7.update_layout(**PLOT_TEMPLATE["layout"], height=240,
#                 xaxis_title="Strike", yaxis_title="Implied Vol (%)",
#                 margin=dict(t=10, b=30, l=10, r=10), showlegend=False)
#             st.plotly_chart(fig7, width='stretch')
#         else:
#             st.error("No valid IV found for this price. Check inputs.")

# # ── Footer ─────────────────────────────────────────────────────────────────
# st.markdown("---")
# st.markdown(
#     '<p style="font-size:10px; color:#3A3A32; text-align:center; letter-spacing:0.12em;">'
#     'OPTIONS PRICING LAB · MONTE CARLO ENGINE · FOR EDUCATIONAL USE ONLY'
#     '</p>',
#     unsafe_allow_html=True,
# )