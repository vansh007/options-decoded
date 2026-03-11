import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime

from utils import (
    get_stock_info, get_historical_volatility, get_options_chain,
    fetch_option_chain_for_expiry, black_scholes_price, simulate_gbm_paths,
    monte_carlo_price, compute_greeks, implied_volatility,
    generate_signal, greek_explanation,
    simulate_discrete_hedge, compare_hedge_frequencies,
)

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Options Intelligence",
    page_icon="◬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Design System ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Source+Code+Pro:wght@300;400;500&family=Lora:ital,wght@0,400;0,500;1,400&display=swap');

:root {
    --ink:       #0F1923;
    --slate:     #1C2B3A;
    --slate2:    #243447;
    --teal:      #00C9A7;
    --teal-dim:  #007A65;
    --teal-glow: #3DFFD9;
    --amber:     #FFB347;
    --red:       #FF6B6B;
    --muted:     #8A9BB0;
    --border:    #2E3F52;
    --text:      #F0EDE8;
    --text-dim:  #7A8EA0;
}
*, html, body { box-sizing: border-box; }
body, [class*="css"], .stApp {
    font-family: 'Source Code Pro', monospace;
    background: var(--ink); color: var(--text);
}
.block-container { padding: 2rem 3rem !important; max-width: 1400px !important; }

.hero { text-align:center; padding:3rem 0 2rem; border-bottom:1px solid var(--border); margin-bottom:2.5rem; }
.hero-eyebrow { font-size:11px; letter-spacing:0.25em; text-transform:uppercase; color:var(--teal); margin-bottom:0.75rem; }
.hero-title { font-family:'Syne',sans-serif; font-size:clamp(2.2rem,5vw,3.8rem); font-weight:800; color:var(--text); line-height:1.05; letter-spacing:-0.03em; margin:0 0 0.5rem; }
.hero-title span { color:var(--teal); }
.hero-sub { font-family:'Lora',serif; font-size:1rem; color:var(--text-dim); font-style:italic; margin:0; }

.stTextInput > div > div > input {
    background:var(--slate) !important; border:1.5px solid var(--border) !important;
    border-radius:4px !important; color:var(--text) !important;
    font-family:'Syne',sans-serif !important; font-size:1.3rem !important;
    font-weight:700 !important; padding:0.75rem 1.2rem !important;
    letter-spacing:0.05em !important; transition:border-color 0.2s !important;
}
.stTextInput > div > div > input:focus {
    border-color:var(--teal) !important;
    box-shadow:0 0 0 3px rgba(0,201,167,0.12) !important;
}
.stTextInput > label {
    font-family:'Source Code Pro',monospace !important; font-size:10px !important;
    letter-spacing:0.2em !important; text-transform:uppercase !important;
    color:var(--text-dim) !important;
}

.stock-bar {
    background:var(--slate); border:1px solid var(--border); border-radius:6px;
    padding:1.2rem 1.8rem; margin-bottom:2rem;
    display:flex; align-items:center; gap:2.5rem; flex-wrap:wrap;
}
.stock-name { font-family:'Syne',sans-serif; font-size:1.4rem; font-weight:700; color:var(--text); }
.stock-price { font-family:'Syne',sans-serif; font-size:2rem; font-weight:800; color:var(--teal); }
.stock-meta { font-size:10px; color:var(--text-dim); letter-spacing:0.1em; text-transform:uppercase; }
.stock-meta strong { color:var(--text); font-size:12px; }

.section-head { font-family:'Syne',sans-serif; font-size:1.1rem; font-weight:700; color:var(--text); letter-spacing:-0.01em; margin:0 0 0.3rem; }
.section-label { font-family:'Source Code Pro',monospace; font-size:9px; letter-spacing:0.25em; text-transform:uppercase; color:var(--teal-dim); margin-bottom:1rem; padding-bottom:0.4rem; border-bottom:1px solid var(--border); }

.signal-card { background:var(--slate); border:1px solid var(--border); border-radius:6px; padding:1.2rem 1.5rem; margin-bottom:1rem; border-left:3px solid var(--teal-dim); }
.signal-card.warning { border-left-color:var(--amber); }
.signal-card.good    { border-left-color:var(--teal); }
.signal-card.danger  { border-left-color:var(--red); }
.signal-title  { font-family:'Syne',sans-serif; font-size:0.9rem; font-weight:700; color:var(--text); margin-bottom:0.4rem; }
.signal-plain  { font-size:12px; color:var(--text-dim); line-height:1.6; margin-bottom:0.6rem; }
.signal-tip    { font-size:11px; color:var(--teal); background:rgba(0,201,167,0.06); border-radius:3px; padding:0.5rem 0.8rem; line-height:1.5; }

[data-testid="metric-container"] { background:var(--slate2) !important; border:1px solid var(--border) !important; border-radius:4px !important; padding:1rem 1.2rem !important; }
[data-testid="metric-container"] label { font-family:'Source Code Pro',monospace !important; font-size:9px !important; letter-spacing:0.18em !important; text-transform:uppercase !important; color:var(--text-dim) !important; }
[data-testid="stMetricValue"]  { font-family:'Syne',sans-serif !important; font-size:1.5rem !important; font-weight:700 !important; color:var(--teal-glow) !important; }
[data-testid="stMetricDelta"]  { font-family:'Source Code Pro',monospace !important; font-size:11px !important; }

.greek-card     { background:var(--slate); border:1px solid var(--border); border-radius:6px; padding:1.2rem; margin-bottom:1rem; }
.greek-name     { font-family:'Syne',sans-serif; font-size:1rem; font-weight:700; color:var(--teal); margin-bottom:0.2rem; }
.greek-value    { font-family:'Source Code Pro',monospace; font-size:1.4rem; color:var(--text); font-weight:500; margin-bottom:0.6rem; }
.greek-what     { font-size:12px; color:var(--text); line-height:1.5; margin-bottom:0.4rem; }
.greek-analogy  { font-size:11px; color:var(--text-dim); line-height:1.5; font-style:italic; margin-bottom:0.4rem; }
.greek-watch    { font-size:10px; color:var(--amber); background:rgba(255,179,71,0.07); border-radius:3px; padding:0.4rem 0.6rem; line-height:1.5; }

.verdict { display:inline-block; font-family:'Syne',sans-serif; font-size:1rem; font-weight:800; letter-spacing:0.08em; padding:0.3rem 1rem; border-radius:3px; text-transform:uppercase; }
.verdict-expensive { background:rgba(255,107,107,0.15); color:var(--red);  border:1px solid var(--red); }
.verdict-cheap     { background:rgba(0,201,167,0.12);   color:var(--teal); border:1px solid var(--teal-dim); }
.verdict-neutral   { background:rgba(138,155,176,0.15); color:var(--muted);border:1px solid var(--border); }

/* Discrete hedging cards */
.hedge-card { background:var(--slate); border:1px solid var(--border); border-radius:6px; padding:1.2rem 1.4rem; }
.hedge-freq { font-family:'Syne',sans-serif; font-size:1rem; font-weight:800; letter-spacing:0.06em; text-transform:uppercase; margin-bottom:0.3rem; }
.hedge-stat-row { display:flex; gap:1.5rem; flex-wrap:wrap; margin-top:0.6rem; }
.hedge-stat { text-align:left; }
.hedge-stat-val { font-family:'Syne',sans-serif; font-size:1.1rem; font-weight:700; }
.hedge-stat-label { font-size:9px; letter-spacing:0.15em; text-transform:uppercase; color:var(--text-dim); margin-top:1px; }

[data-testid="stTab"] button { font-family:'Source Code Pro',monospace !important; font-size:10px !important; letter-spacing:0.15em !important; text-transform:uppercase !important; color:var(--text-dim) !important; }
[data-testid="stTab"] button[aria-selected="true"] { color:var(--teal) !important; border-bottom-color:var(--teal) !important; }

.stSelectbox > div > div { background:var(--slate2) !important; border-color:var(--border) !important; border-radius:4px !important; font-family:'Source Code Pro',monospace !important; font-size:12px !important; }
.stSlider [data-baseweb="slider"] div[role="slider"] { background:var(--teal) !important; }

.info-box { background:rgba(0,201,167,0.05); border:1px solid rgba(0,201,167,0.2); border-radius:6px; padding:1rem 1.2rem; margin:1rem 0; font-size:12px; color:var(--text-dim); line-height:1.6; }
.info-box strong { color:var(--teal); }
.warning-box { background:rgba(255,179,71,0.06); border:1px solid rgba(255,179,71,0.25); border-radius:6px; padding:1rem 1.2rem; margin:1rem 0; font-size:12px; color:var(--text-dim); line-height:1.6; }
.warning-box strong { color:var(--amber); }

hr { border-color:var(--border) !important; margin:1.5rem 0 !important; }
::-webkit-scrollbar { width:4px; height:4px; }
::-webkit-scrollbar-track { background:var(--ink); }
::-webkit-scrollbar-thumb { background:var(--border); border-radius:2px; }
.stButton > button { background:var(--teal) !important; color:var(--ink) !important; border:none !important; font-family:'Syne',sans-serif !important; font-weight:700 !important; font-size:13px !important; letter-spacing:0.05em !important; padding:0.6rem 2rem !important; border-radius:3px !important; transition:all 0.2s !important; }
.stButton > button:hover { background:var(--teal-glow) !important; box-shadow:0 4px 20px rgba(0,201,167,0.3) !important; }
.mode-banner { background:linear-gradient(135deg,rgba(0,201,167,0.08),rgba(0,201,167,0.02)); border:1px solid rgba(0,201,167,0.2); border-radius:6px; padding:0.7rem 1.2rem; margin-bottom:1.5rem; font-size:11px; color:var(--teal); letter-spacing:0.05em; }
#MainMenu, header, footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

# ── Plotly palette ────────────────────────────────────────────────────────────
PLOT_BG   = "#0F1923"
PLOT_SURF = "#1C2B3A"
TEAL      = "#00C9A7"
TEAL_DIM  = "#007A65"
AMBER     = "#FFB347"
RED       = "#FF6B6B"
BLUE      = "#6E9EFF"
MUTED     = "#3D5470"
TEXT_DIM  = "#7A8EA0"
BORDER    = "#2E3F52"

def pl(**kw):
    base = dict(
        paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_SURF,
        font=dict(family="Source Code Pro, monospace", color=TEXT_DIM, size=10),
        xaxis=dict(gridcolor=MUTED, gridwidth=0.5, linecolor=BORDER, tickcolor=TEXT_DIM),
        yaxis=dict(gridcolor=MUTED, gridwidth=0.5, linecolor=BORDER, tickcolor=TEXT_DIM),
        margin=dict(t=35, b=40, l=10, r=10),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=BORDER, borderwidth=1, font=dict(size=10)),
    )
    base.update(kw)
    return base

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-eyebrow">◬ Real-Time Options Intelligence</div>
  <h1 class="hero-title">Options<span> Decoded</span></h1>
  <p class="hero-sub">Understand exactly what the market is pricing — and what it means for you.</p>
</div>
""", unsafe_allow_html=True)

col_mode, _ = st.columns([1, 3])
with col_mode:
    beginner_mode = st.toggle("📖 Plain-English Mode", value=True)
if beginner_mode:
    st.markdown('<div class="mode-banner">📖 Plain-English Mode is ON — every number is explained in simple terms.</div>', unsafe_allow_html=True)

# ── Ticker search ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Enter Any Stock Ticker</div>', unsafe_allow_html=True)
col_search, col_btn, col_ex = st.columns([3, 1, 3])
with col_search:
    ticker_input = st.text_input(
        "Stock Ticker",          # non-empty label — fixes the warning
        value="AAPL",
        placeholder="e.g. AAPL, TSLA, MSFT, GOOGL",
        label_visibility="collapsed",
    )
with col_btn:
    st.button("ANALYSE →", use_container_width=True)
with col_ex:
    st.markdown('<div style="padding:0.6rem 0;font-size:11px;color:#3D5470;">Try: AAPL · TSLA · MSFT · NVDA · SPY · META</div>', unsafe_allow_html=True)

ticker = ticker_input.strip().upper()

# ── Data fetch ────────────────────────────────────────────────────────────────
if ticker:
    with st.spinner(f"Fetching live data for {ticker}…"):
        stock_info        = get_stock_info(ticker)
        hv_value, hv_hist = get_historical_volatility(ticker)
        ticker_obj, expirations = get_options_chain(ticker)

    if not stock_info or not stock_info.get("price"):
        st.error(f"Could not find data for **{ticker}**. Check the ticker symbol and try again.")
        st.stop()

    S0 = stock_info["price"]

    # ── Stock bar ─────────────────────────────────────────────────────────────
    price_52_pct = ""
    if stock_info.get("week_52_high") and stock_info.get("week_52_low"):
        lo, hi = stock_info["week_52_low"], stock_info["week_52_high"]
        pos = (S0 - lo) / (hi - lo) * 100 if hi > lo else 50
        price_52_pct = f"<div class='stock-meta'>52W RANGE<br><strong>{pos:.0f}% of range</strong></div>"
    mc_str = ""
    if stock_info.get("market_cap"):
        mc = stock_info["market_cap"]
        mc_str = f"${mc/1e9:.1f}B" if mc >= 1e9 else f"${mc/1e6:.0f}M"

    st.markdown(f"""
    <div class="stock-bar">
        <div>
            <div class="stock-name">{stock_info['name']}</div>
            <div class="stock-meta">NYSE/NASDAQ · {stock_info.get('sector','—')}</div>
        </div>
        <div class="stock-price">${S0:,.2f}</div>
        <div class="stock-meta">HV (30d)<br><strong>{f"{hv_value*100:.1f}%" if hv_value else "—"}</strong></div>
        <div class="stock-meta">MKT CAP<br><strong>{mc_str or "—"}</strong></div>
        {price_52_pct}
        <div class="stock-meta">UPDATED<br><strong>{datetime.now().strftime("%H:%M:%S")}</strong></div>
    </div>
    """, unsafe_allow_html=True)

    if beginner_mode:
        st.markdown(f"""
        <div class="info-box">
            <strong>What am I looking at?</strong> Live price of {stock_info['name']}.
            <strong>HV (30d)</strong> = Historical Volatility — how much the stock has actually moved over 30 days,
            annualised. We compare this to what options <em>imply</em> will happen — that gap is where real insight lives.
        </div>""", unsafe_allow_html=True)

    # ── Option configuration ──────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-label">Configure Your Option</div>', unsafe_allow_html=True)

    if expirations:
        exp_labels = [f"{e}  ({(datetime.strptime(e,'%Y-%m-%d')-datetime.today()).days}d)"
                      for e in expirations]
        c1, c2, c3, c4 = st.columns([2, 1.5, 1.5, 1.5])
        with c1:
            exp_idx       = st.selectbox("Expiry Date", range(len(expirations)),
                                          format_func=lambda i: exp_labels[i], key="expiry")
            chosen_expiry = expirations[exp_idx]
            T_days        = (datetime.strptime(chosen_expiry, "%Y-%m-%d") - datetime.today()).days
            T             = max(T_days / 365, 1 / 365)
        with c2:
            option_type = st.selectbox("Option Type", ["call", "put"],
                format_func=lambda x: "📈 Call (bet UP)" if x == "call" else "📉 Put (bet DOWN)")
        with c3:
            r = st.slider("Risk-free Rate", 0.0, 0.10, 0.05, 0.005, format="%.3f")
        with c4:
            n_sim = st.select_slider("MC Paths", [5000, 10000, 25000, 50000], 25000)
    else:
        st.warning("No options chain found. Using manual parameters.")
        c1, c2, c3 = st.columns(3)
        with c1:
            T_days = st.slider("Days to Expiry", 7, 365, 45)
            T = T_days / 365
        with c2:
            option_type = st.selectbox("Option Type", ["call", "put"])
        with c3:
            r = st.slider("Risk-free Rate", 0.0, 0.10, 0.05, 0.005, format="%.3f")
        n_sim         = 25000
        chosen_expiry = "custom"

    if beginner_mode:
        st.markdown(
            f'<div class="info-box">📘 A <strong>{"Call" if option_type=="call" else "Put"}</strong> '
            f'gives you the right to {"BUY" if option_type=="call" else "SELL"} the stock at the '
            f'strike price. You profit if the stock goes {"UP" if option_type=="call" else "DOWN"}.</div>',
            unsafe_allow_html=True)

    # ── Options chain + strike ────────────────────────────────────────────────
    calls_df = puts_df = None
    if ticker_obj and chosen_expiry != "custom":
        with st.spinner("Loading options chain…"):
            calls_df, puts_df = fetch_option_chain_for_expiry(ticker_obj, chosen_expiry, S0)
    chain_df = calls_df if option_type == "call" else puts_df

    st.markdown('<div class="section-label" style="margin-top:1rem">Select Strike Price</div>', unsafe_allow_html=True)
    if chain_df is not None and not chain_df.empty:
        strikes   = sorted(chain_df['strike'].tolist())
        atm_idx   = min(range(len(strikes)), key=lambda i: abs(strikes[i] - S0))
        sc1, sc2  = st.columns([2, 3])
        with sc1:
            K = st.select_slider("Strike Price", strikes, value=strikes[atm_idx],
                                  format_func=lambda x: f"${x:.2f}")
        with sc2:
            moneyness = (S0 - K) / K * 100 if option_type == "call" else (K - S0) / K * 100
            status    = ("IN-THE-MONEY 💰" if moneyness > 1
                         else "AT-THE-MONEY ⚖️" if abs(moneyness) <= 1
                         else "OUT-OF-THE-MONEY 📭")
            sc        = TEAL if moneyness > 1 else (AMBER if abs(moneyness) <= 1 else RED)
            st.markdown(f"""
            <div style="padding:0.8rem 1rem;background:var(--slate2);border-radius:4px;border:1px solid var(--border);margin-top:1.5rem">
                <div style="font-size:9px;letter-spacing:0.15em;text-transform:uppercase;color:{sc}">{status}</div>
                <div style="font-family:'Syne',sans-serif;font-size:1rem;font-weight:700;color:var(--text);margin-top:0.2rem">
                    Stock is {abs(moneyness):.1f}% {'above' if moneyness>0 else 'below'} strike
                </div>
            </div>""", unsafe_allow_html=True)
        row          = chain_df[chain_df['strike'] == K]
        market_price = float(row['lastPrice'].iloc[0]) if not row.empty else None
        market_iv    = float(row['impliedVolatility'].iloc[0]) if not row.empty and 'impliedVolatility' in row.columns else None
    else:
        K            = st.slider("Strike Price", float(S0*0.7), float(S0*1.3), float(S0), float(S0*0.01))
        market_price = market_iv = None

    # ── Core pricing ──────────────────────────────────────────────────────────
    st.markdown("---")
    sigma        = market_iv if market_iv and market_iv > 0 else (hv_value if hv_value else 0.25)
    iv_to_use    = market_iv or implied_volatility(market_price or 0, S0, K, r, T, option_type)
    bs_price     = black_scholes_price(S0, K, r, sigma, T, option_type)
    mc_val, mc_se = monte_carlo_price(S0, K, r, sigma, T, n_sim, option_type)
    greeks       = compute_greeks(S0, K, r, sigma, T, option_type)
    signals, verdict, verdict_color, iv_hv_ratio = generate_signal(
        bs_price, market_price, iv_to_use, hv_value,
        greeks["delta"], greeks["theta"], T_days, option_type)

    # ═══════════════════════════════════════════════════════════════════════
    # TABS
    # ═══════════════════════════════════════════════════════════════════════
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "  ◈ SIGNAL & PRICING  ",
        "  ◈ GREEKS EXPLAINED  ",
        "  ◈ PRICE PATHS  ",
        "  ◈ OPTIONS CHAIN  ",
        "  ◈ HEDGE REALITY CHECK  ",
    ])

    # ════════════════════════════════════════════════════════════════════════
    # TAB 1 — Signal & Pricing
    # ════════════════════════════════════════════════════════════════════════
    with tab1:
        verdict_class = f"verdict-{verdict.lower()}"
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:1rem;margin-bottom:1.5rem">
            <div class="section-head">Overall Verdict</div>
            <div class="verdict {verdict_class}">{verdict}</div>
        </div>""", unsafe_allow_html=True)

        if beginner_mode:
            explain = {
                "EXPENSIVE": "The market charges more than historical moves justify. Buyers pay a fear premium. Sellers may be getting a good deal.",
                "CHEAP":     "Options priced below what historical vol suggests. Buyers may be getting a bargain.",
                "NEUTRAL":   "Pricing looks fair. No strong edge from valuation alone.",
            }
            st.markdown(f'<div class="info-box">🎯 <strong>What this means:</strong> {explain.get(verdict,"")}</div>',
                        unsafe_allow_html=True)

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Model Price (BS)",   f"${bs_price:.3f}")
        c2.metric("Monte Carlo Price",  f"${mc_val:.3f}", f"±{mc_se:.3f}")
        if market_price:
            c3.metric("Market Price", f"${market_price:.3f}", f"{market_price-bs_price:+.3f} vs model",
                      delta_color="inverse" if market_price > bs_price else "normal")
        if iv_to_use:
            c4.metric("Implied Vol", f"{iv_to_use*100:.1f}%",
                      f"vs {hv_value*100:.1f}% HV" if hv_value else None)
        if hv_value:
            c5.metric("Historical Vol 30d", f"{hv_value*100:.1f}%")

        if beginner_mode:
            st.markdown("""
            <div class="info-box"><strong>How to read these:</strong><br>
            • <strong>Model Price</strong> — mathematical fair value.<br>
            • <strong>Monte Carlo</strong> — simulation-based price; ± shows precision.<br>
            • <strong>Market Price</strong> — what the market actually charges. Gap = opportunity or trap.<br>
            • <strong>IV vs HV</strong> — the single most important comparison. IV above HV = expensive.</div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        if iv_to_use and hv_value:
            st.markdown('<div class="section-label">Implied vs Historical Volatility</div>', unsafe_allow_html=True)
            fig_vol = go.Figure()
            if hv_hist is not None and len(hv_hist) > 0:
                fig_vol.add_trace(go.Scatter(
                    x=list(range(len(hv_hist))), y=(hv_hist*100).values,
                    mode="lines", name="Historical Vol (30d rolling)",
                    line=dict(color=TEAL, width=1.5),
                    fill="tozeroy", fillcolor="rgba(0,201,167,0.06)"))
            fig_vol.add_hline(y=iv_to_use*100, line=dict(color=AMBER, width=2, dash="dash"),
                annotation_text=f"  IV={iv_to_use*100:.1f}%", annotation_font=dict(color=AMBER, size=11))
            fig_vol.update_layout(**pl(height=220, xaxis_title="Trading days",
                yaxis_title="Volatility (%)",
                title=dict(text="Is the market pricing in MORE or LESS vol than history?",
                           font=dict(size=11, color=TEXT_DIM))))
            st.plotly_chart(fig_vol, use_container_width=True)

        st.markdown("---")
        st.markdown('<div class="section-label">Intelligence Signals</div>', unsafe_allow_html=True)
        for sig in signals:
            st.markdown(f"""
            <div class="signal-card {sig.get('level','neutral')}">
                <div class="signal-title">{sig['icon']} {sig['title']}</div>
                <div class="signal-plain">{sig['plain']}</div>
                <div class="signal-tip">💡 Trader Tip: {sig['trader_tip']}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown('<div class="section-label">Payoff at Expiry</div>', unsafe_allow_html=True)
        S_range  = np.linspace(max(1, S0*0.4), S0*1.85, 400)
        premium  = market_price or bs_price
        intrinsic = np.maximum(S_range - K, 0) if option_type=="call" else np.maximum(K - S_range, 0)
        pnl       = intrinsic - premium
        bs_curve  = np.array([black_scholes_price(s, K, r, sigma, T, option_type) for s in S_range])
        breakeven = K + premium if option_type=="call" else K - premium

        fig_pay = go.Figure()
        fig_pay.add_trace(go.Scatter(x=S_range, y=pnl, mode="lines",
            name="P&L at expiry", line=dict(color=TEAL, width=2),
            fill="tozeroy", fillcolor="rgba(0,201,167,0.05)"))
        fig_pay.add_trace(go.Scatter(x=S_range, y=bs_curve - premium, mode="lines",
            name=f"Value today ({T_days}d left)",
            line=dict(color=AMBER, width=1.5, dash="dot")))
        fig_pay.add_hline(y=0, line=dict(color=BORDER, width=1))
        fig_pay.add_vline(x=S0, line=dict(color=TEAL, width=1, dash="dash"),
            annotation_text=f"  Now ${S0:.2f}", annotation_font=dict(color=TEAL, size=10))
        fig_pay.add_vline(x=K, line=dict(color=RED, width=1, dash="dash"),
            annotation_text=f"  Strike ${K:.2f}", annotation_font=dict(color=RED, size=10))
        fig_pay.add_vline(x=breakeven, line=dict(color=AMBER, width=1, dash="dot"),
            annotation_text=f"  Breakeven ${breakeven:.2f}", annotation_font=dict(color=AMBER, size=10))
        fig_pay.update_layout(**pl(height=320,
            xaxis_title="Stock Price at Expiry", yaxis_title="Profit / Loss ($)",
            title=dict(text=f"Breakeven: ${breakeven:.2f} — stock needs to {'rise' if option_type=='call' else 'fall'} {abs((breakeven-S0)/S0*100):.1f}% from here",
                       font=dict(size=11, color=TEXT_DIM))))
        st.plotly_chart(fig_pay, use_container_width=True)

        if beginner_mode:
            st.markdown(f"""
            <div class="info-box">📊 <strong>Reading the payoff:</strong><br>
            • Teal line = your P&L at expiry for every possible stock price.<br>
            • You break even at <strong>${breakeven:.2f}</strong> — below that the premium works against you.<br>
            • Dotted amber = today's option value with {T_days}d left — you can sell early.<br>
            • Max loss = ${premium:.2f}/share (${premium*100:.0f}/contract). 
              Max gain = {'unlimited' if option_type=='call' else f'${K:.2f} if stock goes to $0'}.</div>
            """, unsafe_allow_html=True)


    # ════════════════════════════════════════════════════════════════════════
    # TAB 2 — Greeks
    # ════════════════════════════════════════════════════════════════════════
    with tab2:
        st.markdown("""
        <div class="info-box" style="margin-bottom:1.5rem">
            <strong>What are the Greeks?</strong> Sensitivities that tell you how your option's value
            changes in response to stock price moves, volatility, time, and interest rates.
        </div>""", unsafe_allow_html=True)

        greek_data = [
            ("Delta  Δ", "delta", greeks["delta"]),
            ("Gamma  Γ", "gamma", greeks["gamma"]),
            ("Vega  ν",  "vega",  greeks["vega"]),
            ("Theta  Θ", "theta", greeks["theta"]),
            ("Rho  ρ",   "rho",   greeks["rho"]),
        ]
        col_l, col_r = st.columns(2)
        for i, (display, key, val) in enumerate(greek_data):
            exp  = greek_explanation(key, val, option_type, S0, K)
            html = f"""
            <div class="greek-card">
                <div class="greek-name">{display}</div>
                <div class="greek-value">{val:+.4f}</div>
                <div class="greek-what">📌 {exp.get('what','')}</div>
                {'<div class="greek-analogy">💭 ' + exp.get('analogy','') + '</div>' if beginner_mode else ''}
                <div class="greek-watch">⚡ {exp.get('watch','')}</div>
            </div>"""
            (col_l if i % 2 == 0 else col_r).markdown(html, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown('<div class="section-label">How Greeks Change as Stock Price Moves</div>', unsafe_allow_html=True)
        S_vals    = np.linspace(max(1, S0*0.5), S0*1.5, 100)
        d_vals    = [compute_greeks(s, K, r, sigma, T, option_type)["delta"] for s in S_vals]
        g_vals    = [compute_greeks(s, K, r, sigma, T, option_type)["gamma"] for s in S_vals]
        v_vals    = [compute_greeks(s, K, r, sigma, T, option_type)["vega"]  for s in S_vals]
        th_vals   = [compute_greeks(s, K, r, sigma, T, option_type)["theta"] for s in S_vals]

        fig_g = make_subplots(rows=2, cols=2,
            subplot_titles=["Delta Δ — direction", "Gamma Γ — acceleration",
                            "Vega ν — vol sensitivity", "Theta Θ — daily decay"],
            vertical_spacing=0.18, horizontal_spacing=0.08)
        for (data, color, row, col) in [
            (d_vals, TEAL, 1, 1), (g_vals, AMBER, 1, 2),
            (v_vals, BLUE, 2, 1), (th_vals, RED, 2, 2)
        ]:
            r_int, g_int, b_int = int(color[1:3],16), int(color[3:5],16), int(color[5:7],16)
            fig_g.add_trace(go.Scatter(x=S_vals, y=data, mode="lines",
                line=dict(color=color, width=2),
                fill="tozeroy", fillcolor=f"rgba({r_int},{g_int},{b_int},0.06)",
                showlegend=False), row=row, col=col)
            fig_g.add_vline(x=S0, row=row, col=col,
                line=dict(color=TEXT_DIM, width=1, dash="dot"))

        ax = dict(gridcolor=MUTED, gridwidth=0.5, linecolor=BORDER, tickcolor=TEXT_DIM)
        fig_g.update_layout(**pl(height=500, showlegend=False))
        for i in range(1, 5):
            fig_g.update_layout(**{
                ("xaxis" if i == 1 else f"xaxis{i}"): ax,
                ("yaxis" if i == 1 else f"yaxis{i}"): ax,
            })
        fig_g.update_annotations(font=dict(family="Source Code Pro, monospace", color=TEXT_DIM, size=10))
        st.plotly_chart(fig_g, use_container_width=True)


    # ════════════════════════════════════════════════════════════════════════
    # TAB 3 — Price Paths
    # ════════════════════════════════════════════════════════════════════════
    with tab3:
        st.markdown(f"""
        <div class="info-box">
            <strong>What is this?</strong> Monte Carlo — {n_sim:,} possible futures for {ticker}
            using real vol ({sigma*100:.1f}%). Teal = profitable paths. Red = loss paths.
        </div>""", unsafe_allow_html=True)

        with st.spinner("Simulating paths…"):
            t_axis, paths = simulate_gbm_paths(S0, r, sigma, T, 252, 300)

        n_show  = st.slider("Paths to display", 20, 200, 60, 10)
        S_T_all = paths[:, -1]
        pct_p   = np.mean(S_T_all > K)*100 if option_type=="call" else np.mean(S_T_all < K)*100

        p1, p2, p3 = st.columns(3)
        p1.metric("Profitable paths",    f"{pct_p:.1f}%")
        p2.metric("Expected stock price", f"${np.mean(S_T_all):,.2f}")
        p3.metric("Simulated paths",     f"{n_sim:,}")

        fig_paths = go.Figure()
        for i in range(min(n_show, paths.shape[0])):
            prof  = paths[i,-1] > K if option_type=="call" else paths[i,-1] < K
            color = "rgba(0,201,167,0.2)" if prof else "rgba(255,107,107,0.15)"
            fig_paths.add_trace(go.Scatter(x=t_axis, y=paths[i], mode="lines",
                line=dict(color=color, width=0.7), showlegend=False))
        fig_paths.add_trace(go.Scatter(x=t_axis, y=np.mean(paths[:n_show],axis=0),
            mode="lines", line=dict(color=AMBER, width=2.5), name="Average path"))
        fig_paths.add_hline(y=K, line=dict(color=RED, width=1.5, dash="dash"),
            annotation_text=f"  Strike ${K:.2f}", annotation_font=dict(color=RED, size=10))
        fig_paths.add_hline(y=S0, line=dict(color=TEXT_DIM, width=1, dash="dot"),
            annotation_text=f"  Start ${S0:.2f}", annotation_font=dict(color=TEXT_DIM, size=10))
        fig_paths.update_layout(**pl(height=430,
            xaxis_title="Time (years)", yaxis_title=f"{ticker} Price",
            title=dict(text=f"Teal = profitable | Red = loss | {pct_p:.1f}% end profitable",
                       font=dict(size=11, color=TEXT_DIM))))
        st.plotly_chart(fig_paths, use_container_width=True)

        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(x=S_T_all, nbinsx=80,
            marker=dict(color=TEAL, line=dict(color=PLOT_BG, width=0.5)), opacity=0.7))
        fig_dist.add_vline(x=K, line=dict(color=RED, width=2, dash="dash"),
            annotation_text=f"  Strike ${K:.2f}", annotation_font=dict(color=RED, size=10))
        fig_dist.add_vline(x=S0, line=dict(color=AMBER, width=1.5, dash="dot"),
            annotation_text=f"  Today ${S0:.2f}", annotation_font=dict(color=AMBER, size=10))
        fig_dist.update_layout(**pl(height=250,
            xaxis_title=f"{ticker} Price at Expiry", yaxis_title="Simulated paths",
            title=dict(text="Distribution of all possible outcomes at expiry",
                       font=dict(size=11, color=TEXT_DIM))))
        st.plotly_chart(fig_dist, use_container_width=True)

        if beginner_mode:
            st.markdown(f"""
            <div class="info-box">
                Histogram = every possible outcome across {n_sim:,} simulated futures.
                Area to the {'right' if option_type=='call' else 'left'} of the red strike line = profitable outcomes ({pct_p:.1f}%).
                Compare to Delta (~{abs(greeks['delta'])*100:.0f}%) — they should be similar.
            </div>""", unsafe_allow_html=True)


    # ════════════════════════════════════════════════════════════════════════
    # TAB 4 — Options Chain
    # ════════════════════════════════════════════════════════════════════════
    with tab4:
        if chain_df is not None and not chain_df.empty:
            st.markdown(f'<div class="section-label">Live {option_type.upper()}S — {ticker} · {chosen_expiry}</div>', unsafe_allow_html=True)
            if beginner_mode:
                st.markdown("""
                <div class="info-box"><strong>Reading the chain:</strong><br>
                • <strong>Strike</strong> — price you'd buy/sell at.<br>
                • <strong>Last $</strong> — recent trade price (×100 = 1 contract cost).<br>
                • <strong>IV%</strong> — higher IV = more expensive = market expects bigger moves.<br>
                • <strong>Volume</strong> — today's activity. Higher = more liquid.<br>
                • <strong>OI</strong> — open contracts. Large OI = key price level traders watch.<br>
                • <strong>Model $</strong> — our fair value. <strong>Edge%</strong> = premium over/under model.</div>
                """, unsafe_allow_html=True)

            disp_cols = [c for c in ['strike','lastPrice','bid','ask','impliedVolatility','volume','openInterest']
                         if c in chain_df.columns]
            disp = chain_df[disp_cols].copy()
            disp['modelPrice'] = disp['strike'].apply(
                lambda k: round(black_scholes_price(S0, k, r, sigma, T, option_type), 3))
            if 'lastPrice' in disp.columns:
                disp['edge%'] = ((disp['lastPrice'] - disp['modelPrice']) / disp['modelPrice'] * 100).round(1)
            if 'impliedVolatility' in disp.columns:
                disp['impliedVolatility'] = (disp['impliedVolatility']*100).round(1).astype(str) + '%'
            disp.columns = [c.replace('impliedVolatility','IV%').replace('lastPrice','Last $')
                             .replace('openInterest','OI').replace('modelPrice','Model $') for c in disp.columns]
            disp['strike'] = disp['strike'].apply(lambda x: f"${x:.2f}")
            st.dataframe(disp.set_index('strike'), use_container_width=True, height=400)

            if 'impliedVolatility' in chain_df.columns:
                iv_data = chain_df[['strike','impliedVolatility']].dropna()
                iv_data = iv_data[iv_data['impliedVolatility'] > 0]
                if not iv_data.empty:
                    st.markdown("---")
                    st.markdown('<div class="section-label">Volatility Smile — Real Market Data</div>', unsafe_allow_html=True)
                    fig_sm = go.Figure()
                    fig_sm.add_trace(go.Scatter(x=iv_data['strike'], y=iv_data['impliedVolatility']*100,
                        mode="lines+markers", line=dict(color=TEAL, width=2),
                        marker=dict(color=AMBER, size=6, line=dict(color=PLOT_BG, width=1))))
                    fig_sm.add_vline(x=S0, line=dict(color=TEAL, width=1.5, dash="dash"),
                        annotation_text=f"  Now ${S0:.2f}", annotation_font=dict(color=TEAL, size=10))
                    fig_sm.update_layout(**pl(height=280, xaxis_title="Strike", yaxis_title="IV (%)",
                        title=dict(text="Real IV across strikes — the skew reveals where the market is most fearful",
                                   font=dict(size=11, color=TEXT_DIM))))
                    st.plotly_chart(fig_sm, use_container_width=True)

                    if beginner_mode:
                        st.markdown("""
                        <div class="info-box"><strong>The Volatility Smile/Skew</strong> — in theory (Black-Scholes)
                        IV should be flat across strikes. In reality OTM puts have higher IV (crash fear).
                        This skew tells you where the market is most worried.</div>""", unsafe_allow_html=True)
        else:
            st.info("Options chain not available. Try AAPL, TSLA, MSFT, NVDA, or SPY.")


    # ════════════════════════════════════════════════════════════════════════
    # TAB 5 — HEDGE REALITY CHECK  ← the new feature
    # ════════════════════════════════════════════════════════════════════════
    with tab5:
        st.markdown("""
        <div class="warning-box">
            <strong>⚠️ The Black-Scholes Assumption Nobody Talks About</strong><br>
            Black-Scholes prices options assuming you can rebalance your hedge <em>continuously</em> — 
            every millisecond, at zero cost. In reality, real traders rebalance daily, weekly, or monthly 
            and pay transaction costs each time. This tab shows you <strong>exactly how much that gap costs</strong> 
            — and why it matters before you sell any option.
        </div>""", unsafe_allow_html=True)

        # Controls
        hc1, hc2, hc3 = st.columns(3)
        with hc1:
            tc_pct = st.slider("Transaction cost per trade (%)", 0.0, 0.5, 0.1, 0.01,
                help="Realistic retail: ~0.05–0.10%. Includes bid/ask spread.") / 100
        with hc2:
            hedge_paths = st.select_slider("Simulation paths", [100, 200, 400, 800], 400,
                help="More paths = more accurate, but slower")
        with hc3:
            single_freq = st.selectbox("Inspect frequency",
                ["daily", "weekly", "monthly", "none"],
                format_func=lambda x: {"daily":"Daily rebalance","weekly":"Weekly rebalance",
                                       "monthly":"Monthly rebalance","none":"No rebalance (hold only)"}[x])

        run_hedge = st.button("▶  RUN HEDGE SIMULATION", use_container_width=False)

        hedge_key = f"hedge_{ticker}_{K}_{T_days}_{tc_pct}_{hedge_paths}"
        if run_hedge or "hedge_results" not in st.session_state or st.session_state.get("hedge_key") != hedge_key:
            with st.spinner("Simulating discrete hedging across all frequencies… (~10 seconds)"):
                all_results = compare_hedge_frequencies(
                    S0=S0, K=K, r=r, sigma=sigma, T=T,
                    option_type=option_type,
                    transaction_cost_pct=tc_pct,
                    n_paths=hedge_paths,
                )
            st.session_state["hedge_results"] = all_results
            st.session_state["hedge_key"]     = hedge_key

        all_results = st.session_state.get("hedge_results")
        if not all_results:
            st.info("Click 'RUN HEDGE SIMULATION' to see results.")
            st.stop()

        # ── Summary comparison cards ──────────────────────────────────────
        st.markdown("---")
        st.markdown('<div class="section-label">Hedging P&L by Rebalancing Frequency</div>', unsafe_allow_html=True)

        if beginner_mode:
            st.markdown("""
            <div class="info-box">
                <strong>What you're seeing:</strong> We simulated selling one option and delta-hedging it
                (buying/selling stock to stay neutral) at four different rebalancing frequencies, across
                hundreds of random stock price paths. <strong>Mean P&L</strong> = average profit from the hedge.
                <strong>Std Dev</strong> = how unpredictable the outcome is (lower = better hedge).
                <strong>Transaction Costs</strong> = what you actually pay in friction every time you rebalance.
            </div>""", unsafe_allow_html=True)

        freq_labels = {
            "daily":   ("Daily",   TEAL,  "Closest to BS theory. Best hedge quality, highest cost."),
            "weekly":  ("Weekly",  AMBER, "Good balance — common in practice."),
            "monthly": ("Monthly", BLUE,  "Cheap to run but significant tracking error."),
            "none":    ("No Rebalance", RED, "Raw premium only — completely unhedged. Pure directional bet."),
        }
        cols = st.columns(4)
        for ci, (freq, res) in enumerate(all_results.items()):
            label, color, desc = freq_labels[freq]
            mean_pnl = res["mean_pnl"]
            std_pnl  = res["std_pnl"]
            avg_tc   = res["avg_tc"]
            pct_win  = res["pct_profitable"]
            with cols[ci]:
                st.markdown(f"""
                <div class="hedge-card" style="border-left:3px solid {color}">
                    <div class="hedge-freq" style="color:{color}">{label}</div>
                    <div style="font-size:10px;color:var(--text-dim);margin-bottom:0.5rem">{desc}</div>
                    <div class="hedge-stat-row">
                        <div class="hedge-stat">
                            <div class="hedge-stat-val" style="color:{'var(--teal)' if mean_pnl>=0 else 'var(--red)'}">${mean_pnl:+.3f}</div>
                            <div class="hedge-stat-label">Mean P&L</div>
                        </div>
                        <div class="hedge-stat">
                            <div class="hedge-stat-val">${std_pnl:.3f}</div>
                            <div class="hedge-stat-label">Std Dev</div>
                        </div>
                    </div>
                    <div class="hedge-stat-row">
                        <div class="hedge-stat">
                            <div class="hedge-stat-val">{pct_win:.0f}%</div>
                            <div class="hedge-stat-label">% Profitable</div>
                        </div>
                        <div class="hedge-stat">
                            <div class="hedge-stat-val" style="color:var(--amber)">${avg_tc:.4f}</div>
                            <div class="hedge-stat-label">Avg Cost</div>
                        </div>
                    </div>
                </div>""", unsafe_allow_html=True)

        # ── P&L Distribution comparison chart ────────────────────────────
        st.markdown("---")
        st.markdown('<div class="section-label">P&L Distribution — All Frequencies Overlaid</div>', unsafe_allow_html=True)

        fig_hedge = go.Figure()
        colors_list = [TEAL, AMBER, BLUE, RED]
        for (freq, res), color in zip(all_results.items(), colors_list):
            label = freq_labels[freq][0]
            fig_hedge.add_trace(go.Histogram(
                x=res["pnl"], nbinsx=50,
                name=label,
                marker=dict(color=color, line=dict(color=PLOT_BG, width=0.3)),
                opacity=0.55,
                histnorm="probability density",
            ))
        fig_hedge.add_vline(x=0, line=dict(color="white", width=1.5, dash="dash"),
            annotation_text="  Break-even", annotation_font=dict(color="white", size=10))
        fig_hedge.update_layout(**pl(
            height=360, barmode="overlay",
            xaxis_title="Hedge P&L ($)", yaxis_title="Probability density",
            title=dict(text="Daily rebalancing = tight distribution. Monthly = wide. No hedge = widest.",
                       font=dict(size=11, color=TEXT_DIM)),
        ))
        st.plotly_chart(fig_hedge, use_container_width=True)

        # ── Single frequency deep-dive ────────────────────────────────────
        st.markdown("---")
        st.markdown(f'<div class="section-label">Deep Dive — {single_freq.capitalize()} Rebalancing</div>', unsafe_allow_html=True)

        res_single = all_results[single_freq]
        pnl_arr    = res_single["pnl"]
        bs_th      = res_single["bs_theoretical"]

        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Mean P&L",           f"${res_single['mean_pnl']:+.4f}")
        d2.metric("Std Dev (tracking error)", f"${res_single['std_pnl']:.4f}")
        d3.metric("BS Theoretical Price", f"${bs_th:.4f}")
        d4.metric("Avg Transaction Cost", f"${res_single['avg_tc']:.4f}")

        fig_dd = make_subplots(rows=1, cols=2,
            subplot_titles=["P&L Distribution", "P&L vs BS Expected"],
            horizontal_spacing=0.08)

        color_dd = freq_labels[single_freq][1]
        fig_dd.add_trace(go.Histogram(x=pnl_arr, nbinsx=40,
            marker=dict(color=color_dd, line=dict(color=PLOT_BG, width=0.3)),
            opacity=0.75, name="P&L"), row=1, col=1)
        fig_dd.add_vline(x=0, row=1, col=1,
            line=dict(color="white", width=1.5, dash="dash"))
        fig_dd.add_vline(x=res_single["mean_pnl"], row=1, col=1,
            line=dict(color=AMBER, width=1.5),
            annotation_text=f"  Mean ${res_single['mean_pnl']:+.3f}",
            annotation_font=dict(color=AMBER, size=10))

        sorted_pnl = np.sort(pnl_arr)
        cumulative  = np.arange(1, len(sorted_pnl)+1) / len(sorted_pnl)
        fig_dd.add_trace(go.Scatter(x=sorted_pnl, y=cumulative,
            mode="lines", line=dict(color=color_dd, width=2), name="CDF"), row=1, col=2)
        fig_dd.add_vline(x=0, row=1, col=2,
            line=dict(color="white", width=1, dash="dash"))
        fig_dd.add_hline(y=0.5, row=1, col=2,
            line=dict(color=AMBER, width=1, dash="dot"),
            annotation_text="  50th pct", annotation_font=dict(color=AMBER, size=9))

        ax = dict(gridcolor=MUTED, gridwidth=0.5, linecolor=BORDER, tickcolor=TEXT_DIM)
        fig_dd.update_layout(**pl(height=320, showlegend=False,
            margin=dict(t=40, b=30, l=10, r=10)))
        fig_dd.update_layout(xaxis=ax, yaxis=ax, xaxis2=ax, yaxis2=ax)
        fig_dd.update_annotations(font=dict(family="Source Code Pro, monospace", color=TEXT_DIM, size=10))
        st.plotly_chart(fig_dd, use_container_width=True)

        if beginner_mode:
            mean = res_single["mean_pnl"]
            std  = res_single["std_pnl"]
            tc   = res_single["avg_tc"]
            st.markdown(f"""
            <div class="info-box">
                <strong>What this tells you ({single_freq} rebalancing):</strong><br>
                On average the hedge makes <strong>${mean:+.4f}</strong> per option across all simulated paths.
                But there's a standard deviation of <strong>${std:.4f}</strong> — meaning your actual result
                in any single trade could be very different. Transaction costs alone average 
                <strong>${tc:.4f}</strong> per path with {tc_pct*100:.2f}% cost per trade.<br><br>
                <strong>The key lesson:</strong> The more you rebalance, the tighter your hedge — 
                but transaction costs eat into profit. Real traders find the sweet spot between 
                hedge quality and cost. This is the real-world flaw in Black-Scholes that the commenter raised.
            </div>""", unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center;padding:1rem 0">
    <div style="font-family:'Source Code Pro',monospace;font-size:9px;letter-spacing:0.2em;color:#2E3F52;text-transform:uppercase">
        Options Decoded · Real-time data via Yahoo Finance · For educational purposes only · Not financial advice
    </div>
</div>""", unsafe_allow_html=True)