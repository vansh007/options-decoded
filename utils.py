import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
import yfinance as yf
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# ═══════════════════════════════════════════════════════════════════════════
# MARKET DATA
# ═══════════════════════════════════════════════════════════════════════════

def get_stock_info(ticker: str):
    """Fetch live stock price, company name, and basic info."""
    try:
        tk = yf.Ticker(ticker)
        info = tk.info
        hist = tk.history(period="1d", interval="1m")
        price = float(hist['Close'].iloc[-1]) if not hist.empty else (
            info.get('currentPrice') or info.get('regularMarketPrice'))
        if not price:
            return None
        return {
            "price": price,
            "name": info.get("longName", ticker),
            "sector": info.get("sector", "—"),
            "market_cap": info.get("marketCap"),
            "pe_ratio": info.get("trailingPE"),
            "week_52_high": info.get("fiftyTwoWeekHigh"),
            "week_52_low": info.get("fiftyTwoWeekLow"),
            "avg_volume": info.get("averageVolume"),
        }
    except:
        return None


def get_historical_volatility(ticker: str, window: int = 30):
    """Compute 30-day historical (realized) volatility from daily returns."""
    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(period="6mo")
        if hist.empty or len(hist) < window:
            return None, None
        log_returns = np.log(hist['Close'] / hist['Close'].shift(1)).dropna()
        hv_series = log_returns.rolling(window).std() * np.sqrt(252)
        return float(hv_series.iloc[-1]), hv_series.dropna()
    except:
        return None, None


def get_options_chain(ticker: str):
    """Fetch real options chain from Yahoo Finance."""
    try:
        tk = yf.Ticker(ticker)
        expirations = tk.options
        if not expirations:
            return None, None
        today = datetime.today()
        valid = [
            (e, (datetime.strptime(e, "%Y-%m-%d") - today).days)
            for e in expirations
            if (datetime.strptime(e, "%Y-%m-%d") - today).days >= 7
        ]
        if not valid:
            return None, None
        return tk, [e for e, _ in sorted(valid, key=lambda x: x[1])[:6]]
    except:
        return None, None


def fetch_option_chain_for_expiry(ticker_obj, expiry: str, spot: float):
    """Return calls and puts for a given expiry, filtered near ATM."""
    try:
        chain = ticker_obj.option_chain(expiry)
        calls = chain.calls[(chain.calls['strike'] >= spot * 0.7) &
                            (chain.calls['strike'] <= spot * 1.3)].copy()
        puts  = chain.puts[(chain.puts['strike']  >= spot * 0.7) &
                           (chain.puts['strike']  <= spot * 1.3)].copy()
        return calls, puts
    except:
        return None, None


# ═══════════════════════════════════════════════════════════════════════════
# PRICING MODELS
# ═══════════════════════════════════════════════════════════════════════════

def black_scholes_price(S0, K, r, sigma, T, option_type="call"):
    if T <= 0 or sigma <= 0:
        return max(S0 - K, 0) if option_type == "call" else max(K - S0, 0)
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)


def simulate_gbm_paths(S0, r, sigma, T, steps, n_paths):
    """Vectorized GBM — returns (t_array, paths matrix)."""
    dt = T / steps
    t  = np.linspace(0, T, steps + 1)
    Z  = np.random.standard_normal((n_paths, steps))
    log_ret = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    paths = S0 * np.exp(np.concatenate(
        [np.zeros((n_paths, 1)), np.cumsum(log_ret, axis=1)], axis=1))
    return t, paths


def monte_carlo_price(S0, K, r, sigma, T, n_paths=30000, option_type="call"):
    """Antithetic-variate Monte Carlo pricing with standard error."""
    Z      = np.random.standard_normal(n_paths // 2)
    Z_full = np.concatenate([Z, -Z])
    S_T    = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z_full)
    payoff = np.maximum(S_T - K, 0) if option_type == "call" else np.maximum(K - S_T, 0)
    disc   = np.exp(-r * T)
    return disc * np.mean(payoff), disc * np.std(payoff) / np.sqrt(n_paths)


def implied_volatility(market_price, S0, K, r, T, option_type="call"):
    try:
        intrinsic = max(S0 - K, 0) if option_type == "call" else max(K - S0, 0)
        if market_price <= intrinsic or market_price <= 0:
            return None
        f = lambda sigma: black_scholes_price(S0, K, r, sigma, T, option_type) - market_price
        return brentq(f, 1e-4, 10.0, maxiter=500)
    except:
        return None


# ═══════════════════════════════════════════════════════════════════════════
# GREEKS
# ═══════════════════════════════════════════════════════════════════════════

def compute_greeks(S0, K, r, sigma, T, option_type="call"):
    if T <= 0 or sigma <= 0:
        return {"delta": 0, "gamma": 0, "vega": 0, "theta": 0, "rho": 0}
    d1     = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2     = d1 - sigma * np.sqrt(T)
    pdf_d1 = norm.pdf(d1)
    gamma  = pdf_d1 / (S0 * sigma * np.sqrt(T))
    vega   = S0 * pdf_d1 * np.sqrt(T) / 100
    if option_type == "call":
        delta = norm.cdf(d1)
        theta = (-(S0 * pdf_d1 * sigma) / (2 * np.sqrt(T))
                 - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
        rho   = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    else:
        delta = norm.cdf(d1) - 1
        theta = (-(S0 * pdf_d1 * sigma) / (2 * np.sqrt(T))
                 + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
        rho   = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


def greek_explanation(name, value, option_type, S0, K):
    """Return plain-English explanation dict for each Greek."""
    return {
        "delta": {
            "what":    f"For every $1 the stock {'rises' if option_type=='call' else 'falls'}, this option gains ${abs(value):.3f}.",
            "analogy": f"Owning this option feels like holding {abs(value)*100:.1f} shares per 100 contracts.",
            "watch":   "Delta → 1.0 deep ITM, → 0 deep OTM. At-the-money ≈ 0.50.",
        },
        "gamma": {
            "what":    f"Delta changes by {value:.4f} for every $1 stock move.",
            "analogy": "Gamma is acceleration — highest at-the-money near expiry. Small stock moves create big option swings.",
            "watch":   "High gamma = explosive option value changes. Exciting and risky.",
        },
        "vega": {
            "what":    f"A 1% rise in implied volatility adds ${value:.3f} to this option.",
            "analogy": "Vega is your fear exposure. Market panic spikes vol → your option gains even if stock stays flat.",
            "watch":   "Buyers want high Vega. Sellers want low Vega.",
        },
        "theta": {
            "what":    f"This option loses ${abs(value):.4f} every single day from time decay alone.",
            "analogy": f"A melting ice cube. Over 30 days: ~${abs(value)*30:.2f} evaporates just from time passing.",
            "watch":   "Theta accelerates near expiry. Sellers collect it. Buyers race against it.",
        },
        "rho": {
            "what":    f"A 1% rate rise {'adds' if value > 0 else 'removes'} ${abs(value):.4f} from this option.",
            "analogy": "Usually the least important Greek for short-dated options. Matters more for LEAPS (1yr+).",
            "watch":   "In aggressive rate-hike cycles Rho becomes worth monitoring.",
        },
    }.get(name, {})


# ═══════════════════════════════════════════════════════════════════════════
# DISCRETE DELTA HEDGING SIMULATION
# (addresses the commenter's valid point about continuous-time rebalancing)
# ═══════════════════════════════════════════════════════════════════════════

def simulate_discrete_hedge(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    option_type: str = "call",
    rebal_freq: str = "daily",       # "daily" | "weekly" | "monthly" | "none"
    transaction_cost_pct: float = 0.001,  # 0.1% per trade (realistic retail)
    n_paths: int = 500,
    steps_per_year: int = 252,
) -> dict:
    """
    Simulate discrete delta-hedging for a short option position.

    The hedger SELLS one option and tries to stay delta-neutral by
    trading the underlying stock at discrete intervals.  In
    continuous time (BS) the hedge is perfect.  In discrete time
    it is not — tracking error and transaction costs eat into P&L.

    Returns a dict with per-path P&L distributions and summary stats
    for each rebalancing frequency so the UI can compare them.
    """
    total_steps = int(steps_per_year * T)
    total_steps = max(total_steps, 10)
    dt = T / total_steps

    # Map frequency → rebalance every N steps
    freq_map = {
        "daily":   max(1, int(steps_per_year / 252)),
        "weekly":  max(1, int(steps_per_year / 52)),
        "monthly": max(1, int(steps_per_year / 12)),
        "none":    total_steps,   # rebalance only at inception
    }
    rebal_every = freq_map.get(rebal_freq, 1)

    # Simulate stock paths
    np.random.seed(42)
    Z = np.random.standard_normal((n_paths, total_steps))
    log_ret = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    S = np.zeros((n_paths, total_steps + 1))
    S[:, 0] = S0
    for i in range(total_steps):
        S[:, i + 1] = S[:, i] * np.exp(log_ret[:, i])

    pnl_paths     = np.zeros(n_paths)
    hedge_errors  = []        # tracking error per rebalance across all paths
    cost_per_path = np.zeros(n_paths)

    for path_i in range(n_paths):
        cash        = 0.0
        shares_held = 0.0
        total_tc    = 0.0

        # Inception: sell option, delta-hedge immediately
        t_now   = T
        S_now   = S[path_i, 0]
        option_premium = black_scholes_price(S_now, K, r, sigma, t_now, option_type)
        cash += option_premium   # receive premium from selling option

        g       = compute_greeks(S_now, K, r, sigma, t_now, option_type)
        delta0  = g["delta"]
        # Buy delta shares to hedge
        trade   = delta0 - shares_held
        tc      = abs(trade * S_now) * transaction_cost_pct
        cash   -= trade * S_now + tc
        shares_held = delta0
        total_tc   += tc

        for step in range(1, total_steps + 1):
            S_now = S[path_i, step]
            t_rem = max(T - step * dt, 1e-6)

            # Rebalance only at scheduled intervals (or final step)
            if step % rebal_every == 0 or step == total_steps:
                new_delta = compute_greeks(S_now, K, r, sigma, t_rem, option_type)["delta"]
                trade     = new_delta - shares_held
                tc        = abs(trade * S_now) * transaction_cost_pct
                cash     -= trade * S_now + tc
                hedge_errors.append(abs(new_delta - shares_held))
                shares_held = new_delta
                total_tc   += tc

        # Settlement: close stock position, pay option payoff
        cash  += shares_held * S_now           # sell stock
        if option_type == "call":
            payoff = max(S_now - K, 0)
        else:
            payoff = max(K - S_now, 0)
        cash -= payoff                         # pay option holder

        pnl_paths[path_i]  = cash
        cost_per_path[path_i] = total_tc

    bs_theoretical = black_scholes_price(S0, K, r, sigma, T, option_type)

    return {
        "pnl":              pnl_paths,
        "mean_pnl":         float(np.mean(pnl_paths)),
        "std_pnl":          float(np.std(pnl_paths)),
        "pct_profitable":   float(np.mean(pnl_paths > 0) * 100),
        "avg_tc":           float(np.mean(cost_per_path)),
        "total_tc":         float(np.sum(cost_per_path)),
        "mean_hedge_error": float(np.mean(hedge_errors)) if hedge_errors else 0.0,
        "bs_theoretical":   bs_theoretical,
        "rebal_freq":       rebal_freq,
        "n_paths":          n_paths,
    }


def compare_hedge_frequencies(
    S0, K, r, sigma, T, option_type="call",
    transaction_cost_pct=0.001, n_paths=400
):
    """
    Run discrete hedging simulation for all four frequencies.
    Returns a dict keyed by frequency label.
    """
    results = {}
    for freq in ["daily", "weekly", "monthly", "none"]:
        results[freq] = simulate_discrete_hedge(
            S0=S0, K=K, r=r, sigma=sigma, T=T,
            option_type=option_type,
            rebal_freq=freq,
            transaction_cost_pct=transaction_cost_pct,
            n_paths=n_paths,
        )
    return results


# ═══════════════════════════════════════════════════════════════════════════
# SIGNAL ENGINE
# ═══════════════════════════════════════════════════════════════════════════

def generate_signal(bs_price, market_price, iv, hv, delta, theta, T_days, option_type):
    """Return plain-English signals, verdict, and IV/HV ratio."""
    signals      = []
    verdict      = "NEUTRAL"
    verdict_color = "gray"
    iv_hv_ratio  = None

    if iv and hv and hv > 0:
        iv_hv_ratio = iv / hv
        if iv_hv_ratio > 1.3:
            signals.append({
                "icon": "🔴", "level": "warning",
                "title": "Options Are Expensive (High IV vs HV)",
                "plain": (f"The market is pricing in {iv*100:.1f}% annualized moves, but the stock "
                          f"has only been moving {hv*100:.1f}% historically. "
                          f"Options cost MORE than recent price action justifies."),
                "trader_tip": (f"Selling options could be profitable — you'd collect inflated premium. "
                               f"Buying means paying for volatility that may not materialise."),
            })
            verdict, verdict_color = "EXPENSIVE", "red"
        elif iv_hv_ratio < 0.75:
            signals.append({
                "icon": "🟢", "level": "good",
                "title": "Options Are Cheap (Low IV vs HV)",
                "plain": (f"The market is pricing in only {iv*100:.1f}% moves but the stock has been "
                          f"moving {hv*100:.1f}%. Options cost LESS than price history justifies."),
                "trader_tip": "Good buying environment — you're paying below-average premium.",
            })
            verdict, verdict_color = "CHEAP", "green"
        else:
            signals.append({
                "icon": "🟡", "level": "neutral",
                "title": "Options Are Fairly Priced",
                "plain": f"IV ({iv*100:.1f}%) and historical vol ({hv*100:.1f}%) are close. No strong mispricing.",
                "trader_tip": "No vol edge. Focus on directional conviction.",
            })

    if market_price and bs_price and bs_price > 0:
        diff_pct = (market_price - bs_price) / bs_price * 100
        if abs(diff_pct) > 10:
            signals.append({
                "icon": "📊", "level": "warning" if diff_pct > 0 else "good",
                "title": f"Market Price {abs(diff_pct):.1f}% {'Above' if diff_pct>0 else 'Below'} Model Price",
                "plain": (f"Model says ${bs_price:.2f}, market charges ${market_price:.2f}. "
                          + (f"You're paying a {abs(diff_pct):.1f}% premium over fair value."
                             if diff_pct > 0 else f"Rare {abs(diff_pct):.1f}% discount vs fair value.")),
                "trader_tip": "Large model-market gaps often reflect supply/demand or upcoming events.",
            })

    if T_days < 30:
        signals.append({
            "icon": "⏰", "level": "warning",
            "title": f"Only {T_days:.0f} Days Left — Theta Burns Fast",
            "plain": (f"Losing ${abs(theta):.4f}/day to time decay. "
                      f"With {T_days} days left this accelerates exponentially."),
            "trader_tip": "Buyers need an immediate catalyst. Sellers are in their sweet spot.",
        })

    if delta:
        prob = abs(delta) * 100
        signals.append({
            "icon": "🎯", "level": "neutral",
            "title": f"~{prob:.0f}% Probability of Finishing In-The-Money",
            "plain": (f"Delta {delta:.2f} implies ~{prob:.0f}% odds this option has value at expiry. "
                      + ("Near coin-flip." if 40 < prob < 60
                         else "High probability, lower leverage." if prob > 70
                         else "Long shot, high potential return.")),
            "trader_tip": f"Every $1 move {'up' if option_type=='call' else 'down'} → option gains ~${abs(delta):.2f}.",
        })

    return signals, verdict, verdict_color, iv_hv_ratio