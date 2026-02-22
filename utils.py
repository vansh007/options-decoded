import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


# ── Real Market Data ────────────────────────────────────────────────────────

def get_stock_info(ticker: str):
    """Fetch live stock price, company name, and basic info."""
    try:
        tk = yf.Ticker(ticker)
        info = tk.info
        hist = tk.history(period="1d", interval="1m")
        price = float(hist['Close'].iloc[-1]) if not hist.empty else info.get('currentPrice', info.get('regularMarketPrice', None))
        return {
            "price": price,
            "name": info.get("longName", ticker),
            "sector": info.get("sector", "—"),
            "market_cap": info.get("marketCap", None),
            "pe_ratio": info.get("trailingPE", None),
            "week_52_high": info.get("fiftyTwoWeekHigh", None),
            "week_52_low": info.get("fiftyTwoWeekLow", None),
            "avg_volume": info.get("averageVolume", None),
        }
    except Exception as e:
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
        current_hv = float(hv_series.iloc[-1])
        hv_history = hv_series.dropna()
        return current_hv, hv_history
    except:
        return None, None


def get_options_chain(ticker: str):
    """Fetch real options chain from Yahoo Finance."""
    try:
        tk = yf.Ticker(ticker)
        expirations = tk.options
        if not expirations:
            return None, None
        # Pick nearest expiry that's at least 7 days out
        today = datetime.today()
        valid_exps = []
        for exp in expirations:
            exp_date = datetime.strptime(exp, "%Y-%m-%d")
            days = (exp_date - today).days
            if days >= 7:
                valid_exps.append((exp, days))
        if not valid_exps:
            return None, None
        return tk, [e[0] for e in sorted(valid_exps, key=lambda x: x[1])[:6]]
    except:
        return None, None


def fetch_option_chain_for_expiry(ticker_obj, expiry: str, spot: float):
    """Return calls and puts for a given expiry, filtered near ATM."""
    try:
        chain = ticker_obj.option_chain(expiry)
        calls = chain.calls.copy()
        puts = chain.puts.copy()
        # Filter to strikes within ±30% of spot
        calls = calls[(calls['strike'] >= spot * 0.7) & (calls['strike'] <= spot * 1.3)]
        puts = puts[(puts['strike'] >= spot * 0.7) & (puts['strike'] <= spot * 1.3)]
        return calls, puts
    except:
        return None, None


# ── Pricing Models ──────────────────────────────────────────────────────────

def black_scholes_price(S0, K, r, sigma, T, option_type="call"):
    if T <= 0 or sigma <= 0:
        if option_type == "call":
            return max(S0 - K, 0)
        return max(K - S0, 0)
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)


def simulate_gbm_paths(S0, r, sigma, T, steps, n_paths):
    dt = T / steps
    t = np.linspace(0, T, steps + 1)
    Z = np.random.standard_normal((n_paths, steps))
    log_returns = (r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z
    log_paths = np.concatenate(
        [np.zeros((n_paths, 1)), np.cumsum(log_returns, axis=1)], axis=1
    )
    return t, S0 * np.exp(log_paths)


def monte_carlo_price(S0, K, r, sigma, T, n_paths=30000, option_type="call"):
    Z = np.random.standard_normal(n_paths // 2)
    Z_full = np.concatenate([Z, -Z])
    S_T = S0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z_full)
    payoff = np.maximum(S_T - K, 0) if option_type == "call" else np.maximum(K - S_T, 0)
    disc = np.exp(-r * T)
    price = disc * np.mean(payoff)
    se = disc * np.std(payoff) / np.sqrt(n_paths)
    return price, se


# ── Greeks ──────────────────────────────────────────────────────────────────

def compute_greeks(S0, K, r, sigma, T, option_type="call"):
    if T <= 0 or sigma <= 0:
        return {"delta": 0, "gamma": 0, "vega": 0, "theta": 0, "rho": 0}
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    pdf_d1 = norm.pdf(d1)
    gamma = pdf_d1 / (S0 * sigma * np.sqrt(T))
    vega = S0 * pdf_d1 * np.sqrt(T) / 100
    if option_type == "call":
        delta = norm.cdf(d1)
        theta = (-(S0 * pdf_d1 * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    else:
        delta = norm.cdf(d1) - 1
        theta = (-(S0 * pdf_d1 * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


# ── Implied Volatility ───────────────────────────────────────────────────────

def implied_volatility(market_price, S0, K, r, T, option_type="call"):
    try:
        intrinsic = max(S0 - K, 0) if option_type == "call" else max(K - S0, 0)
        if market_price <= intrinsic or market_price <= 0:
            return None
        f = lambda sigma: black_scholes_price(S0, K, r, sigma, T, option_type) - market_price
        return brentq(f, 1e-4, 10.0, maxiter=500)
    except:
        return None


# ── Signal Engine — Plain English Analysis ──────────────────────────────────

def generate_signal(bs_price, market_price, iv, hv, delta, theta, T_days, option_type):
    """Returns a plain-English signal dict for traders and beginners alike."""
    signals = []
    verdict = "NEUTRAL"
    verdict_color = "gray"

    # IV vs HV — the core edge signal
    iv_hv_ratio = None
    if iv and hv:
        iv_hv_ratio = iv / hv
        if iv_hv_ratio > 1.3:
            signals.append({
                "icon": "🔴",
                "title": "Options Are Expensive (High IV vs HV)",
                "plain": f"The market is pricing in {iv*100:.1f}% annualized moves, but the stock has only been moving {hv*100:.1f}% historically. "
                         f"This means options cost MORE than they 'should' based on recent price action.",
                "trader_tip": f"{'Selling' if option_type == 'call' else 'Selling puts'} could be profitable here — you'd collect premium that may be inflated. "
                              f"Buying options right now means you're paying a premium for volatility that may not materialize.",
                "level": "warning"
            })
            verdict = "EXPENSIVE"
            verdict_color = "red"
        elif iv_hv_ratio < 0.75:
            signals.append({
                "icon": "🟢",
                "title": "Options Are Cheap (Low IV vs HV)",
                "plain": f"The market is pricing in only {iv*100:.1f}% annualized moves, but the stock has been moving {hv*100:.1f}% historically. "
                         f"Options cost LESS than you'd expect given how the stock has actually been trading.",
                "trader_tip": f"Buying options could offer good value — you're paying relatively little for protection or upside exposure. "
                              f"This is often a good environment for buying calls or puts.",
                "level": "good"
            })
            verdict = "CHEAP"
            verdict_color = "green"
        else:
            signals.append({
                "icon": "🟡",
                "title": "Options Are Fairly Priced",
                "plain": f"IV ({iv*100:.1f}%) and historical volatility ({hv*100:.1f}%) are close. "
                         f"The market isn't dramatically over- or underpricing risk right now.",
                "trader_tip": "No strong edge from vol mispricing. Focus on direction or wait for a better setup.",
                "level": "neutral"
            })

    # Market price vs model price
    if market_price and bs_price:
        diff_pct = (market_price - bs_price) / bs_price * 100
        if abs(diff_pct) > 10:
            direction = "above" if diff_pct > 0 else "below"
            signals.append({
                "icon": "📊",
                "title": f"Market Price is {abs(diff_pct):.1f}% {direction.capitalize()} Model Price",
                "plain": f"Our Black-Scholes model says this option is worth ${bs_price:.2f}, "
                         f"but the market is charging ${market_price:.2f}. "
                         + (f"You're paying a {abs(diff_pct):.1f}% premium over fair value." if diff_pct > 0
                            else f"The market is offering a {abs(diff_pct):.1f}% discount vs fair value."),
                "trader_tip": "Models aren't perfect — market prices reflect supply/demand, news, and sentiment too. "
                              "But large gaps can signal opportunity or a trap.",
                "level": "warning" if diff_pct > 10 else "good"
            })

    # Time decay warning
    if T_days < 30:
        signals.append({
            "icon": "⏰",
            "title": f"Only {T_days:.0f} Days to Expiry — Theta Burning Fast",
            "plain": f"This option loses ~${abs(theta):.3f} per day just from time passing, "
                     f"even if the stock doesn't move. With less than 30 days left, this decay accelerates.",
            "trader_tip": "Option buyers face an uphill battle this close to expiry. The stock needs to move quickly and in your direction. "
                          "Sellers, on the other hand, love this environment.",
            "level": "warning"
        })

    # Delta interpretation
    if delta:
        prob_itm = abs(delta) * 100
        signals.append({
            "icon": "🎯",
            "title": f"~{prob_itm:.0f}% Chance of Finishing In-The-Money",
            "plain": f"Delta of {delta:.2f} means the market implies roughly a {prob_itm:.0f}% probability "
                     f"this option expires with value. "
                     + ("It's close to 50/50 — very sensitive to price moves." if 40 < prob_itm < 60
                        else "High probability of profit, but less upside leverage." if prob_itm > 70
                        else "Long shot — higher potential return, lower probability."),
            "trader_tip": f"Every $1 the stock moves {'up' if option_type == 'call' else 'down'}, "
                          f"this option gains ~${abs(delta):.2f} in value.",
            "level": "neutral"
        })

    return signals, verdict, verdict_color, iv_hv_ratio


def greek_explanation(name, value, option_type, S0, K):
    """Return plain English explanation for each Greek."""
    explanations = {
        "delta": {
            "what": f"For every $1 the stock {'rises' if option_type == 'call' else 'falls'}, this option gains ${abs(value):.3f}.",
            "analogy": f"Think of Delta as your 'effective stock exposure'. Owning this option feels like owning {abs(value)*100:.1f} shares of stock per 100 contracts.",
            "watch": "Delta moves toward 1.0 as the option goes deeper in-the-money, and toward 0 as it goes out-of-the-money."
        },
        "gamma": {
            "what": f"Delta itself changes by {value:.4f} for every $1 move in the stock.",
            "analogy": "Gamma is like acceleration — it tells you how fast your option's sensitivity is changing. High gamma = option value changes rapidly.",
            "watch": "Gamma is highest when the option is at-the-money and near expiry. This is both exciting and risky."
        },
        "vega": {
            "what": f"For every 1% increase in implied volatility, this option gains ${value:.3f}.",
            "analogy": "If the market suddenly gets more fearful/uncertain (volatility spikes), Vega tells you how much you gain — even if the stock price doesn't change.",
            "watch": "Options buyers want high Vega (they profit from vol spikes). Sellers want low Vega."
        },
        "theta": {
            "what": f"This option loses ${abs(value):.4f} every single day, just from time passing.",
            "analogy": "Theta is like a melting ice cube. Every day you hold the option, a tiny bit of its value evaporates — even if nothing else changes.",
            "watch": f"Over 30 days, this option loses ~${abs(value)*30:.2f} from time decay alone. Buyers need the stock to move. Sellers collect this decay."
        },
        "rho": {
            "what": f"For every 1% rise in interest rates, this option {'gains' if value > 0 else 'loses'} ${abs(value):.4f}.",
            "analogy": "Rho is usually the least important Greek for short-term options. It matters more for long-dated options (LEAPS).",
            "watch": "In a rising rate environment (like 2022-2023), Rho became more important than usual."
        }
    }
    return explanations.get(name, {})

# import numpy as np
# from scipy.stats import norm
# from scipy.optimize import brentq


# # ── Vectorized GBM Simulation ──────────────────────────────────────────────
# def simulate_gbm_paths(S0, r, sigma, T, steps, n_paths):
#     dt = T / steps
#     t = np.linspace(0, T, steps + 1)
#     Z = np.random.standard_normal((n_paths, steps))
#     log_returns = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
#     log_paths = np.concatenate(
#         [np.zeros((n_paths, 1)), np.cumsum(log_returns, axis=1)], axis=1
#     )
#     paths = S0 * np.exp(log_paths)
#     return t, paths


# # ── Monte Carlo Pricing (with Antithetic Variates) ─────────────────────────
# def monte_carlo_option_price(S0, r, sigma, T, K, n_paths=50000, option_type="call"):
#     Z = np.random.standard_normal(n_paths // 2)
#     Z_full = np.concatenate([Z, -Z])  # antithetic variates
#     S_T = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z_full)
#     if option_type == "call":
#         payoff = np.maximum(S_T - K, 0)
#     else:
#         payoff = np.maximum(K - S_T, 0)
#     discount = np.exp(-r * T)
#     price = discount * np.mean(payoff)
#     se = discount * np.std(payoff) / np.sqrt(n_paths)
#     return price, se


# # ── Black-Scholes Analytical Price ────────────────────────────────────────
# def black_scholes_price(S0, K, r, sigma, T, option_type="call"):
#     if T <= 0:
#         if option_type == "call":
#             return max(S0 - K, 0)
#         return max(K - S0, 0)
#     d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
#     d2 = d1 - sigma * np.sqrt(T)
#     if option_type == "call":
#         return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
#     return K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)


# # ── Greeks ─────────────────────────────────────────────────────────────────
# def compute_greeks(S0, K, r, sigma, T, option_type="call"):
#     if T <= 0:
#         return {"delta": 0, "gamma": 0, "vega": 0, "theta": 0, "rho": 0}
#     d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
#     d2 = d1 - sigma * np.sqrt(T)
#     pdf_d1 = norm.pdf(d1)
#     if option_type == "call":
#         delta = norm.cdf(d1)
#         rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
#         theta = (
#             -(S0 * pdf_d1 * sigma) / (2 * np.sqrt(T))
#             - r * K * np.exp(-r * T) * norm.cdf(d2)
#         ) / 365
#     else:
#         delta = norm.cdf(d1) - 1
#         rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
#         theta = (
#             -(S0 * pdf_d1 * sigma) / (2 * np.sqrt(T))
#             + r * K * np.exp(-r * T) * norm.cdf(-d2)
#         ) / 365
#     gamma = pdf_d1 / (S0 * sigma * np.sqrt(T))
#     vega = S0 * pdf_d1 * np.sqrt(T) / 100
#     return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


# # ── Convergence Data ───────────────────────────────────────────────────────
# def convergence_data(S0, K, r, sigma, T, option_type="call", max_paths=20000):
#     bs = black_scholes_price(S0, K, r, sigma, T, option_type)
#     path_counts = np.unique(np.logspace(2, np.log10(max_paths), 40).astype(int))
#     Z_all = np.random.standard_normal(max_paths)
#     S_all = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z_all)
#     if option_type == "call":
#         payoffs = np.maximum(S_all - K, 0) * np.exp(-r * T)
#     else:
#         payoffs = np.maximum(K - S_all, 0) * np.exp(-r * T)
#     mc_prices, errors = [], []
#     for n in path_counts:
#         p = np.mean(payoffs[:n])
#         mc_prices.append(p)
#         errors.append(abs(p - bs))
#     return path_counts, mc_prices, errors, bs


# # ── Asian Option (Arithmetic Average) ─────────────────────────────────────
# def asian_option_mc(S0, K, r, sigma, T, steps, n_paths, option_type="call"):
#     dt = T / steps
#     Z = np.random.standard_normal((n_paths, steps))
#     log_returns = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
#     paths = S0 * np.exp(np.cumsum(log_returns, axis=1))
#     avg_price = np.mean(paths, axis=1)
#     if option_type == "call":
#         payoff = np.maximum(avg_price - K, 0)
#     else:
#         payoff = np.maximum(K - avg_price, 0)
#     return np.exp(-r * T) * np.mean(payoff)


# # ── Barrier Option (Down-and-Out Call) ────────────────────────────────────
# def barrier_option_mc(S0, K, B, r, sigma, T, steps, n_paths):
#     dt = T / steps
#     Z = np.random.standard_normal((n_paths, steps))
#     log_returns = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
#     paths = S0 * np.exp(np.cumsum(log_returns, axis=1))
#     knocked_out = np.any(paths <= B, axis=1)
#     S_T = paths[:, -1]
#     payoff = np.where(knocked_out, 0, np.maximum(S_T - K, 0))
#     return np.exp(-r * T) * np.mean(payoff)


# # ── Implied Volatility Solver ──────────────────────────────────────────────
# def implied_volatility(market_price, S0, K, r, T, option_type="call"):
#     try:
#         f = lambda sigma: black_scholes_price(S0, K, r, sigma, T, option_type) - market_price
#         return brentq(f, 1e-6, 10.0)
#     except Exception:
#         return None