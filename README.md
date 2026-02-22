# 📊 Options Decoded · Real-Time Options Intelligence

A professional-grade options analytics platform that combines real-time market data, quantitative pricing models, and intuitive risk visualization to deliver actionable volatility insights.

Built for traders, quant enthusiasts, and FinTech engineers.

---

## 🚀 Overview

Options Decoded is a live options intelligence system that integrates:

- Real-time option chain data
- Black–Scholes theoretical pricing
- Monte Carlo simulation (25,000+ paths)
- Implied vs Historical Volatility analysis
- Greeks-based risk analytics
- Volatility smile visualization
- Institutional-style mispricing detection

This is not a toy pricing calculator — it is a structured volatility analysis engine.

---

## 🧠 Core Features

### 1️⃣ Live Market Integration
- Real-time stock and option chain data via Yahoo Finance
- Automatic extraction of Implied Volatility
- Dynamic strike & expiry selection

### 2️⃣ Theoretical Pricing Engine
Implements the Black–Scholes model to compute:
- Fair option value
- Delta, Gamma, Vega, Theta, Rho
- Intrinsic vs Time value breakdown

### 3️⃣ Monte Carlo Simulation Engine
- Geometric Brownian Motion simulation
- 25,000+ simulated price paths
- Probability of finishing ITM
- Distribution-based pricing estimate
- Expected payoff visualization

### 4️⃣ Volatility Intelligence
- Historical Volatility (HV) computation
- Implied Volatility (IV) extraction
- IV vs HV mispricing detection
- Volatility regime signal
- Volatility smile plotting

### 5️⃣ Trader-Friendly Insights Mode
- Plain-English explanations
- Risk warnings (Theta decay, leverage risk)
- Actionable volatility signals
- Educational overlays for beginners

---

## 📈 Why This Project Is Different

Most academic option projects:
- Use static inputs
- Ignore real market data
- Skip volatility structure analysis

This platform:
- Pulls live chain data
- Compares model vs market pricing
- Surfaces volatility misalignment
- Visualizes risk exposures clearly

It bridges the gap between:
Retail trading tools and institutional analytics.

---

## 🏗 Architecture

Frontend:
- Streamlit interactive dashboard

Data Layer:
- Yahoo Finance API

Pricing Layer:
- Black–Scholes closed-form solution
- Monte Carlo engine (NumPy vectorized simulation)

Analytics Layer:
- Volatility comparison engine
- Greek sensitivity calculator
- Mispricing detector

---

## 📊 Models Implemented

### Black–Scholes Model
Used for theoretical fair value pricing under:
- Constant volatility
- Log-normal returns
- No arbitrage assumptions

### Monte Carlo Simulation
Used for:
- Probabilistic payoff estimation
- Non-deterministic scenario modeling
- Validation against analytical pricing

---

## 🧪 Example Use Cases

- Detect overpriced options when IV >> HV
- Estimate probability of profit before buying options
- Evaluate Theta decay impact for short-dated contracts
- Analyze volatility smile across strikes
- Compare market premium vs theoretical fair value

---

## ⚙ Installation

Clone the repository:

```bash
git clone https://github.com/vansh007/options-decoded.git
cd options-decoded
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run app.py
```

---

## 📌 Dependencies

- streamlit
- numpy
- pandas
- matplotlib
- scipy
- yfinance

---


## 🎯 Future Improvements

- Portfolio-level Greeks aggregation
- IV Rank & IV Percentile tracking
- Liquidity score (bid-ask spread analysis)
- Async data fetching
- Docker containerization
- Backend API modularization
- Stochastic volatility model integration

---

## 👨‍💻 Author

Vansh Mundhra  
B.Tech Computer Science  

Focused on:
- Quantitative Finance
- Financial Engineering
- FinTech Systems

---

## 📜 Disclaimer

This project is for educational and research purposes only.  
It is not financial advice.

---

## ⭐ If You Found This Interesting

Consider starring the repository.