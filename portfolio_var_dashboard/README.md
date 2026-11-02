# Interactive Portfolio VaR & Risk Engine Dashboard

An institutional-grade quantitative risk dashboard built with Streamlit and Plotly, directly implementing and extending the methodologies from `VaR_Engine.ipynb`.

## 🚀 Key Features by Page

### 1. Portfolio Page
- **Ticker Input**: Select from the 50-stock S&P large-cap universe, preloaded subsets (All 50 Stocks, Magnificent 7 Tech, Major Financials), or input custom tickers.
- **Weight Input**: Easily rebalance with equal weighting (1/N) or customize individual constituent weights with auto-normalization.
- **Portfolio Value Input**: Live dollar-rebased portfolio evaluation (default: $1,000,000).
- **Date Range Selector**: Flexible historical start and end date controls.
- **Portfolio Return Chart**: Interactive Plotly visualizations for cumulative portfolio growth, SPY benchmark comparison, daily return bar charts, 21-day rolling volatility, allocation donut chart, and complete return distribution statistics.

### 2. VaR Page
- **Confidence Level Selector**: Toggle between 90%, 95%, 97.5%, 99%, and 99.5% confidence levels.
- **Historical Window Selector**: Select from 60, 125, 250, 500, or 750 trading days.
- **VaR Model Selector**: Switch between:
  - Historical Simulation VaR
  - EWMA Volatility-Scaled VaR
  - GARCH(1,1) Volatility-Scaled VaR
  - GJR-GARCH(1,1,1) Volatility-Scaled VaR
  - Multi-Model Comparison View
- **VaR Output**: Loss threshold in percentage (%) and absolute dollar amount ($).
- **Expected Shortfall Output**: Coherent tail risk / CVaR in percentage (%) and dollar amount ($).
- **Distribution Charts**: Return histogram with VaR & ES vertical cutoffs and multi-window rolling VaR time series.

### 3. Volatility Scaling Page
- **EWMA Lambda Input**: Fine-tune decay parameter $\lambda$ from 0.80 to 0.99 (default: 0.94 RiskMetrics).
- **GARCH(1,1) Model Output**: Constant mean $\mu$, baseline variance $\omega$, ARCH shock $\alpha$, GARCH persistence $\beta$, total persistence, and volatility shock half-life.
- **GJR-GARCH(1,1,1) Model Output**: Leverage parameter $\gamma$ capturing asymmetric response to negative market shocks.
- **Leverage Effect Visualizer**: Empirical analysis of next-day volatility reaction for negative vs positive shocks by quantile magnitude.
- **Volatility Comparison Chart**: Side-by-side time series comparison of EWMA, GARCH, GJR-GARCH, and 21-day rolling historical volatility.

### 4. Backtesting Page
- **Actual P&L vs VaR**: Time series overlay of realized daily returns / dollar P&L against rolling 1-step ahead VaR thresholds with highlighted exception breaches.
- **Exceptions Table**: Searchable, filterable audit log of every VaR breach date, actual return, dollar P&L, threshold, and excess loss magnitude (with CSV export).
- **Kupiec POF Test Result**: Likelihood Ratio ($LR_{POF}$) statistic, p-value, and Pass/Fail conclusion at 5% significance level.
- **Christoffersen Independence & Conditional Coverage**: Tests for first-order Markov exception clustering ($LR_{ind}$) and joint coverage ($LR_{cc} \sim \chi^2(2)$).
- **Basel Committee Traffic Light Framework**: Standardized 250-day window evaluation classifying models into Green (0–4), Yellow (5–9), or Red (10+) zones with supervisory capital multiplier schedule.

### 5. Stress Testing Page
- **Scenario Selector**:
  - **Historical Scenarios**: 2008 Financial Crisis, COVID Crash 2020, Inflation Shock 2022, Tech Selloff 2022, Banking Stress 2023.
  - **Hypothetical Scenarios**: Equity Market Down 5%, 10%, 20%, Volatility Doubling (2.0x factor), Correlation Spike (75% convergence toward +1.0), Financial Sector Shock (-20%), and Custom Uniform Return Shock slider.
- **Stress Loss Output**: Cumulative scenario loss (%) and dollar impact ($).
- **Stress VaR & ES Output**: 95% Stressed VaR and Stressed Expected Shortfall under stressed conditions.
- **Scenario Comparison Charts**: Ranked horizontal bar charts comparing Cumulative Loss, Stressed VaR, Stressed ES, and Worst-Case single-day loss across all scenarios.

---

## 🛠️ How to Run

From the root project directory:

```bash
streamlit run app.py
```

Or from inside `portfolio_var_dashboard`:

```bash
cd portfolio_var_dashboard
streamlit run app.py
```

### Dependencies
- Python 3.10+
- `streamlit>=1.35`
- `plotly>=5.20`
- `pandas>=2.0`
- `numpy>=1.24`
- `scipy>=1.10`
- `arch>=6.3`
- `yfinance>=0.2.40`
