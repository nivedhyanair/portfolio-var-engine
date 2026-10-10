"""
var_engine_core.py - Core Quantitative Engine for Portfolio VaR & Stress Testing
Replicates the quantitative methodology of VaR_Engine.ipynb:
- Phase 1 & 2: Data ingestion, cleaning, log-returns
- Phase 3: Portfolio construction, wealth indexing, daily P&L
- Phase 4: Historical Simulation VaR & Expected Shortfall (rolling windows)
- Phase 5: EWMA volatility scaling (RiskMetrics lambda)
- Phase 6: GARCH(1,1) volatility scaling & 1-day forecast
- Phase 7: GJR-GARCH(1,1,1) asymmetric leverage effect
- Phase 8: Out-of-sample backtesting (Kupiec POF, Christoffersen independence & CC, Basel Traffic Light)
- Phase 10: Historical & hypothetical stress testing suite
"""

import os
import warnings
from datetime import date, datetime, timedelta
import numpy as np
import pandas as pd
from scipy.stats import chi2, norm

try:
    import yfinance as yf
except ImportError:
    yf = None

try:
    from arch import arch_model
except ImportError:
    arch_model = None

# Default tickers from VaR_Engine.ipynb Phase 1
DEFAULT_TICKERS = [
    "AMZN", "LIN", "CVX", "UNH", "MCD", "AMT", "DIS", "ECL", "CRM", "ISRG",
    "TMUS", "HON", "PEP", "FCX", "UPS", "ABBV", "MSFT", "DE", "PG", "WFC",
    "BAC", "ORCL", "JNJ", "KO", "COST", "XOM", "DUK", "HD", "BA", "JPM",
    "GOOGL", "AAPL", "BKNG", "PLD", "TSLA", "CAT", "GS", "AMD", "MS", "META",
    "COP", "LLY", "MRK", "BLK", "NVDA", "AVGO", "SHW", "NEE", "NFLX", "GE"
]

FINANCIAL_STOCKS = ["JPM", "BAC", "WFC", "C", "GS", "MS", "BLK"]
TECH_STOCKS = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AMD", "AVGO", "CRM"]

def find_file(filename):
    """Find file in current dir, parent dir, or workspace."""
    search_paths = [
        filename,
        os.path.join("..", filename),
        os.path.join(os.path.dirname(__file__), filename),
        os.path.join(os.path.dirname(__file__), "..", filename),
        os.path.join(r"c:\Users\nived\Desktop\My Files\Quant Finance (QFI) Internship\My Work", filename)
    ]
    for p in search_paths:
        if os.path.exists(p):
            return p
    return None

def load_default_prices():
    """Load pre-processed adjusted closing prices from CSV if available."""
    p = find_file("adjusted_close_prices.csv")
    if p and os.path.exists(p):
        df = pd.read_csv(p, index_col=0, parse_dates=True)
        return df
    return None

def load_benchmark_prices():
    """Load benchmark SPY prices if available."""
    p = find_file("benchmark_prices.csv")
    if p and os.path.exists(p):
        df = pd.read_csv(p, index_col=0, parse_dates=True)
        return df
    return None

def load_2008_crisis_prices():
    """Load 2008 crisis prices if available."""
    p = find_file("crisis_2008_prices.csv")
    if p and os.path.exists(p):
        df = pd.read_csv(p, index_col=0, parse_dates=True)
        return df
    return None

def fetch_prices(tickers, start_date, end_date):
    """Download prices via yfinance for custom tickers or live dates."""
    if yf is None:
        raise ValueError("yfinance package is not installed.")
    tickers = [t.strip().upper() for t in tickers if t.strip()]
    if not tickers:
        raise ValueError("Please provide at least one valid ticker.")
    
    data = yf.download(
        tickers,
        start=str(start_date),
        end=str(end_date),
        auto_adjust=True,
        progress=False,
        threads=True
    )
    if data is None or data.empty:
        raise ValueError(f"No price data returned for {tickers}. Check ticker symbols or date range.")
    
    if isinstance(data.columns, pd.MultiIndex):
        if "Close" in data.columns.get_level_values(0):
            prices = data["Close"].copy()
        elif "Close" in data.columns.get_level_values(-1):
            prices = data.xs("Close", axis=1, level=-1).copy()
        else:
            raise ValueError("Could not extract Close prices.")
    else:
        prices = data[["Close"]].rename(columns={"Close": tickers[0]})
        
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])
        
    prices = prices.dropna(how="all").sort_index()
    return prices

def compute_portfolio_series(prices, weights_dict, initial_value=1_000_000.0, start_date=None, end_date=None):
    """
    Computes stock log returns, weighted daily portfolio return, indexed wealth, and daily P&L.
    Matches Phase 2 & 3 of VaR_Engine.ipynb.
    """
    df_prices = prices.copy()
    if start_date:
        df_prices = df_prices.loc[df_prices.index >= pd.to_datetime(start_date)]
    if end_date:
        df_prices = df_prices.loc[df_prices.index <= pd.to_datetime(end_date)]
        
    # Drop columns that have too many missing values in this window
    df_prices = df_prices.dropna(axis=1, how="all").dropna(how="all")
    if df_prices.empty or len(df_prices) < 20:
        raise ValueError("Insufficient price data in the selected date range.")
        
    # Calculate stock log returns: r_i = ln(P_i,t / P_i,t-1)
    stock_returns = np.log(df_prices / df_prices.shift(1)).dropna(how="all")
    
    # Filter stocks with sufficient valid return data
    valid_stocks = [c for c in stock_returns.columns if stock_returns[c].notna().sum() > 10]
    stock_returns = stock_returns[valid_stocks].dropna(how="any")
    
    if stock_returns.empty or len(stock_returns) < 15:
        raise ValueError("Insufficient continuous return data after cleaning.")
        
    # Align and normalize weights
    w_series = pd.Series({s: weights_dict.get(s, 1.0 / len(valid_stocks)) for s in valid_stocks}, dtype=float)
    if w_series.sum() <= 0:
        w_series = pd.Series(1.0 / len(valid_stocks), index=valid_stocks)
    else:
        w_series = w_series / w_series.sum()
        
    # Daily portfolio log return = dot product: sum(w_i * r_i)
    port_return = stock_returns.dot(w_series)
    port_return.name = "Portfolio Return"
    
    # Cumulative growth and portfolio value
    port_growth = np.exp(port_return.cumsum())
    wealth = initial_value * port_growth
    
    # Prepend initial investment
    initial_row = pd.Series([initial_value], index=[df_prices.index[0]], name="Portfolio Value")
    portfolio_value = pd.concat([initial_row, wealth])
    portfolio_value = portfolio_value[~portfolio_value.index.duplicated(keep="first")].sort_index()
    
    # Daily dollar P&L
    daily_pnl = portfolio_value.diff()
    daily_pnl.iloc[0] = 0.0
    daily_pnl.name = "Daily P&L"
    
    # Rolling 21-day volatility
    rolling_vol_21 = port_return.rolling(window=21).std() * np.sqrt(252)
    rolling_vol_21.name = "21D Rolling Vol (Ann)"
    
    return {
        "stock_prices": df_prices[valid_stocks],
        "stock_returns": stock_returns,
        "weights": w_series,
        "portfolio_returns": port_return,
        "portfolio_value": portfolio_value,
        "daily_pnl": daily_pnl,
        "rolling_vol_21": rolling_vol_21
    }

# ==============================================================================
# PHASE 4: HISTORICAL SIMULATION VaR & EXPECTED SHORTFALL
# ==============================================================================

def calc_historical_var_es(returns, confidence_level=0.99, window=250):
    """
    Calculates Historical Simulation VaR and Expected Shortfall over a given window.
    VaR is reported as both a negative return threshold and a positive loss magnitude.
    """
    s = pd.Series(returns).dropna()
    if len(s) < window:
        sub = s
    else:
        sub = s.iloc[-window:]
        
    alpha = 1.0 - confidence_level
    var_threshold = float(sub.quantile(alpha))
    
    tail = sub[sub <= var_threshold]
    if len(tail) > 0:
        es_threshold = float(tail.mean())
    else:
        es_threshold = var_threshold
        
    var_loss = abs(var_threshold)
    es_loss = abs(es_threshold)
    
    return {
        "var_threshold": var_threshold,
        "es_threshold": es_threshold,
        "var_loss": var_loss,
        "es_loss": es_loss,
        "sample": sub
    }

def calc_rolling_historical_var(returns, windows=[250, 500, 750], conf_levels=[0.95, 0.99, 0.995]):
    """Calculates rolling Historical VaR across multiple windows and confidence levels."""
    s = pd.Series(returns).dropna()
    results = pd.DataFrame(index=s.index)
    
    for w in windows:
        if len(s) >= w:
            for c in conf_levels:
                alpha = 1.0 - c
                col_name = f"{w}D VaR ({c:.1%})"
                results[col_name] = s.rolling(window=w).quantile(alpha)
    return results

# ==============================================================================
# PHASE 5: EWMA VOLATILITY SCALING
# ==============================================================================

def calc_ewma_volatility(returns, lam=0.94):
    """
    Calculates EWMA conditional variance and volatility.
    sigma_t^2 = lambda * sigma_{t-1}^2 + (1 - lambda) * r_{t-1}^2
    Initial variance = sample variance of first 20 observations.
    """
    r = pd.Series(returns).dropna()
    T = len(r)
    if T < 2:
        return pd.Series(index=r.index, dtype=float), pd.Series(index=r.index, dtype=float)
        
    var_arr = np.zeros(T)
    init_window = min(20, T)
    var_arr[0] = float(r.iloc[:init_window].var())
    if var_arr[0] == 0:
        var_arr[0] = 1e-4
        
    for t in range(1, T):
        var_arr[t] = lam * var_arr[t - 1] + (1.0 - lam) * (float(r.iloc[t - 1]) ** 2)
        
    variance = pd.Series(var_arr, index=r.index, name="EWMA Variance")
    volatility = pd.Series(np.sqrt(np.maximum(var_arr, 0.0)), index=r.index, name="EWMA Volatility")
    return variance, volatility

def calc_ewma_scaled_returns(returns, ewma_volatility):
    """
    Scales historical returns by current EWMA volatility:
    r_t* = r_t * (sigma_T / sigma_t)
    """
    r = pd.Series(returns).dropna()
    v = pd.Series(ewma_volatility).reindex(r.index).dropna()
    common_idx = r.index.intersection(v.index)
    r = r.loc[common_idx]
    v = v.loc[common_idx]
    
    current_vol = float(v.iloc[-1])
    safe_v = v.replace(0, np.nan)
    scaled_r = r * (current_vol / safe_v)
    scaled_r.name = "EWMA Scaled Return"
    return scaled_r

def calc_ewma_var_es(returns, ewma_volatility, confidence_level=0.99):
    """Calculates EWMA-scaled VaR and Expected Shortfall."""
    scaled_r = calc_ewma_scaled_returns(returns, ewma_volatility)
    alpha = 1.0 - confidence_level
    var_threshold = float(scaled_r.quantile(alpha))
    tail = scaled_r[scaled_r <= var_threshold]
    es_threshold = float(tail.mean()) if len(tail) > 0 else var_threshold
    
    return {
        "var_threshold": var_threshold,
        "es_threshold": es_threshold,
        "var_loss": abs(var_threshold),
        "es_loss": abs(es_threshold),
        "scaled_returns": scaled_r
    }

# ==============================================================================
# PHASE 6 & 7: GARCH(1,1) & GJR-GARCH(1,1,1)
# ==============================================================================

def fit_garch_model(returns, gjr=False):
    """
    Fits GARCH(1,1) or GJR-GARCH(1,1,1) with Constant Mean.
    Returns model fit object, conditional volatility (in decimal units), and parameters.
    """
    if arch_model is None:
        raise ImportError("The 'arch' package is not installed.")
        
    r = pd.Series(returns).dropna()
    if len(r) < 100:
        raise ValueError("At least 100 return observations are required to reliably fit GARCH models.")
        
    # Convert to percentage returns for numerical stability
    r_pct = r * 100.0
    
    model = arch_model(
        r_pct,
        mean="Constant",
        vol="GARCH",
        p=1,
        o=1 if gjr else 0,
        q=1,
        dist="normal"
    )
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = model.fit(disp="off")
        
    cond_vol_decimal = pd.Series(
        np.asarray(fit.conditional_volatility) / 100.0,
        index=r.index,
        name="GJR-GARCH Volatility" if gjr else "GARCH Volatility"
    )
    
    # 1-step ahead forecast volatility
    forecast = fit.forecast(horizon=1, reindex=True)
    fc_var = float(forecast.variance["h.1"].dropna().iloc[-1])
    fc_vol_decimal = np.sqrt(fc_var) / 100.0
    
    params = fit.params.to_dict()
    summary_dict = {
        "mu": float(params.get("mu", 0.0)),
        "omega": float(params.get("omega", 0.0)),
        "alpha": float(params.get("alpha[1]", 0.0)),
        "beta": float(params.get("beta[1]", 0.0)),
        "gamma": float(params.get("gamma[1]", 0.0)) if gjr else 0.0,
        "log_likelihood": float(fit.loglikelihood),
        "aic": float(fit.aic),
        "bic": float(fit.bic),
        "current_vol": float(cond_vol_decimal.iloc[-1]),
        "forecast_vol": fc_vol_decimal,
        "fit": fit
    }
    
    # Compute shock persistence & half-life
    if gjr:
        persistence = summary_dict["alpha"] + summary_dict["beta"] + 0.5 * summary_dict["gamma"]
    else:
        persistence = summary_dict["alpha"] + summary_dict["beta"]
    summary_dict["persistence"] = persistence
    
    if 0 < persistence < 1:
        summary_dict["half_life"] = np.log(0.5) / np.log(persistence)
    else:
        summary_dict["half_life"] = np.nan
        
    return fit, cond_vol_decimal, summary_dict

def calc_garch_scaled_var_es(returns, fit_result, cond_vol, forecast_vol, confidence_level=0.99, gjr=False):
    """
    Calculates VaR and ES using GARCH/GJR-GARCH volatility scaling.
    r_t* = r_t * (sigma_forecast / sigma_t)
    """
    r = pd.Series(returns).dropna()
    v = pd.Series(cond_vol).reindex(r.index).dropna()
    common_idx = r.index.intersection(v.index)
    r = r.loc[common_idx]
    v = v.loc[common_idx]
    
    safe_v = v.replace(0, np.nan)
    scaled_r = r * (forecast_vol / safe_v)
    scaled_r.name = "GJR-GARCH Scaled Return" if gjr else "GARCH Scaled Return"
    
    alpha = 1.0 - confidence_level
    var_threshold = float(scaled_r.quantile(alpha))
    tail = scaled_r[scaled_r <= var_threshold]
    es_threshold = float(tail.mean()) if len(tail) > 0 else var_threshold
    
    return {
        "var_threshold": var_threshold,
        "es_threshold": es_threshold,
        "var_loss": abs(var_threshold),
        "es_loss": abs(es_threshold),
        "scaled_returns": scaled_r
    }

def analyze_leverage_effect(gjr_fit, cond_vol):
    """
    Analyzes asymmetric leverage effect: reactions of conditional volatility to positive vs negative shocks.
    Matches Cells 119-122 of VaR_Engine.ipynb.
    """
    shocks = gjr_fit.resid
    next_vol = cond_vol.shift(-1)
    
    df = pd.concat([
        shocks.rename("Shock"),
        next_vol.rename("Next Day Volatility")
    ], axis=1).dropna()
    
    df["Absolute Shock"] = df["Shock"].abs()
    df["Shock Type"] = np.where(df["Shock"] < 0, "Negative Shock", "Positive Shock")
    
    try:
        df["Shock Group"] = pd.qcut(df["Absolute Shock"], q=5, duplicates="drop")
        empirical_leverage = df.groupby(["Shock Group", "Shock Type"], observed=True)["Next Day Volatility"].mean().unstack()
        return empirical_leverage
    except Exception:
        return None

# ==============================================================================
# PHASE 8: BACKTESTING SUITE (KUPIEC, CHRISTOFFERSEN, BASEL TRAFFIC LIGHT)
# ==============================================================================

def kupiec_pof_test(exceptions, confidence_level=0.99):
    """
    Kupiec Proportion of Failures (POF) test for unconditional coverage.
    Matches Cell 142 of VaR_Engine.ipynb.
    """
    x_arr = np.asarray(exceptions, dtype=int)
    T = len(x_arr)
    x = int(np.sum(x_arr))
    p = 1.0 - confidence_level
    
    if T == 0:
        return 0, 0.0, 0.0, np.nan, np.nan, "N/A"
        
    p_hat = x / T
    
    if x == 0:
        ll_null = T * np.log(1.0 - p)
        ll_alt = T * np.log(1.0 - p_hat) if p_hat < 1 else 0
    elif x == T:
        ll_null = T * np.log(p)
        ll_alt = T * np.log(p_hat) if p_hat > 0 else 0
    else:
        ll_null = x * np.log(p) + (T - x) * np.log(1.0 - p)
        ll_alt = x * np.log(p_hat) + (T - x) * np.log(1.0 - p_hat)
        
    lr_pof = max(0.0, 2.0 * (ll_alt - ll_null))
    p_value = float(1.0 - chi2.cdf(lr_pof, df=1))
    decision = "Pass" if p_value >= 0.05 else "Fail"
    
    return x, T * p, p_hat, lr_pof, p_value, decision

def christoffersen_independence_test(exceptions):
    """
    Christoffersen Independence Test for first-order Markov exception clustering.
    Matches Cell 145 of VaR_Engine.ipynb.
    """
    x = np.asarray(exceptions, dtype=int)
    if len(x) < 2:
        return np.nan, np.nan, 0, 0, 0, 0, "N/A"
        
    prev = x[:-1]
    curr = x[1:]
    
    n00 = int(np.sum((prev == 0) & (curr == 0)))
    n01 = int(np.sum((prev == 0) & (curr == 1)))
    n10 = int(np.sum((prev == 1) & (curr == 0)))
    n11 = int(np.sum((prev == 1) & (curr == 1)))
    
    n0 = n00 + n01
    n1 = n10 + n11
    
    pi01 = n01 / n0 if n0 > 0 else 0.0
    pi11 = n11 / n1 if n1 > 0 else 0.0
    
    total_ex = n01 + n11
    total_trans = len(x) - 1
    pi = total_ex / total_trans if total_trans > 0 else 0.0
    
    # Log-likelihood under independence
    ll_ind = 0.0
    if total_ex > 0 and 0 < pi < 1:
        ll_ind += total_ex * np.log(pi)
    if (total_trans - total_ex) > 0 and 0 < pi < 1:
        ll_ind += (total_trans - total_ex) * np.log(1.0 - pi)
        
    # Log-likelihood under Markov alternative
    ll_alt = 0.0
    if n01 > 0 and 0 < pi01 < 1:
        ll_alt += n01 * np.log(pi01)
    if n00 > 0 and 0 < pi01 < 1:
        ll_alt += n00 * np.log(1.0 - pi01)
    if n11 > 0 and 0 < pi11 < 1:
        ll_alt += n11 * np.log(pi11)
    if n10 > 0 and 0 < pi11 < 1:
        ll_alt += n10 * np.log(1.0 - pi11)
        
    lr_ind = max(0.0, 2.0 * (ll_alt - ll_ind))
    p_value = float(1.0 - chi2.cdf(lr_ind, df=1))
    decision = "Pass" if p_value >= 0.05 else "Fail"
    
    return lr_ind, p_value, n00, n01, n10, n11, decision

def christoffersen_conditional_coverage_test(lr_pof, lr_ind):
    """
    Christoffersen Conditional Coverage Test combining POF and Independence:
    LR_cc = LR_pof + LR_ind ~ chi2(2)
    Matches Cell 148 of VaR_Engine.ipynb.
    """
    if np.isnan(lr_pof) or np.isnan(lr_ind):
        return np.nan, np.nan, "N/A"
    lr_cc = lr_pof + lr_ind
    p_value = float(1.0 - chi2.cdf(lr_cc, df=2))
    decision = "Pass" if p_value >= 0.05 else "Fail"
    return lr_cc, p_value, decision

def basel_traffic_light(exceptions_count, total_obs=250):
    """
    Basel Committee Traffic Light Framework for 99% VaR.
    Standardized thresholds for 250 observations:
    - Green Zone: 0 - 4 exceptions (Pass, multiplier = 3.00)
    - Yellow Zone: 5 - 9 exceptions (Warning, multiplier = 3.00 + penalty)
    - Red Zone: 10+ exceptions (Fail, model rejected, multiplier = 4.00)
    Matches Cells 150-152 of VaR_Engine.ipynb.
    """
    # Scale thresholds if window differs from 250
    scale = total_obs / 250.0
    green_cutoff = int(np.floor(4 * scale))
    yellow_cutoff = int(np.floor(9 * scale))
    
    if exceptions_count <= green_cutoff:
        zone = "Green"
        status = "Pass (Model Accepted)"
        penalty = 0.0
        multiplier = 3.00
        color = "#28a745"
    elif exceptions_count <= yellow_cutoff:
        zone = "Yellow"
        status = "Warning (Supervisory Capital Add-on)"
        # Standard Basel penalty schedule
        penalty_table = {5: 0.40, 6: 0.50, 7: 0.65, 8: 0.75, 9: 0.85}
        base_ex = min(9, max(5, int(round(exceptions_count / scale))))
        penalty = penalty_table.get(base_ex, 0.50)
        multiplier = 3.00 + penalty
        color = "#ffc107"
    else:
        zone = "Red"
        status = "Fail (Model Rejected)"
        penalty = 1.00
        multiplier = 4.00
        color = "#dc3545"
        
    return {
        "zone": zone,
        "status": status,
        "exceptions": exceptions_count,
        "penalty": penalty,
        "multiplier": multiplier,
        "color": color
    }

def run_oos_backtest(returns, estimation_window=750, conf_level=0.99, lam=0.94):
    """
    Executes Out-of-Sample Rolling VaR Backtesting for all 4 models:
    - Historical Simulation (rolling 250-day window)
    - EWMA Volatility Scaling
    - GARCH(1,1) Volatility Scaling
    - GJR-GARCH(1,1,1) Volatility Scaling
    Matches Cells 126-159 of VaR_Engine.ipynb.
    """
    r = pd.Series(returns).dropna()
    T_total = len(r)
    
    if T_total <= estimation_window + 30:
        raise ValueError(f"Need at least {estimation_window + 30} observations for backtesting. Current sample: {T_total}")
        
    est_returns = r.iloc[:estimation_window]
    oos_returns = r.iloc[estimation_window:]
    T_oos = len(oos_returns)
    alpha = 1.0 - conf_level
    
    # 1. Historical Simulation OOS VaR (rolling 250d)
    hist_var_oos = pd.Series(index=oos_returns.index, dtype=float)
    hist_window = min(250, estimation_window)
    for i, dt in enumerate(oos_returns.index):
        pos = estimation_window + i
        past_window = r.iloc[pos - hist_window : pos]
        hist_var_oos.loc[dt] = float(past_window.quantile(alpha))
        
    # 2. EWMA OOS VaR
    _, full_ewma_vol = calc_ewma_volatility(r, lam=lam)
    est_ewma_vol = full_ewma_vol.iloc[:estimation_window]
    ewma_std_returns = est_returns / est_ewma_vol.replace(0, np.nan)
    ewma_q_std = float(ewma_std_returns.dropna().quantile(alpha))
    
    ewma_var_oos = pd.Series(index=oos_returns.index, dtype=float)
    for dt in oos_returns.index:
        current_vol = float(full_ewma_vol.loc[dt])
        ewma_var_oos.loc[dt] = ewma_q_std * current_vol
        
    # 3. GARCH(1,1) OOS VaR
    garch_var_oos = pd.Series(index=oos_returns.index, dtype=float)
    gjr_var_oos = pd.Series(index=oos_returns.index, dtype=float)
    
    if arch_model is not None and len(est_returns) >= 100:
        try:
            # Fit on estimation period
            garch_est_pct = est_returns * 100.0
            gm = arch_model(garch_est_pct, mean="Constant", vol="GARCH", p=1, q=1, dist="normal")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gf = gm.fit(disp="off")
                
            mu_g = float(gf.params["mu"])
            omega_g = float(gf.params["omega"])
            alpha_g = float(gf.params["alpha[1]"])
            beta_g = float(gf.params["beta[1]"])
            
            # Standardized quantile from estimation period
            g_std = (garch_est_pct - mu_g) / gf.conditional_volatility
            q_std_g = float(g_std.dropna().quantile(alpha))
            
            # Recursive 1-step forecast across OOS
            all_pct = r * 100.0
            prev_var = float(gf.conditional_volatility.iloc[-1] ** 2)
            prev_ret = float(all_pct.iloc[estimation_window - 1])
            
            for dt in oos_returns.index:
                prev_eps = prev_ret - mu_g
                fc_var = omega_g + alpha_g * (prev_eps ** 2) + beta_g * prev_var
                fc_vol = np.sqrt(max(fc_var, 1e-6))
                
                # VaR in decimal
                garch_var_oos.loc[dt] = (mu_g + q_std_g * fc_vol) / 100.0
                
                prev_ret = float(all_pct.loc[dt])
                prev_var = fc_var
        except Exception:
            garch_var_oos = ewma_var_oos.copy()
            
        try:
            # GJR-GARCH on estimation period
            gjr_est_pct = est_returns * 100.0
            gjm = arch_model(gjr_est_pct, mean="Constant", vol="GARCH", p=1, o=1, q=1, dist="normal")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gjf = gjm.fit(disp="off")
                
            mu_gjr = float(gjf.params["mu"])
            omega_gjr = float(gjf.params["omega"])
            alpha_gjr = float(gjf.params["alpha[1]"])
            gamma_gjr = float(gjf.params["gamma[1]"])
            beta_gjr = float(gjf.params["beta[1]"])
            
            gjr_std = (gjr_est_pct - mu_gjr) / gjf.conditional_volatility
            q_std_gjr = float(gjr_std.dropna().quantile(alpha))
            
            prev_var = float(gjf.conditional_volatility.iloc[-1] ** 2)
            prev_ret = float(all_pct.iloc[estimation_window - 1])
            
            for dt in oos_returns.index:
                prev_eps = prev_ret - mu_gjr
                neg_shock = 1 if prev_eps < 0 else 0
                fc_var = omega_gjr + alpha_gjr * (prev_eps ** 2) + gamma_gjr * (prev_eps ** 2) * neg_shock + beta_gjr * prev_var
                fc_vol = np.sqrt(max(fc_var, 1e-6))
                
                gjr_var_oos.loc[dt] = (mu_gjr + q_std_gjr * fc_vol) / 100.0
                
                prev_ret = float(all_pct.loc[dt])
                prev_var = fc_var
        except Exception:
            gjr_var_oos = ewma_var_oos.copy()
    else:
        garch_var_oos = ewma_var_oos.copy()
        gjr_var_oos = ewma_var_oos.copy()
        
    oos_df = pd.DataFrame({
        "Actual Return": oos_returns,
        "Historical VaR": hist_var_oos,
        "EWMA VaR": ewma_var_oos,
        "GARCH VaR": garch_var_oos,
        "GJR-GARCH VaR": gjr_var_oos
    }).dropna()
    
    # Exceptions
    models = ["Historical", "EWMA", "GARCH", "GJR-GARCH"]
    for m in models:
        oos_df[f"{m} Exception"] = (oos_df["Actual Return"] < oos_df[f"{m} VaR"]).astype(int)
        
    # Statistical backtest summary
    summary_rows = []
    # Most recent 250-day window for Basel standard test
    basel_window_df = oos_df.iloc[-250:] if len(oos_df) >= 250 else oos_df
    
    for m in models:
        ex_series = oos_df[f"{m} Exception"]
        ex_cnt, exp_cnt, ex_ratio, lr_pof, p_pof, dec_pof = kupiec_pof_test(ex_series, conf_level)
        lr_ind, p_ind, n00, n01, n10, n11, dec_ind = christoffersen_independence_test(ex_series)
        lr_cc, p_cc, dec_cc = christoffersen_conditional_coverage_test(lr_pof, lr_ind)
        
        # Basel traffic light on final 250 observations
        basel_ex = int(basel_window_df[f"{m} Exception"].sum())
        basel_info = basel_traffic_light(basel_ex, total_obs=len(basel_window_df))
        
        summary_rows.append({
            "Model": m,
            "Observations": len(oos_df),
            "Exceptions": ex_cnt,
            "Expected": exp_cnt,
            "Exception Rate": ex_ratio,
            "Target Rate": 1.0 - conf_level,
            "Kupiec LR": lr_pof,
            "Kupiec p-value": p_pof,
            "Kupiec Decision": dec_pof,
            "Christoffersen LR Ind": lr_ind,
            "Christoffersen p-value": p_ind,
            "Independence Decision": dec_ind,
            "Conditional Coverage LR": lr_cc,
            "Conditional Coverage p-value": p_cc,
            "CC Decision": dec_cc,
            "Basel Exceptions (250D)": basel_ex,
            "Basel Zone": basel_info["zone"],
            "Basel Multiplier": basel_info["multiplier"],
            "Basel Decision": basel_info["status"]
        })
        
    summary_df = pd.DataFrame(summary_rows)
    return oos_df, summary_df

# ==============================================================================
# PHASE 10: STRESS TESTING SUITE
# ==============================================================================

HISTORICAL_SCENARIOS = {
    "2008 Financial Crisis": {"start": "2008-09-15", "end": "2008-11-20", "desc": "Global Financial Crisis & Lehman Bankruptcy"},
    "COVID Crash 2020": {"start": "2020-02-19", "end": "2020-03-23", "desc": "Rapid market crash triggered by pandemic lockdowns"},
    "Inflation Shock 2022": {"start": "2022-01-03", "end": "2022-10-12", "desc": "Fed aggressive tightening & high inflation drawdown"},
    "Tech Selloff 2022": {"start": "2022-01-03", "end": "2022-06-16", "desc": "Severe correction in growth and tech valuations"},
    "Banking Stress 2023": {"start": "2023-03-08", "end": "2023-03-20", "desc": "Regional bank crisis (SVB, Signature, Credit Suisse)"}
}

def run_stress_testing_suite(portfolio_prices, weights_dict, initial_value=1_000_000.0, custom_shock=-0.10):
    """
    Executes comprehensive Historical and Hypothetical Stress Testing.
    Matches Cells 171-219 of VaR_Engine.ipynb:
    1. Historical: 2008 Crisis, COVID 2020, Inflation 2022, Tech Selloff 2022, Banking Stress 2023
    2. Hypothetical Equity: Down 5%, Down 10%, Down 20%
    3. Volatility Doubles: 2.0x standard deviation expansion
    4. Correlation Spike: 75% convergence toward +1.0
    5. Financial Sector Down 20%: specific shock to financial stocks
    6. Custom Uniform Shock
    """
    # Stock-level log returns
    stock_log_r = np.log(portfolio_prices / portfolio_prices.shift(1)).dropna(how="all")
    w = pd.Series({c: weights_dict.get(c, 1.0 / len(portfolio_prices.columns)) for c in portfolio_prices.columns}, dtype=float)
    w = w / w.sum()
    
    # Portfolio simple returns
    stock_simple_r = np.exp(stock_log_r) - 1.0
    baseline_port_simple = stock_simple_r.dot(w).dropna()
    baseline_var_95 = float(baseline_port_simple.quantile(0.05))
    baseline_es_95 = float(baseline_port_simple[baseline_port_simple <= baseline_var_95].mean())
    baseline_worst = float(baseline_port_simple.min())
    
    results = []
    
    # ---------------------------------------------------------
    # A. Historical Scenarios
    # ---------------------------------------------------------
    # 1. 2008 Crisis: check if crisis prices are available
    crisis_prices = load_2008_crisis_prices()
    if crisis_prices is not None and not crisis_prices.empty:
        c_log = np.log(crisis_prices / crisis_prices.shift(1))
        c_win = c_log.loc[(c_log.index >= "2008-09-15") & (c_log.index <= "2008-11-20")]
        avail_stocks = c_win.columns[c_win.notna().all()]
        if len(avail_stocks) > 0:
            c_simple = (np.exp(c_win[avail_stocks]) - 1.0).mean(axis=1)
            cum_loss = float((1.0 + c_simple).prod() - 1.0)
            worst_loss = float(c_simple.min())
            var_95 = float(c_simple.quantile(0.05))
            es_95 = float(c_simple[c_simple <= var_95].mean()) if len(c_simple[c_simple <= var_95]) > 0 else var_95
            
            results.append({
                "Scenario": "2008 Financial Crisis",
                "Scenario Type": "Historical",
                "Stocks Used": len(avail_stocks),
                "Stress Loss": cum_loss,
                "95% Stressed VaR": var_95,
                "95% Stressed ES": es_95,
                "Worst-Case Loss": worst_loss,
                "Dollar Loss": abs(cum_loss) * initial_value
            })
            
    # Other historical scenarios from active dataset
    for sc_name, sc_info in HISTORICAL_SCENARIOS.items():
        if sc_name == "2008 Financial Crisis":
            continue
        sc_data = stock_log_r.loc[(stock_log_r.index >= sc_info["start"]) & (stock_log_r.index <= sc_info["end"])].dropna(how="all")
        if not sc_data.empty and len(sc_data) >= 5:
            avail_stocks = sc_data.columns[sc_data.notna().all()]
            if len(avail_stocks) > 0:
                sc_w = w.reindex(avail_stocks).fillna(0)
                if sc_w.sum() > 0:
                    sc_w = sc_w / sc_w.sum()
                sc_simple = (np.exp(sc_data[avail_stocks]) - 1.0).dot(sc_w)
                cum_loss = float((1.0 + sc_simple).prod() - 1.0)
                worst_loss = float(sc_simple.min())
                var_95 = float(sc_simple.quantile(0.05))
                tail = sc_simple[sc_simple <= var_95]
                es_95 = float(tail.mean()) if len(tail) > 0 else var_95
                
                results.append({
                    "Scenario": sc_name,
                    "Scenario Type": "Historical",
                    "Stocks Used": len(avail_stocks),
                    "Stress Loss": cum_loss,
                    "95% Stressed VaR": var_95,
                    "95% Stressed ES": es_95,
                    "Worst-Case Loss": worst_loss,
                    "Dollar Loss": abs(cum_loss) * initial_value
                })
                
    # ---------------------------------------------------------
    # B. Hypothetical Broad Market Shocks (-5%, -10%, -20%)
    # ---------------------------------------------------------
    equity_shocks = {
        "Equity Market Down 5%": -0.05,
        "Equity Market Down 10%": -0.10,
        "Equity Market Down 20%": -0.20
    }
    for sc_name, shock in equity_shocks.items():
        log_shock = np.log1p(shock)
        stressed_log = stock_log_r + log_shock
        stressed_simple = (np.exp(stressed_log) - 1.0).dot(w)
        var_95 = float(stressed_simple.quantile(0.05))
        es_95 = float(stressed_simple[stressed_simple <= var_95].mean())
        worst_loss = float(stressed_simple.min())
        
        results.append({
            "Scenario": sc_name,
            "Scenario Type": "Hypothetical",
            "Stocks Used": len(w),
            "Stress Loss": shock,
            "95% Stressed VaR": var_95,
            "95% Stressed ES": es_95,
            "Worst-Case Loss": worst_loss,
            "Dollar Loss": abs(shock) * initial_value
        })
        
    # ---------------------------------------------------------
    # C. Volatility Doubling Stress (Factor = 2.0)
    # Matches Cell 190 of VaR_Engine.ipynb
    # ---------------------------------------------------------
    port_log_r = np.log1p(baseline_port_simple)
    hist_mean_log = float(port_log_r.mean())
    vol_stressed_log = hist_mean_log + 2.0 * (port_log_r - hist_mean_log)
    vol_stressed_simple = np.expm1(vol_stressed_log)
    vol_var_95 = float(vol_stressed_simple.quantile(0.05))
    vol_es_95 = float(vol_stressed_simple[vol_stressed_simple <= vol_var_95].mean())
    vol_worst = float(vol_stressed_simple.min())
    
    results.append({
        "Scenario": "Volatility Doubles",
        "Scenario Type": "Hypothetical",
        "Stocks Used": len(w),
        "Stress Loss": np.nan,
        "95% Stressed VaR": vol_var_95,
        "95% Stressed ES": vol_es_95,
        "Worst-Case Loss": vol_worst,
        "Dollar Loss": abs(vol_var_95) * initial_value
    })
    
    # ---------------------------------------------------------
    # D. Correlation Spike Stress (75% to +1.0)
    # Matches Cells 191-198 of VaR_Engine.ipynb
    # ---------------------------------------------------------
    try:
        clean_stock_log = stock_log_r.dropna()
        stock_means = clean_stock_log.mean().values
        stock_vols = clean_stock_log.std().values
        curr_corr = clean_stock_log.corr().values
        
        stressed_corr = curr_corr + 0.75 * (1.0 - curr_corr)
        np.fill_diagonal(stressed_corr, 1.0)
        
        stressed_cov = np.outer(stock_vols, stock_vols) * stressed_corr
        # Ensure positive semi-definite
        eigenvals, eigenvecs = np.linalg.eigh(stressed_cov)
        eigenvals = np.maximum(eigenvals, 1e-8)
        stressed_cov_psd = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        
        rng = np.random.default_rng(42)
        sim_stock_r = rng.multivariate_normal(stock_means, stressed_cov_psd, size=20000)
        sim_stock_simple = np.expm1(sim_stock_r)
        corr_stressed_port = sim_stock_simple.dot(w.values)
        
        corr_var_95 = float(np.percentile(corr_stressed_port, 5))
        corr_es_95 = float(corr_stressed_port[corr_stressed_port <= corr_var_95].mean())
        corr_worst = float(corr_stressed_port.min())
        
        results.append({
            "Scenario": "Correlation Spike",
            "Scenario Type": "Hypothetical",
            "Stocks Used": len(w),
            "Stress Loss": np.nan,
            "95% Stressed VaR": corr_var_95,
            "95% Stressed ES": corr_es_95,
            "Worst-Case Loss": corr_worst,
            "Dollar Loss": abs(corr_var_95) * initial_value
        })
    except Exception:
        pass
        
    # ---------------------------------------------------------
    # E. Financial Sector Down 20%
    # Matches Cells 199-206 of VaR_Engine.ipynb
    # ---------------------------------------------------------
    avail_fin = [s for s in FINANCIAL_STOCKS if s in stock_log_r.columns]
    if len(avail_fin) > 0:
        fin_w_sum = float(w.reindex(avail_fin).fillna(0).sum())
        direct_fin_impact = -0.20 * fin_w_sum
        
        sec_stressed_log = stock_log_r.copy()
        sec_shock_log = np.log1p(-0.20)
        sec_stressed_log[avail_fin] = sec_stressed_log[avail_fin] + sec_shock_log
        sec_stressed_simple = (np.exp(sec_stressed_log) - 1.0).dot(w)
        
        sec_var_95 = float(sec_stressed_simple.quantile(0.05))
        sec_es_95 = float(sec_stressed_simple[sec_stressed_simple <= sec_var_95].mean())
        sec_worst = float(sec_stressed_simple.min())
        
        results.append({
            "Scenario": "Financial Sector Down 20%",
            "Scenario Type": "Hypothetical",
            "Stocks Used": len(w),
            "Stress Loss": direct_fin_impact,
            "95% Stressed VaR": sec_var_95,
            "95% Stressed ES": sec_es_95,
            "Worst-Case Loss": sec_worst,
            "Dollar Loss": abs(direct_fin_impact) * initial_value
        })
        
    # ---------------------------------------------------------
    # F. Custom Uniform Return Shock
    # ---------------------------------------------------------
    if custom_shock is not None and custom_shock != 0:
        c_log_shock = np.log1p(custom_shock)
        cust_stressed_log = stock_log_r + c_log_shock
        cust_stressed_simple = (np.exp(cust_stressed_log) - 1.0).dot(w)
        c_var_95 = float(cust_stressed_simple.quantile(0.05))
        c_es_95 = float(cust_stressed_simple[cust_stressed_simple <= c_var_95].mean())
        c_worst = float(cust_stressed_simple.min())
        
        results.append({
            "Scenario": f"Custom Shock ({custom_shock:.1%})",
            "Scenario Type": "Hypothetical",
            "Stocks Used": len(w),
            "Stress Loss": custom_shock,
            "95% Stressed VaR": c_var_95,
            "95% Stressed ES": c_es_95,
            "Worst-Case Loss": c_worst,
            "Dollar Loss": abs(custom_shock) * initial_value
        })
        
    master_df = pd.DataFrame(results)
    # Ranking by Stressed Expected Shortfall (most severe first)
    if not master_df.empty and "95% Stressed ES" in master_df.columns:
        master_df["ES Rank"] = master_df["95% Stressed ES"].rank(ascending=True, method="min").astype(int)
        master_df = master_df.sort_values("ES Rank").reset_index(drop=True)
        
    return master_df, {
        "baseline_var_95": baseline_var_95,
        "baseline_es_95": baseline_es_95,
        "baseline_worst": baseline_worst
    }
