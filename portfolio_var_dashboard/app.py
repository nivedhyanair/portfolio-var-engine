"""
Portfolio VaR Engine - Interactive Quantitative Risk Dashboard
==============================================================
Faithfully implements all 11 phases of VaR_Engine.ipynb:
- Portfolio Page: Tickers, Weights, Portfolio Value, Date Range, Return & Wealth Charts
- VaR Page: Confidence Level, Estimation Window, Model Selector, VaR & ES Outputs, Tail Distributions
- Volatility Scaling Page: EWMA Lambda, GARCH(1,1), GJR-GARCH(1,1,1), Leverage Effect, Volatility Comparison
- Backtesting Page: Actual P&L vs VaR, Exceptions Table, Kupiec POF, Christoffersen Independence & CC, Basel Traffic Light
- Stress Testing Page: Historical & Hypothetical Scenarios, Stress Loss, Stress VaR/ES, Comparative Rankings
"""

import sys
import os
from datetime import date, datetime, timedelta
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add local directory to path for core engine imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import var_engine_core as vec

# ==============================================================================
# STREAMLIT CONFIGURATION & CUSTOM STYLING
# ==============================================================================
st.set_page_config(
    page_title="Portfolio VaR & Risk Engine",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Bloomberg / Quant Terminal aesthetics
st.markdown("""
<style>
    /* Metric Card Styling */
    div[data-testid="stMetric"] {
        background-color: rgba(28, 33, 40, 0.65);
        border: 1px solid rgba(84, 110, 122, 0.25);
        padding: 14px 18px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    div[data-testid="stMetric"]:hover {
        border-color: rgba(66, 165, 245, 0.6);
        transition: 0.2s ease-in-out;
    }
    div[data-testid="stMetricValue"] {
        font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        font-weight: 600;
        font-size: 1.55rem;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #90a4ae;
    }
    /* Section Headers */
    .section-title {
        font-size: 1.25rem;
        font-weight: 600;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
        border-bottom: 2px solid #263238;
        padding-bottom: 4px;
        color: #eceff1;
    }
    /* Badges */
    .badge-green {
        background-color: #1b5e20;
        color: #c8e6c9;
        padding: 4px 10px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-yellow {
        background-color: #f57f17;
        color: #fff9c4;
        padding: 4px 10px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-red {
        background-color: #b71c1c;
        color: #ffcdd2;
        padding: 4px 10px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-blue {
        background-color: #0d47a1;
        color: #bbdefb;
        padding: 4px 10px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# CACHED DATA LOADERS
# ==============================================================================
@st.cache_data(show_spinner=False)
def get_default_prices():
    return vec.load_default_prices()

@st.cache_data(show_spinner=False)
def get_benchmark_prices():
    return vec.load_benchmark_prices()

@st.cache_data(show_spinner=False)
def download_custom_prices(tickers, start, end):
    return vec.fetch_prices(tickers, start, end)

@st.cache_data(show_spinner=False)
def run_cached_backtest(returns_series, est_win, conf, lam):
    return vec.run_oos_backtest(returns_series, estimation_window=est_win, conf_level=conf, lam=lam)

@st.cache_data(show_spinner=False)
def run_cached_stress(stock_prices_df, weights_dict, pvalue, custom_shock):
    return vec.run_stress_testing_suite(stock_prices_df, weights_dict, initial_value=pvalue, custom_shock=custom_shock)

# ==============================================================================
# SIDEBAR - GLOBAL PARAMETERS & NAVIGATION
# ==============================================================================
with st.sidebar:
    st.title("🛡️ VaR Engine")
    st.caption("Quantitative Risk Management & Stress Testing")
    
    # Page Navigation
    page = st.radio(
        "Navigation",
        [
            "Portfolio Page",
            "VaR Page",
            "Volatility Scaling Page",
            "Backtesting Page",
            "Stress Testing Page"
        ],
        index=0
    )
    
    st.divider()
    st.subheader("Global Portfolio Settings")
    
    pvalue = st.number_input(
        "Portfolio Value ($)",
        min_value=1_000.0,
        max_value=1_000_000_000.0,
        value=1_000_000.0,
        step=50_000.0,
        format="%.0f",
        help="Initial investment or current portfolio market value."
    )
    
    # Price Source Selector
    price_source = st.selectbox(
        "Data Source",
        ["VaR Engine Local Dataset (50 Stocks)", "Live Market Data (yfinance)", "Upload CSV File"],
        index=0,
        help="Use pre-cleaned dataset from VaR_Engine.ipynb or fetch live tickers."
    )
    
    if price_source == "Upload CSV File":
        uploaded_file = st.file_uploader("Upload CSV (Date + Asset Prices)", type=["csv"])
    else:
        uploaded_file = None

# ==============================================================================
# DATA INGESTION & PORTFOLIO RESOLUTION
# ==============================================================================
try:
    if price_source == "VaR Engine Local Dataset (50 Stocks)":
        raw_prices = get_default_prices()
        if raw_prices is None:
            st.error("Default price data 'adjusted_close_prices.csv' not found. Please switch to Live Market Data or upload a CSV.")
            st.stop()
    elif price_source == "Upload CSV File":
        if uploaded_file is None:
            st.info("Please upload a CSV file with a Date column and asset prices to proceed.")
            st.stop()
        raw_prices = pd.read_csv(uploaded_file)
        date_col = next((c for c in raw_prices.columns if c.lower() in ("date", "datetime", "timestamp")), None)
        if date_col is None:
            st.error("Uploaded CSV must have a 'Date' column.")
            st.stop()
        raw_prices[date_col] = pd.to_datetime(raw_prices[date_col])
        raw_prices = raw_prices.set_index(date_col).sort_index().apply(pd.to_numeric, errors="coerce")
    else:
        # Live Market Data
        st.sidebar.markdown("**Live Market Configuration**")
        custom_ticker_input = st.sidebar.text_input(
            "Enter Tickers (comma-separated)",
            "AAPL, MSFT, NVDA, GOOGL, AMZN, JPM, JNJ, XOM"
        )
        live_tickers = [t.strip().upper() for t in custom_ticker_input.split(",") if t.strip()]
        d_start = st.sidebar.date_input("Download Start", date(2018, 1, 1))
        d_end = st.sidebar.date_input("Download End", date.today())
        
        with st.spinner("Downloading live market data..."):
            raw_prices = download_custom_prices(live_tickers, d_start, d_end)
except Exception as e:
    st.error(f"Error loading price data: {e}")
    st.stop()

# Available date limits
min_avail_date = raw_prices.index.min().date()
max_avail_date = raw_prices.index.max().date()

# ==============================================================================
# SIDEBAR - TICKER & WEIGHT CONTROLS (IF ON PORTFOLIO PAGE OR AS GLOBAL)
# ==============================================================================
all_available_stocks = list(raw_prices.columns)

# Session state initialization for selected tickers and weights
if "selected_tickers" not in st.session_state:
    st.session_state.selected_tickers = all_available_stocks[:15] if len(all_available_stocks) > 15 else all_available_stocks
if "weight_mode" not in st.session_state:
    st.session_state.weight_mode = "Equal Weight"

# Sidebar Quick Presets
with st.sidebar:
    st.subheader("Asset Selection")
    preset_choice = st.selectbox(
        "Ticker Presets",
        ["Current Selection", "All 50 Stocks (VaR Engine Universe)", "Magnificent 7 Tech", "Major Financials", "Custom Multi-Select"],
        index=0
    )
    
    if preset_choice == "All 50 Stocks (VaR Engine Universe)":
        st.session_state.selected_tickers = [s for s in vec.DEFAULT_TICKERS if s in all_available_stocks]
    elif preset_choice == "Magnificent 7 Tech":
        st.session_state.selected_tickers = [s for s in ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA"] if s in all_available_stocks]
    elif preset_choice == "Major Financials":
        st.session_state.selected_tickers = [s for s in vec.FINANCIAL_STOCKS if s in all_available_stocks]
        
    active_tickers = st.multiselect(
        "Active Tickers",
        options=all_available_stocks,
        default=st.session_state.selected_tickers
    )
    if not active_tickers:
        st.warning("Please select at least one ticker.")
        st.stop()
    st.session_state.selected_tickers = active_tickers
    
    st.subheader("Date Range Selector")
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        sel_start_date = st.date_input("Start Date", min_avail_date, min_value=min_avail_date, max_value=max_avail_date)
    with col_d2:
        sel_end_date = st.date_input("End Date", max_avail_date, min_value=min_avail_date, max_value=max_avail_date)
        
    if sel_start_date >= sel_end_date:
        st.error("Start Date must be before End Date.")
        st.stop()

# Weights calculation
active_prices = raw_prices[active_tickers]
N_stocks = len(active_tickers)

# Weights Configuration in session state
if "weights_dict" not in st.session_state or len(st.session_state.weights_dict) != N_stocks:
    st.session_state.weights_dict = {t: 1.0 / N_stocks for t in active_tickers}

# Compute Portfolio Series
try:
    port_results = vec.compute_portfolio_series(
        active_prices,
        st.session_state.weights_dict,
        initial_value=pvalue,
        start_date=sel_start_date,
        end_date=sel_end_date
    )
except Exception as e:
    st.error(f"Error computing portfolio returns: {e}")
    st.stop()

port_returns = port_results["portfolio_returns"]
port_value_series = port_results["portfolio_value"]
daily_pnl = port_results["daily_pnl"]
rolling_vol_21 = port_results["rolling_vol_21"]
weights_series = port_results["weights"]
clean_prices = port_results["stock_prices"]

# ==============================================================================
# PAGE 1: PORTFOLIO PAGE
# ==============================================================================
if page == "Portfolio Page":
    st.title("💼 Portfolio Overview & Construction")
    st.caption("Asset allocation, indexed portfolio growth, risk-return statistics, and daily return dynamics")
    
    # Portfolio Inputs & Weight Editor Expandable Section
    with st.expander("⚙️ Portfolio Weight Editor & Rebalancing", expanded=False):
        w_col1, w_col2 = st.columns([1, 2])
        with w_col1:
            st.markdown("**Weight Scheme**")
            equal_w_btn = st.button("Set Equal Weights (1/N)", use_container_width=True)
            if equal_w_btn:
                for t in active_tickers:
                    st.session_state.weights_dict[t] = 1.0 / N_stocks
                st.rerun()
                
            normalize_btn = st.button("Normalize to 100%", use_container_width=True)
            if normalize_btn:
                total_w = sum(st.session_state.weights_dict.get(t, 0.0) for t in active_tickers)
                if total_w > 0:
                    for t in active_tickers:
                        st.session_state.weights_dict[t] /= total_w
                st.rerun()
                
        with w_col2:
            st.markdown("**Constituent Allocations**")
            weight_cols = st.columns(min(4, N_stocks))
            for i, ticker in enumerate(active_tickers):
                with weight_cols[i % len(weight_cols)]:
                    curr_val = float(st.session_state.weights_dict.get(ticker, 1.0 / N_stocks))
                    new_val = st.number_input(
                        f"{ticker}",
                        min_value=0.0,
                        max_value=1.0,
                        value=curr_val,
                        step=0.01,
                        format="%.3f",
                        key=f"w_input_{ticker}"
                    )
                    st.session_state.weights_dict[ticker] = new_val

    # Top KPI Summary Cards
    cum_tot_ret = (port_value_series.iloc[-1] / pvalue - 1.0)
    ann_ret = port_returns.mean() * 252
    ann_vol = port_returns.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    
    # Max Drawdown
    peak = port_value_series.cummax()
    drawdown = (port_value_series - peak) / peak
    max_dd = drawdown.min()
    
    kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)
    kpi1.metric("Final Value", f"${port_value_series.iloc[-1]:,.0f}", f"{cum_tot_ret:.2%}")
    kpi2.metric("Total Return", f"{cum_tot_ret:.2%}")
    kpi3.metric("Annualized Return", f"{ann_ret:.2%}")
    kpi4.metric("Annualized Volatility", f"{ann_vol:.2%}")
    kpi5.metric("Sharpe Ratio (Rf=0%)", f"{sharpe:.2f}")
    kpi6.metric("Max Drawdown", f"{max_dd:.2%}")
    
    st.markdown("---")
    
    # Charts Section
    chart_col1, chart_col2 = st.columns([1.6, 1])
    
    with chart_col1:
        st.markdown('<div class="section-title">Portfolio Indexed Value & Cumulative Growth</div>', unsafe_allow_html=True)
        # Indexed Value Chart with SPY Benchmark Comparison if available
        fig_wealth = go.Figure()
        fig_wealth.add_trace(go.Scatter(
            x=port_value_series.index,
            y=port_value_series.values,
            mode="lines",
            name="Portfolio Value",
            line=dict(color="#29b6f6", width=2.2)
        ))
        
        # Benchmark comparison
        bench_df = get_benchmark_prices()
        if bench_df is not None:
            bench_sub = bench_df.loc[(bench_df.index >= port_value_series.index[0]) & (bench_df.index <= port_value_series.index[-1])]
            if not bench_sub.empty and len(bench_sub) > 1:
                bench_ret = np.log(bench_sub / bench_sub.shift(1)).dropna()
                bench_growth = np.exp(bench_ret.cumsum())
                bench_val = pvalue * bench_growth
                fig_wealth.add_trace(go.Scatter(
                    x=bench_val.index,
                    y=bench_val.iloc[:, 0],
                    mode="lines",
                    name="SPY Benchmark",
                    line=dict(color="#ffa726", width=1.5, dash="dash")
                ))
                
        fig_wealth.update_layout(
            template="plotly_dark",
            height=380,
            margin=dict(l=40, r=20, t=20, b=30),
            yaxis=dict(title="Portfolio Value ($)", tickprefix="$", tickformat=","),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_wealth, use_container_width=True)
        
        # Daily Return & Daily P&L Subplots
        st.markdown('<div class="section-title">Daily Return Series & Rolling Volatility</div>', unsafe_allow_html=True)
        fig_ret = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.6, 0.4])
        
        # Bar chart of daily returns
        colors = ["#ef5350" if x < 0 else "#26a69a" for x in port_returns]
        fig_ret.add_trace(go.Bar(
            x=port_returns.index,
            y=port_returns.values,
            marker_color=colors,
            name="Daily Return",
            showlegend=False
        ), row=1, col=1)
        
        # 21-day rolling volatility
        fig_ret.add_trace(go.Scatter(
            x=rolling_vol_21.index,
            y=rolling_vol_21.values,
            line=dict(color="#ab47bc", width=1.8),
            name="21D Vol (Ann)",
            showlegend=True
        ), row=2, col=1)
        
        fig_ret.update_yaxes(title_text="Daily Return", tickformat=".1%", row=1, col=1)
        fig_ret.update_yaxes(title_text="21D Vol (Ann)", tickformat=".1%", row=2, col=1)
        fig_ret.update_layout(
            template="plotly_dark",
            height=340,
            margin=dict(l=40, r=20, t=10, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_ret, use_container_width=True)

    with chart_col2:
        st.markdown('<div class="section-title">Portfolio Asset Allocation</div>', unsafe_allow_html=True)
        alloc_df = weights_series.rename("Weight").reset_index().rename(columns={"index": "Ticker"})
        alloc_df["Dollar Allocation"] = alloc_df["Weight"] * port_value_series.iloc[-1]
        
        fig_pie = px.pie(
            alloc_df,
            names="Ticker",
            values="Weight",
            hole=0.45,
            template="plotly_dark"
        )
        fig_pie.update_traces(textinfo="label+percent", textposition="inside")
        fig_pie.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # Summary Distribution Statistics Table
        st.markdown('<div class="section-title">Return Distribution Statistics</div>', unsafe_allow_html=True)
        stats_data = {
            "Statistic": [
                "Observations (Trading Days)",
                "Daily Mean Return",
                "Daily Std Dev",
                "Annualized Return (252D)",
                "Annualized Volatility",
                "Best Daily Return",
                "Worst Daily Return",
                "Largest Daily Dollar Gain",
                "Largest Daily Dollar Loss",
                "Skewness",
                "Kurtosis (Excess)"
            ],
            "Value": [
                f"{len(port_returns):,}",
                f"{port_returns.mean():.4%}",
                f"{port_returns.std():.4%}",
                f"{ann_ret:.2%}",
                f"{ann_vol:.2%}",
                f"{port_returns.max():.2%}",
                f"{port_returns.min():.2%}",
                f"${daily_pnl.max():,.2f}",
                f"${daily_pnl.min():,.2f}",
                f"{port_returns.skew():.3f}",
                f"{port_returns.kurtosis():.3f}"
            ]
        }
        st.dataframe(pd.DataFrame(stats_data), hide_index=True, use_container_width=True)

# ==============================================================================
# PAGE 2: VaR PAGE
# ==============================================================================
elif page == "VaR Page":
    st.title("📉 Value at Risk (VaR) & Expected Shortfall (ES)")
    st.caption("Parametric and non-parametric downside risk measurement across confidence levels and estimation windows")
    
    # Control Selectors Bar
    ctl_col1, ctl_col2, ctl_col3 = st.columns(3)
    with ctl_col1:
        conf_level = st.selectbox(
            "Confidence Level",
            [0.90, 0.95, 0.975, 0.99, 0.995],
            index=3,
            format_func=lambda x: f"{x:.1%}"
        )
    with ctl_col2:
        hist_window = st.select_slider(
            "Historical Estimation Window (Days)",
            options=[60, 125, 250, 500, 750],
            value=250,
            help="Number of historical observations used for empirical quantile estimation."
        )
    with ctl_col3:
        var_model_choice = st.selectbox(
            "VaR Model Selector",
            ["Historical Simulation", "EWMA Volatility-Scaled", "GARCH(1,1)", "GJR-GARCH(1,1,1)", "Compare All Models"],
            index=0
        )
        
    # Calculate Risk Outputs for all 4 models
    # 1. Historical
    h_res = vec.calc_historical_var_es(port_returns, confidence_level=conf_level, window=hist_window)
    # 2. EWMA
    ew_var, ew_vol = vec.calc_ewma_volatility(port_returns, lam=0.94)
    ew_res = vec.calc_ewma_var_es(port_returns, ew_vol, confidence_level=conf_level)
    
    # 3 & 4. GARCH & GJR-GARCH
    g_res, gj_res = None, None
    g_err, gj_err = None, None
    try:
        gf, g_vol, g_sum = vec.fit_garch_model(port_returns, gjr=False)
        g_res = vec.calc_garch_scaled_var_es(port_returns, gf, g_vol, g_sum["forecast_vol"], confidence_level=conf_level, gjr=False)
    except Exception as e:
        g_err = str(e)
        
    try:
        gjf, gj_vol, gj_sum = vec.fit_garch_model(port_returns, gjr=True)
        gj_res = vec.calc_garch_scaled_var_es(port_returns, gjf, gj_vol, gj_sum["forecast_vol"], confidence_level=conf_level, gjr=True)
    except Exception as e:
        gj_err = str(e)
        
    # Active selected model mapping
    if var_model_choice == "Historical Simulation":
        active_res = h_res
        model_name = "Historical Simulation"
    elif var_model_choice == "EWMA Volatility-Scaled":
        active_res = ew_res
        model_name = "EWMA (λ = 0.94)"
    elif var_model_choice == "GARCH(1,1)":
        active_res = g_res if g_res else h_res
        model_name = "GARCH(1,1)"
    elif var_model_choice == "GJR-GARCH(1,1,1)":
        active_res = gj_res if gj_res else h_res
        model_name = "GJR-GARCH(1,1,1)"
    else:
        active_res = h_res
        model_name = "Historical Simulation"
        
    # Top Risk KPI Metrics
    st.markdown("---")
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    
    var_dollar = active_res["var_loss"] * pvalue
    es_dollar = active_res["es_loss"] * pvalue
    
    m_col1.metric(
        f"{conf_level:.1%} Value at Risk (VaR)",
        f"${var_dollar:,.0f}",
        f"-{active_res['var_loss']:.2%} of portfolio",
        delta_color="inverse"
    )
    m_col2.metric(
        f"{conf_level:.1%} Expected Shortfall (ES)",
        f"${es_dollar:,.0f}",
        f"-{active_res['es_loss']:.2%} of portfolio",
        delta_color="inverse"
    )
    m_col3.metric("Selected Model", model_name, f"Window: {hist_window} days")
    m_col4.metric("Tail Ratio (ES / VaR)", f"{(active_res['es_loss'] / active_res['var_loss']):.2f}x", "Tail thickness")
    
    # Interpretation Banner
    st.info(f"💡 **Risk Interpretation**: Over a 1-day horizon, there is a **{1-conf_level:.1%}** probability that portfolio loss exceeds **${var_dollar:,.0f}** ({active_res['var_loss']:.2%}). If this threshold is breached, the average expected loss is **${es_dollar:,.0f}** ({active_res['es_loss']:.2%}).")
    
    # Model Comparison Table & Bar Chart
    comp_rows = [
        {"Model": "Historical Simulation", "VaR (%)": h_res["var_loss"], "ES (%)": h_res["es_loss"], "VaR ($)": h_res["var_loss"] * pvalue, "ES ($)": h_res["es_loss"] * pvalue},
        {"Model": "EWMA Volatility-Scaled", "VaR (%)": ew_res["var_loss"], "ES (%)": ew_res["es_loss"], "VaR ($)": ew_res["var_loss"] * pvalue, "ES ($)": ew_res["es_loss"] * pvalue}
    ]
    if g_res:
        comp_rows.append({"Model": "GARCH(1,1)", "VaR (%)": g_res["var_loss"], "ES (%)": g_res["es_loss"], "VaR ($)": g_res["var_loss"] * pvalue, "ES ($)": g_res["es_loss"] * pvalue})
    if gj_res:
        comp_rows.append({"Model": "GJR-GARCH(1,1,1)", "VaR (%)": gj_res["var_loss"], "ES (%)": gj_res["es_loss"], "VaR ($)": gj_res["var_loss"] * pvalue, "ES ($)": gj_res["es_loss"] * pvalue})
        
    comp_df = pd.DataFrame(comp_rows)
    
    st.markdown('<div class="section-title">Comparative VaR & Expected Shortfall Across Models</div>', unsafe_allow_html=True)
    comp_c1, comp_c2 = st.columns([1.5, 1])
    with comp_c1:
        fig_bar = px.bar(
            comp_df,
            x="Model",
            y=["VaR ($)", "ES ($)"],
            barmode="group",
            template="plotly_dark",
            color_discrete_sequence=["#e57373", "#b71c1c"],
            title=f"VaR vs Expected Shortfall by Model at {conf_level:.1%} Confidence"
        )
        fig_bar.update_layout(height=340, yaxis=dict(title="Loss Magnitude ($)", tickprefix="$", tickformat=","))
        st.plotly_chart(fig_bar, use_container_width=True)
    with comp_c2:
        st.markdown("**Model Risk Comparison Table**")
        styled_df = comp_df.copy()
        styled_df["VaR (%)"] = styled_df["VaR (%)"].apply(lambda x: f"{x:.2%}")
        styled_df["ES (%)"] = styled_df["ES (%)"].apply(lambda x: f"{x:.2%}")
        styled_df["VaR ($)"] = styled_df["VaR ($)"].apply(lambda x: f"${x:,.0f}")
        styled_df["ES ($)"] = styled_df["ES ($)"].apply(lambda x: f"${x:,.0f}")
        st.dataframe(styled_df, hide_index=True, use_container_width=True)
        st.caption("Expected Shortfall (CVaR) satisfies subadditivity and is a coherent risk measure, unlike standard VaR.")

    # Distribution Histogram with VaR & ES Thresholds
    st.markdown('<div class="section-title">Return Distribution with Critical Downside Thresholds</div>', unsafe_allow_html=True)
    sample_r = port_returns.iloc[-hist_window:] if len(port_returns) >= hist_window else port_returns
    
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(
        x=sample_r,
        nbinsx=65,
        name="Daily Return Sample",
        marker_color="#455a64",
        opacity=0.75
    ))
    
    fig_dist.add_vline(
        x=-active_res["var_loss"],
        line_width=2.5,
        line_dash="dash",
        line_color="#ff7043",
        annotation_text=f"VaR ({conf_level:.1%}): -{active_res['var_loss']:.2%}",
        annotation_position="top left"
    )
    fig_dist.add_vline(
        x=-active_res["es_loss"],
        line_width=2.5,
        line_dash="dot",
        line_color="#d32f2f",
        annotation_text=f"ES ({conf_level:.1%}): -{active_res['es_loss']:.2%}",
        annotation_position="bottom left"
    )
    
    fig_dist.update_layout(
        template="plotly_dark",
        height=360,
        xaxis=dict(title="Portfolio Return", tickformat=".1%"),
        yaxis=dict(title="Observation Frequency"),
        margin=dict(l=40, r=20, t=20, b=30)
    )
    st.plotly_chart(fig_dist, use_container_width=True)

    # Rolling Historical Simulation VaR Series (Phase 4 Cell 50 & 57)
    st.markdown('<div class="section-title">Multi-Window Rolling Historical VaR Evolution</div>', unsafe_allow_html=True)
    rolling_var_df = vec.calc_rolling_historical_var(port_returns, windows=[250, 500, 750], conf_levels=[0.95, 0.99])
    if not rolling_var_df.dropna(how="all").empty:
        fig_roll_var = px.line(
            rolling_var_df,
            template="plotly_dark",
            title="Rolling VaR Time Series (250D, 500D, 750D Windows)"
        )
        fig_roll_var.update_layout(height=340, yaxis=dict(title="VaR Return Threshold", tickformat=".1%"))
        st.plotly_chart(fig_roll_var, use_container_width=True)

# ==============================================================================
# PAGE 3: VOLATILITY SCALING PAGE
# ==============================================================================
elif page == "Volatility Scaling Page":
    st.title("⚡ Volatility Scaling & Clustering Analysis")
    st.caption("EWMA, GARCH(1,1), and GJR-GARCH(1,1,1) asymmetric leverage modeling")
    
    # EWMA Lambda Input Bar
    v_col1, v_col2 = st.columns([1, 2])
    with v_col1:
        ewma_lam = st.slider(
            "EWMA Decay Factor (λ)",
            min_value=0.80,
            max_value=0.99,
            value=0.94,
            step=0.01,
            help="J.P. Morgan RiskMetrics standard λ = 0.94 for daily returns."
        )
    with v_col2:
        st.markdown(rf"**EWMA Formula**: $\sigma_t^2 = {ewma_lam:.2f} \sigma_{{t-1}}^2 + {(1-ewma_lam):.2f} r_{{t-1}}^2$. Higher $\lambda$ gives longer memory to historical volatility.")
        
    # Fit Models
    ew_var, ew_vol = vec.calc_ewma_volatility(port_returns, lam=ewma_lam)
    
    with st.spinner("Fitting GARCH(1,1) and GJR-GARCH(1,1,1) econometric models..."):
        gf, g_vol, g_sum = vec.fit_garch_model(port_returns, gjr=False)
        gjf, gj_vol, gj_sum = vec.fit_garch_model(port_returns, gjr=True)
        leverage_df = vec.analyze_leverage_effect(gjf, gj_vol)
        
    # Current Volatility Metric Cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Latest EWMA Volatility", f"{ew_vol.iloc[-1]:.2%}", f"Ann: {ew_vol.iloc[-1]*np.sqrt(252):.2%}")
    c2.metric("Latest GARCH(1,1) Vol", f"{g_sum['current_vol']:.2%}", f"1D Forecast: {g_sum['forecast_vol']:.2%}")
    c3.metric("Latest GJR-GARCH Vol", f"{gj_sum['current_vol']:.2%}", f"1D Forecast: {gj_sum['forecast_vol']:.2%}")
    c4.metric("GJR Leverage γ (Gamma)", f"{gj_sum['gamma']:.4f}", "Asymmetric Volatility Effect")
    
    st.markdown("---")
    
    # Econometric Parameter Outputs
    st.markdown('<div class="section-title">Econometric Model Estimation Outputs</div>', unsafe_allow_html=True)
    m_out1, m_out2 = st.columns(2)
    
    with m_out1:
        st.subheader("GARCH(1,1) Model Output")
        garch_params_table = pd.DataFrame({
            "Parameter": ["Constant Mean (μ)", "Baseline Variance (ω)", "ARCH Shock (α)", "GARCH Persistence (β)", "Total Persistence (α+β)", "Shock Half-Life", "Log-Likelihood", "AIC", "BIC"],
            "Estimate": [
                f"{g_sum['mu']:.6f}",
                f"{g_sum['omega']:.6f}",
                f"{g_sum['alpha']:.4f}",
                f"{g_sum['beta']:.4f}",
                f"{g_sum['persistence']:.4f}",
                f"{g_sum['half_life']:.1f} days" if np.isfinite(g_sum['half_life']) else "N/A",
                f"{g_sum['log_likelihood']:.1f}",
                f"{g_sum['aic']:.1f}",
                f"{g_sum['bic']:.1f}"
            ]
        })
        st.dataframe(garch_params_table, hide_index=True, use_container_width=True)
        st.caption("Standard GARCH captures volatility clustering: large shocks are followed by large shocks of either sign.")
        
    with m_out2:
        st.subheader("GJR-GARCH(1,1,1) Model Output")
        gjr_params_table = pd.DataFrame({
            "Parameter": ["Constant Mean (μ)", "Baseline Variance (ω)", "ARCH Shock (α)", "Leverage Asymmetry (γ)", "GARCH Persistence (β)", "Total Persistence (α+β+0.5γ)", "Log-Likelihood", "AIC", "BIC"],
            "Estimate": [
                f"{gj_sum['mu']:.6f}",
                f"{gj_sum['omega']:.6f}",
                f"{gj_sum['alpha']:.4f}",
                f"{gj_sum['gamma']:.4f}",
                f"{gj_sum['beta']:.4f}",
                f"{gj_sum['persistence']:.4f}",
                f"{gj_sum['log_likelihood']:.1f}",
                f"{gj_sum['aic']:.1f}",
                f"{gj_sum['bic']:.1f}"
            ]
        })
        st.dataframe(gjr_params_table, hide_index=True, use_container_width=True)
        st.caption("GJR-GARCH incorporates the leverage effect: γ > 0 proves negative shocks generate higher subsequent volatility than positive shocks.")

    # Volatility Comparison Chart
    st.markdown('<div class="section-title">Conditional Volatility Comparison Time Series</div>', unsafe_allow_html=True)
    vol_comp_df = pd.DataFrame({
        "EWMA Volatility": ew_vol,
        "GARCH(1,1)": g_vol,
        "GJR-GARCH(1,1,1)": gj_vol,
        "21D Rolling Hist Vol": port_returns.rolling(21).std()
    }).dropna()
    
    fig_vol = px.line(
        vol_comp_df,
        template="plotly_dark",
        title="Conditional Volatility Comparison Across Models (Daily Scale)"
    )
    fig_vol.update_layout(height=400, yaxis=dict(title="Daily Conditional Volatility", tickformat=".2%"))
    st.plotly_chart(fig_vol, use_container_width=True)
    
    # Asymmetric Leverage Effect Empirical Chart (Phase 7 Cell 121-122)
    if leverage_df is not None:
        st.markdown('<div class="section-title">Empirical Leverage Effect: Negative vs Positive Shocks</div>', unsafe_allow_html=True)
        lev_c1, lev_c2 = st.columns([1.5, 1])
        with lev_c1:
            fig_lev = go.Figure()
            fig_lev.add_trace(go.Scatter(
                x=[str(x) for x in leverage_df.index],
                y=leverage_df["Negative Shock"].values,
                mode="lines+markers",
                name="Negative Shock (Down Days)",
                line=dict(color="#ef5350", width=2.5),
                marker=dict(size=8)
            ))
            fig_lev.add_trace(go.Scatter(
                x=[str(x) for x in leverage_df.index],
                y=leverage_df["Positive Shock"].values,
                mode="lines+markers",
                name="Positive Shock (Up Days)",
                line=dict(color="#26a69a", width=2.5),
                marker=dict(size=8)
            ))
            fig_lev.update_layout(
                template="plotly_dark",
                height=320,
                title="Next-Day Volatility by Shock Magnitude Quantile",
                xaxis=dict(title="Shock Magnitude (Quantile Group)"),
                yaxis=dict(title="Next-Day Conditional Volatility", tickformat=".2%")
            )
            st.plotly_chart(fig_lev, use_container_width=True)
        with lev_c2:
            st.markdown("**Empirical Leverage Data Table**")
            st.dataframe(leverage_df.style.format("{:.2%}"), use_container_width=True)
            st.caption("Confirms that severe negative return shocks produce substantially higher forward conditional volatility than equivalent positive shocks.")

# ==============================================================================
# PAGE 4: BACKTESTING PAGE
# ==============================================================================
elif page == "Backtesting Page":
    st.title("🔬 Out-of-Sample VaR Backtesting & Regulatory Verification")
    st.caption("Rolling 1-step ahead exceptions, Kupiec POF, Christoffersen independence & conditional coverage, and Basel Traffic Light framework")
    
    # Backtest Configuration Bar
    b_col1, b_col2, b_col3 = st.columns(3)
    with b_col1:
        bt_conf = st.selectbox(
            "Backtest Confidence Level",
            [0.95, 0.99, 0.995],
            index=1,
            format_func=lambda x: f"{x:.1%}"
        )
    with b_col2:
        bt_est_win = st.selectbox(
            "Estimation Window (Days)",
            [250, 500, 750],
            index=2,
            help="Observations reserved for in-sample initial estimation. Phase 8 default is 750 days."
        )
    with b_col3:
        bt_model_view = st.selectbox(
            "Display Focus Model",
            ["Historical", "EWMA", "GARCH", "GJR-GARCH", "Overlay All Models"],
            index=0
        )
        
    # Run OOS Backtest Engine
    with st.spinner("Executing out-of-sample rolling recursive backtests for all 4 models..."):
        try:
            oos_df, summary_df = run_cached_backtest(port_returns, bt_est_win, bt_conf, 0.94)
        except Exception as e:
            st.error(f"Error executing backtest: {e}")
            st.stop()
            
    # Top Level Backtest KPI Summary Cards
    target_p = 1.0 - bt_conf
    oos_len = len(oos_df)
    exp_ex = oos_len * target_p
    
    # Active model stats
    active_m_key = "Historical" if bt_model_view in ("Historical", "Overlay All Models") else bt_model_view
    active_row = summary_df[summary_df["Model"] == active_m_key].iloc[0]
    
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("OOS Observations", f"{oos_len:,}", f"Estimation: {bt_est_win}D")
    k2.metric(f"Observed Exceptions ({active_m_key})", f"{active_row['Exceptions']}", f"Expected: {exp_ex:.1f}")
    k3.metric("Exception Rate", f"{active_row['Exception Rate']:.2%}", f"Target: {target_p:.2%}")
    k4.metric("Kupiec POF Result", active_row["Kupiec Decision"], f"p-val: {active_row['Kupiec p-value']:.4f}")
    
    # Basel Zone Badge
    b_zone = active_row["Basel Zone"]
    k5.metric("Basel Traffic Light", f"{b_zone} Zone", f"Multiplier: {active_row['Basel Multiplier']:.2f}x")
    
    st.markdown("---")
    
    # 1. Actual Return / P&L vs VaR Chart (Phase 8 Cell 156)
    st.markdown('<div class="section-title">Actual Realized Returns vs Rolling VaR Thresholds</div>', unsafe_allow_html=True)
    
    fig_bt = go.Figure()
    # Realized returns line
    fig_bt.add_trace(go.Scatter(
        x=oos_df.index,
        y=oos_df["Actual Return"],
        mode="lines",
        name="Realized Return",
        line=dict(color="#90a4ae", width=0.8),
        opacity=0.85
    ))
    
    # Model Thresholds
    colors_map = {
        "Historical": "#42a5f5",
        "EWMA": "#ffa726",
        "GARCH": "#ab47bc",
        "GJR-GARCH": "#26a69a"
    }
    
    if bt_model_view == "Overlay All Models":
        models_to_plot = ["Historical", "EWMA", "GARCH", "GJR-GARCH"]
    else:
        models_to_plot = [bt_model_view]
        
    for m in models_to_plot:
        fig_bt.add_trace(go.Scatter(
            x=oos_df.index,
            y=oos_df[f"{m} VaR"],
            mode="lines",
            name=f"{m} VaR Threshold",
            line=dict(color=colors_map[m], width=1.5, dash="dash" if m != "GJR-GARCH" else "solid")
        ))
        
        # Exceptions Scatter Marker
        ex_dates = oos_df[oos_df[f"{m} Exception"] == 1]
        fig_bt.add_trace(go.Scatter(
            x=ex_dates.index,
            y=ex_dates["Actual Return"],
            mode="markers",
            name=f"{m} Exceptions ({len(ex_dates)})",
            marker=dict(symbol="x", size=8, color=colors_map[m], line=dict(width=1.5))
        ))
        
    fig_bt.add_hline(y=0, line_dash="solid", line_color="#37474f", line_width=0.8)
    fig_bt.update_layout(
        template="plotly_dark",
        height=420,
        margin=dict(l=40, r=20, t=20, b=30),
        xaxis=dict(title="Date"),
        yaxis=dict(title="Daily Return / VaR", tickformat=".1%"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_bt, use_container_width=True)

    # 2. Statistical Backtesting Summary Table
    st.markdown('<div class="section-title">Comprehensive Statistical Backtesting Scorecard</div>', unsafe_allow_html=True)
    scorecard_df = summary_df[[
        "Model", "Observations", "Exceptions", "Expected", "Exception Rate",
        "Kupiec LR", "Kupiec p-value", "Kupiec Decision",
        "Christoffersen LR Ind", "Christoffersen p-value", "Independence Decision",
        "Conditional Coverage LR", "Conditional Coverage p-value", "CC Decision",
        "Basel Zone", "Basel Multiplier"
    ]].copy()
    
    # Format table numbers
    scorecard_df["Expected"] = scorecard_df["Expected"].apply(lambda x: f"{x:.1f}")
    scorecard_df["Exception Rate"] = scorecard_df["Exception Rate"].apply(lambda x: f"{x:.2%}")
    scorecard_df["Kupiec LR"] = scorecard_df["Kupiec LR"].apply(lambda x: f"{x:.3f}")
    scorecard_df["Kupiec p-value"] = scorecard_df["Kupiec p-value"].apply(lambda x: f"{x:.4f}")
    scorecard_df["Christoffersen LR Ind"] = scorecard_df["Christoffersen LR Ind"].apply(lambda x: f"{x:.3f}")
    scorecard_df["Christoffersen p-value"] = scorecard_df["Christoffersen p-value"].apply(lambda x: f"{x:.4f}")
    scorecard_df["Conditional Coverage LR"] = scorecard_df["Conditional Coverage LR"].apply(lambda x: f"{x:.3f}")
    scorecard_df["Conditional Coverage p-value"] = scorecard_df["Conditional Coverage p-value"].apply(lambda x: f"{x:.4f}")
    
    st.dataframe(scorecard_df, hide_index=True, use_container_width=True)

    # 3. Basel Traffic Light Visual & Explanation
    st.markdown('<div class="section-title">Basel Committee Traffic Light Framework (Final 250 Trading Days)</div>', unsafe_allow_html=True)
    b_c1, b_c2 = st.columns([1.5, 1])
    
    with b_c1:
        basel_chart_df = summary_df[["Model", "Basel Exceptions (250D)"]].copy()
        fig_basel = px.bar(
            basel_chart_df,
            x="Model",
            y="Basel Exceptions (250D)",
            template="plotly_dark",
            title="Basel 99% VaR Exceptions (Standard 250-Day Backtest Window)"
        )
        fig_basel.add_hline(y=4.5, line_dash="dash", line_color="#4caf50", annotation_text="Green/Yellow Cutoff (4)")
        fig_basel.add_hline(y=9.5, line_dash="dash", line_color="#f44336", annotation_text="Yellow/Red Cutoff (9)")
        fig_basel.update_layout(height=300)
        st.plotly_chart(fig_basel, use_container_width=True)
        
    with b_c2:
        st.markdown("""
        **Regulatory Basel Zones & Capital Multipliers:**
        - 🟢 **Green Zone (0–4 Exceptions)**: Model is accepted by regulators. Base capital multiplier is **3.00x**.
        - 🟡 **Yellow Zone (5–9 Exceptions)**: Supervisory warning. Mandatory capital add-on scaling penalty (+0.40 to +0.85).
        - 🔴 **Red Zone (10+ Exceptions)**: Automatic model failure. Model must be replaced immediately. Capital multiplier rises to **4.00x+**.
        """)
        
    # 4. Exceptions Table
    st.markdown('<div class="section-title">Exceptions Audit Table</div>', unsafe_allow_html=True)
    sel_ex_model = st.selectbox("Select Model for Exceptions Audit", ["Historical", "EWMA", "GARCH", "GJR-GARCH"], index=0)
    
    ex_filter = oos_df[oos_df[f"{sel_ex_model} Exception"] == 1].copy()
    if not ex_filter.empty:
        audit_table = pd.DataFrame({
            "Date": ex_filter.index.strftime("%Y-%m-%d"),
            "Actual Realized Return": ex_filter["Actual Return"].apply(lambda x: f"{x:.3%}"),
            "Daily Dollar P&L": (ex_filter["Actual Return"] * pvalue).apply(lambda x: f"${x:,.0f}"),
            "Model VaR Threshold": ex_filter[f"{sel_ex_model} VaR"].apply(lambda x: f"{x:.3%}"),
            "Model VaR ($)": (ex_filter[f"{sel_ex_model} VaR"].abs() * pvalue).apply(lambda x: f"${x:,.0f}"),
            "Excess Loss Above VaR": ((ex_filter["Actual Return"] - ex_filter[f"{sel_ex_model} VaR"]) * pvalue).apply(lambda x: f"${abs(x):,.0f}")
        })
        st.dataframe(audit_table, hide_index=True, use_container_width=True)
        csv_bytes = audit_table.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Exceptions CSV", csv_bytes, f"var_exceptions_{sel_ex_model.lower()}.csv", "text/csv")
    else:
        st.success(f"No exceptions recorded for {sel_ex_model} at this confidence level!")

# ==============================================================================
# PAGE 5: STRESS TESTING PAGE
# ==============================================================================
elif page == "Stress Testing Page":
    st.title("⚡ Stress Testing & Crisis Scenario Analysis")
    st.caption("Historical market crashes, hypothetical market shocks, volatility doubling, correlation spikes, and sector-specific dislocations")
    
    # Stress Testing Controls
    sc_col1, sc_col2 = st.columns([1.5, 1])
    with sc_col1:
        scenario_choice = st.selectbox(
            "Scenario Selector",
            [
                "2008 Financial Crisis",
                "COVID Crash 2020",
                "Inflation Shock 2022",
                "Tech Selloff 2022",
                "Banking Stress 2023",
                "Equity Market Down 5%",
                "Equity Market Down 10%",
                "Equity Market Down 20%",
                "Volatility Doubles",
                "Correlation Spike",
                "Financial Sector Down 20%",
                "Custom Uniform Shock"
            ],
            index=1
        )
    with sc_col2:
        custom_shock_val = st.slider(
            "Custom Shock Magnitude (if selected)",
            min_value=-0.50,
            max_value=0.10,
            value=-0.15,
            step=0.01,
            format="%.0f%%"
        )
        
    # Run Full Stress Testing Suite
    with st.spinner("Executing stress testing suite across all scenarios..."):
        stress_master, base_stress_info = run_cached_stress(clean_prices, st.session_state.weights_dict, pvalue, custom_shock_val)
        
    # Extract Active Scenario Row
    if scenario_choice == "Custom Uniform Shock":
        sc_lookup = f"Custom Shock ({custom_shock_val:.1%})"
        active_sc_row = stress_master[stress_master["Scenario"].str.startswith("Custom Shock")].iloc[0]
    else:
        active_sc_row = stress_master[stress_master["Scenario"] == scenario_choice].iloc[0]
        
    # Active Scenario Metrics
    st.markdown("---")
    s_m1, s_m2, s_m3, s_m4 = st.columns(4)
    
    sc_loss = active_sc_row["Stress Loss"]
    sc_loss_dollar = active_sc_row["Dollar Loss"]
    sc_var = active_sc_row["95% Stressed VaR"]
    sc_es = active_sc_row["95% Stressed ES"]
    sc_worst = active_sc_row["Worst-Case Loss"]
    
    if pd.notna(sc_loss):
        s_m1.metric("Cumulative Stress Loss", f"${sc_loss_dollar:,.0f}", f"{sc_loss:.2%}", delta_color="inverse")
    else:
        s_m1.metric("Cumulative Stress Loss", "N/A", "Statistical Shift")
        
    s_m2.metric("95% Stressed VaR", f"${abs(sc_var)*pvalue:,.0f}", f"{sc_var:.2%}", delta_color="inverse")
    s_m3.metric("95% Stressed Expected Shortfall", f"${abs(sc_es)*pvalue:,.0f}", f"{sc_es:.2%}", delta_color="inverse")
    s_m4.metric("Worst Single-Day Loss", f"${abs(sc_worst)*pvalue:,.0f}", f"{sc_worst:.2%}", delta_color="inverse")
    
    # Baseline Comparison Notice
    base_var_dl = abs(base_stress_info["baseline_var_95"]) * pvalue
    diff_pct = (abs(sc_var) - abs(base_stress_info["baseline_var_95"])) / abs(base_stress_info["baseline_var_95"])
    st.info(f"📊 **Baseline Normal Risk vs Stress**: Baseline 95% VaR is **${base_var_dl:,.0f}** ({base_stress_info['baseline_var_95']:.2%}). Under **{scenario_choice}**, downside risk expands by **{diff_pct:+.1%}**.")
    
    # Scenario Comparison Charts (Phase 10 Cells 215-217)
    st.markdown('<div class="section-title">Scenario Comparison Charts across All Stress Tests</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["📉 Cumulative Stress Loss", "🛡️ 95% Stressed VaR", "🔥 Stressed Expected Shortfall & Worst Loss"])
    
    with tab1:
        loss_chart_df = stress_master[stress_master["Stress Loss"].notna()].sort_values("Stress Loss")
        fig_sc_loss = px.bar(
            loss_chart_df,
            x="Stress Loss",
            y="Scenario",
            orientation="h",
            color="Scenario Type",
            color_discrete_map={"Historical": "#ef5350", "Hypothetical": "#ab47bc"},
            template="plotly_dark",
            title="Cumulative Portfolio Loss by Stress Scenario"
        )
        fig_sc_loss.update_layout(height=380, xaxis=dict(tickformat=".1%"))
        st.plotly_chart(fig_sc_loss, use_container_width=True)
        
    with tab2:
        var_chart_df = stress_master.sort_values("95% Stressed VaR")
        fig_sc_var = px.bar(
            var_chart_df,
            x="95% Stressed VaR",
            y="Scenario",
            orientation="h",
            color="Scenario Type",
            color_discrete_map={"Historical": "#42a5f5", "Hypothetical": "#26a69a"},
            template="plotly_dark",
            title="95% Stressed Value at Risk (Daily Loss Threshold)"
        )
        fig_sc_var.update_layout(height=400, xaxis=dict(tickformat=".1%"))
        st.plotly_chart(fig_sc_var, use_container_width=True)
        
    with tab3:
        es_chart_df = stress_master.sort_values("95% Stressed ES")
        fig_sc_es = px.bar(
            es_chart_df,
            x="95% Stressed ES",
            y="Scenario",
            orientation="h",
            color="Worst-Case Loss",
            color_continuous_scale="Reds",
            template="plotly_dark",
            title="95% Stressed Expected Shortfall & Worst-Case Loss Ranking"
        )
        fig_sc_es.update_layout(height=400, xaxis=dict(tickformat=".1%"))
        st.plotly_chart(fig_sc_es, use_container_width=True)

    # Master Stress Summary Table (Phase 10 Cell 212-214)
    st.markdown('<div class="section-title">Master Stress Testing Results Table (Phase 10 Specification)</div>', unsafe_allow_html=True)
    display_master = stress_master.copy()
    display_master["Stress Loss"] = display_master["Stress Loss"].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")
    display_master["95% Stressed VaR"] = display_master["95% Stressed VaR"].apply(lambda x: f"{x:.2%}")
    display_master["95% Stressed ES"] = display_master["95% Stressed ES"].apply(lambda x: f"{x:.2%}")
    display_master["Worst-Case Loss"] = display_master["Worst-Case Loss"].apply(lambda x: f"{x:.2%}")
    display_master["Dollar Impact ($)"] = display_master["Dollar Loss"].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A")
    
    st.dataframe(display_master[[
        "ES Rank", "Scenario", "Scenario Type", "Stocks Used", "Stress Loss", "95% Stressed VaR", "95% Stressed ES", "Worst-Case Loss", "Dollar Impact ($)"
    ]], hide_index=True, use_container_width=True)

# Footer
st.divider()
st.caption("Quantitative Risk Engine & Interactive Dashboard | Based on methodology in VaR_Engine.ipynb | Created for Quant Finance Portfolio Risk Management")
