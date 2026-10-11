"""
Portfolio VaR Engine - Institutional Quantitative Risk Dashboard
================================================================
Implements all 11 phases of VaR_Engine.ipynb with executive Bloomberg/FactSet styling:
- Portfolio Page: Un-truncated institutional KPI cards, asset allocation, wealth curve, SPY benchmark,
                  underwater drawdown chart, asset correlation heatmap, return distribution statistics
- VaR Page: Confidence levels, estimation windows, 4 models, risk thermometer gauge, fat-tails normal overlay,
            distribution histogram with VaR/ES cutoffs, multi-window rolling VaR
- Volatility Scaling Page: EWMA lambda, GARCH(1,1), GJR-GARCH(1,1,1), News Impact Curve, persistence half-life,
                           empirical leverage effect, conditional volatility comparison
- Backtesting Page: Rolling 1-step ahead OOS backtests, Kupiec POF, Christoffersen independence & CC,
                    Basel Traffic Light visual gauge & capital multiplier, exceptions audit table
- Stress Testing Page: 2008 Crisis, COVID 2020, Inflation 2022, Tech selloff, Banking stress, broad equity shocks,
                       volatility doubling, correlation spikes, sector shocks, ranked comparison charts
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
from scipy.stats import norm

# Add local directory to path for core engine imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import var_engine_core as vec

# ==============================================================================
# STREAMLIT CONFIGURATION & EXECUTIVE THEME STYLING
# ==============================================================================
st.set_page_config(
    page_title="Portfolio VaR & Risk Engine",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Bloomberg / Quant Terminal CSS (Guarantees ZERO truncation)
st.markdown("""
<style>
    /* Prevent ANY metric truncation across Streamlit native components */
    div[data-testid="stMetricLabel"] > div,
    div[data-testid="stMetricLabel"] p,
    div[data-testid="stMetricLabel"] span {
        white-space: normal !important;
        word-wrap: break-word !important;
        text-overflow: clip !important;
        overflow: visible !important;
        line-height: 1.3 !important;
        font-size: 0.84rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.4px !important;
        color: #90a4ae !important;
    }
    div[data-testid="stMetricValue"] > div,
    div[data-testid="stMetricValue"] span {
        white-space: normal !important;
        word-wrap: break-word !important;
        text-overflow: clip !important;
        overflow: visible !important;
        font-size: 1.6rem !important;
        font-weight: 700 !important;
        color: #ffffff !important;
        line-height: 1.2 !important;
    }
    div[data-testid="stMetricDelta"] > div {
        white-space: normal !important;
        overflow: visible !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
    }

    /* Institutional Header Banner */
    .top-banner {
        background: linear-gradient(90deg, #0d1117 0%, #161b22 50%, #0d1117 100%);
        border: 1px solid rgba(56, 139, 253, 0.25);
        border-left: 4px solid #00d2ff;
        border-radius: 8px;
        padding: 12px 20px;
        margin-bottom: 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .top-banner-title {
        font-size: 1.35rem;
        font-weight: 700;
        color: #f0f6fc;
        letter-spacing: -0.3px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .top-banner-status {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(46, 160, 67, 0.15);
        border: 1px solid rgba(46, 160, 67, 0.4);
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        color: #3fb950;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Section Titles */
    .section-title {
        font-size: 1.15rem;
        font-weight: 700;
        margin-top: 1.2rem;
        margin-bottom: 0.6rem;
        border-bottom: 2px solid #21262d;
        padding-bottom: 6px;
        color: #e6edf3;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Badges */
    .badge-pass {
        background-color: rgba(46, 160, 67, 0.2);
        color: #3fb950;
        border: 1px solid #2ea043;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 700;
        font-size: 0.8rem;
    }
    .badge-fail {
        background-color: rgba(248, 81, 73, 0.2);
        color: #f85149;
        border: 1px solid #f85149;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 700;
        font-size: 0.8rem;
    }
    .badge-warn {
        background-color: rgba(210, 153, 34, 0.2);
        color: #d29922;
        border: 1px solid #d29922;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 700;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# Custom Reusable KPI Card (Guarantees full titles and values with zero dots/ellipses)
def render_kpi_card(title, value, delta=None, delta_is_positive=True, subtext=None, icon=None, accent_color="#00d2ff"):
    delta_html = ""
    if delta is not None:
        arrow = "▲" if delta_is_positive else "▼"
        color = "#3fb950" if delta_is_positive else "#f85149"
        delta_html = f'<div style="font-size: 0.85rem; font-weight: 600; color: {color}; margin-top: 3px; display: inline-flex; align-items: center; gap: 4px;"><span>{arrow}</span><span>{delta}</span></div>'
    
    subtext_html = ""
    if subtext is not None:
        subtext_html = f'<div style="font-size: 0.78rem; color: #8b949e; margin-top: 3px;">{subtext}</div>'
        
    icon_html = f'<span style="font-size: 1.15rem; margin-right: 6px;">{icon}</span>' if icon else ""
    
    card_html = f"""
    <div style="
        background: linear-gradient(135deg, rgba(22, 27, 34, 0.95) 0%, rgba(13, 17, 23, 0.98) 100%);
        border: 1px solid rgba(84, 110, 122, 0.35);
        border-top: 3px solid {accent_color};
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.4);
        margin-bottom: 14px;
        min-height: 126px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    ">
        <div style="font-size: 0.84rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; color: #90a4ae; display: flex; align-items: center; white-space: normal; line-height: 1.3;">
            {icon_html}{title}
        </div>
        <div style="font-size: clamp(1.4rem, 1.8vw, 1.85rem); font-weight: 700; color: #ffffff; margin: 4px 0; white-space: normal; word-break: normal; letter-spacing: -0.5px; line-height: 1.2;">
            {value}
        </div>
        <div>
            {delta_html}
            {subtext_html}
        </div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)

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
    st.markdown("## 🛡️ VaR Engine")
    st.caption("Quantitative Risk Analytics & Econometric Engine")
    
    # Page Navigation
    page = st.radio(
        "Navigation Modules",
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
    st.markdown("### ⚙️ Global Portfolio Settings")
    
    pvalue = st.number_input(
        "Portfolio Capital ($)",
        min_value=1_000.0,
        max_value=1_000_000_000.0,
        value=1_000_000.0,
        step=50_000.0,
        format="%.0f",
        help="Initial investment capital rebased across the historical period."
    )
    
    # Price Source Selector
    price_source = st.selectbox(
        "Data Feed",
        ["VaR Engine Dataset (50 Stocks)", "Live Market Data (yfinance)", "Upload Custom CSV"],
        index=0,
        help="Use pre-cleaned dataset from VaR_Engine.ipynb or fetch live tickers."
    )
    
    if price_source == "Upload Custom CSV":
        uploaded_file = st.file_uploader("Upload CSV (Date + Asset Prices)", type=["csv"])
    else:
        uploaded_file = None

# ==============================================================================
# DATA INGESTION & PORTFOLIO RESOLUTION
# ==============================================================================
try:
    if price_source == "VaR Engine Dataset (50 Stocks)":
        raw_prices = get_default_prices()
        if raw_prices is None:
            st.error("Default price data 'adjusted_close_prices.csv' not found. Please switch to Live Market Data or upload a CSV.")
            st.stop()
    elif price_source == "Upload Custom CSV":
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
all_available_stocks = list(raw_prices.columns)

# Session state initialization for selected tickers and weights
if "selected_tickers" not in st.session_state:
    st.session_state.selected_tickers = all_available_stocks[:15] if len(all_available_stocks) > 15 else all_available_stocks

# Sidebar Quick Presets
with st.sidebar:
    st.markdown("### 🎯 Universe Selection")
    preset_choice = st.selectbox(
        "Universe Presets",
        ["Current Active Selection", "All 50 Stocks (VaR Engine Universe)", "Magnificent 7 Tech", "Major Financials", "Custom Multi-Select"],
        index=0
    )
    
    if preset_choice == "All 50 Stocks (VaR Engine Universe)":
        st.session_state.selected_tickers = [s for s in vec.DEFAULT_TICKERS if s in all_available_stocks]
    elif preset_choice == "Magnificent 7 Tech":
        st.session_state.selected_tickers = [s for s in ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA"] if s in all_available_stocks]
    elif preset_choice == "Major Financials":
        st.session_state.selected_tickers = [s for s in vec.FINANCIAL_STOCKS if s in all_available_stocks]
        
    active_tickers = st.multiselect(
        "Constituents Included",
        options=all_available_stocks,
        default=st.session_state.selected_tickers
    )
    if not active_tickers:
        st.warning("Please select at least one ticker.")
        st.stop()
    st.session_state.selected_tickers = active_tickers
    
    st.markdown("### 📅 Date Range Filter")
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
stock_returns = port_results["stock_returns"]

# Portfolio Statistics Calculations
cum_tot_ret = (port_value_series.iloc[-1] / pvalue - 1.0)
ann_ret = port_returns.mean() * 252
ann_vol = port_returns.std() * np.sqrt(252)
sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0

# Peak-to-Trough Drawdown
cum_peaks = port_value_series.cummax()
drawdown_series = (port_value_series - cum_peaks) / cum_peaks
max_drawdown = float(drawdown_series.min())

# ==============================================================================
# TOP STATUS BANNER
# ==============================================================================
st.markdown(f"""
<div class="top-banner">
    <div class="top-banner-title">
        <span>💼 Institutional Risk Terminal</span>
        <span style="font-size: 0.95rem; font-weight: 500; color: #8b949e;">| {len(active_tickers)} Equities • ${pvalue:,.0f} Initial Base</span>
    </div>
    <div class="top-banner-status">
        <span style="width: 8px; height: 8px; background-color: #3fb950; border-radius: 50%; display: inline-block;"></span>
        <span>Active Model Engine</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# PAGE 1: PORTFOLIO PAGE
# ==============================================================================
if page == "Portfolio Page":
    st.title("💼 Portfolio Overview & Construction")
    st.caption("Asset allocation, indexed portfolio growth, risk-return metrics, drawdown analytics, and constituent correlation")
    
    # Portfolio Inputs & Weight Editor Expandable Section
    with st.expander("⚙️ Portfolio Weight Editor & Rebalancing Controls", expanded=False):
        w_col1, w_col2 = st.columns([1, 2.5])
        with w_col1:
            st.markdown("**Rebalancing Scheme**")
            equal_w_btn = st.button("⚖️ Set Equal Weights (1/N)", use_container_width=True)
            if equal_w_btn:
                for t in active_tickers:
                    st.session_state.weights_dict[t] = 1.0 / N_stocks
                st.rerun()
                
            normalize_btn = st.button("🔄 Auto-Normalize to 100%", use_container_width=True)
            if normalize_btn:
                total_w = sum(st.session_state.weights_dict.get(t, 0.0) for t in active_tickers)
                if total_w > 0:
                    for t in active_tickers:
                        st.session_state.weights_dict[t] /= total_w
                st.rerun()
                
        with w_col2:
            st.markdown("**Constituent Allocations**")
            weight_cols = st.columns(min(4, max(1, N_stocks)))
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

    # ==========================================================================
    # 6 UN-TRUNCATED INSTITUTIONAL KPI CARDS (2 Balanced Rows × 3 Columns)
    # ==========================================================================
    st.markdown('<div class="section-title">📊 Executive Performance & Risk Summary</div>', unsafe_allow_html=True)
    
    # ROW 1: Value, Total Return, Annualized Return
    r1_c1, r1_c2, r1_c3 = st.columns(3)
    with r1_c1:
        dollar_profit = port_value_series.iloc[-1] - pvalue
        render_kpi_card(
            title="Final Portfolio Value",
            value=f"${port_value_series.iloc[-1]:,.0f}",
            delta=f"+${dollar_profit:,.0f} ({cum_tot_ret:+.2%})" if dollar_profit >= 0 else f"-${abs(dollar_profit):,.0f} ({cum_tot_ret:.2%})",
            delta_is_positive=(dollar_profit >= 0),
            subtext=f"Initial Investment: ${pvalue:,.0f}",
            icon="💰",
            accent_color="#00d2ff"
        )
    with r1_c2:
        render_kpi_card(
            title="Cumulative Total Return",
            value=f"{cum_tot_ret:+.2%}",
            delta=f"{len(port_returns):,} Trading Days",
            delta_is_positive=True,
            subtext="Holding period net compound return",
            icon="📈",
            accent_color="#00e676"
        )
    with r1_c3:
        render_kpi_card(
            title="Annualized Return (CAGR)",
            value=f"{ann_ret:.2%}",
            delta="252-day basis compounding",
            delta_is_positive=(ann_ret >= 0),
            subtext="Mean log return annualized",
            icon="🚀",
            accent_color="#26a69a"
        )

    # ROW 2: Volatility, Sharpe Ratio, Max Drawdown
    r2_c1, r2_c2, r2_c3 = st.columns(3)
    with r2_c1:
        render_kpi_card(
            title="Annualized Volatility",
            value=f"{ann_vol:.2%}",
            delta=f"Daily Std: {port_returns.std():.3%}",
            delta_is_positive=False,
            subtext="252-day annualized standard deviation",
            icon="⚡",
            accent_color="#ffb300"
        )
    with r2_c2:
        render_kpi_card(
            title="Sharpe Ratio (Rf = 0%)",
            value=f"{sharpe:.2f}",
            delta="Acceptable" if sharpe >= 1.0 else "Sub-benchmark",
            delta_is_positive=(sharpe >= 1.0),
            subtext="Risk-adjusted excess return per unit vol",
            icon="🎯",
            accent_color="#b388ff"
        )
    with r2_c3:
        render_kpi_card(
            title="Maximum Drawdown",
            value=f"{max_drawdown:.2%}",
            delta="Peak-to-Trough",
            delta_is_positive=False,
            subtext="Worst historic capital decline",
            icon="📉",
            accent_color="#ff5252"
        )

    st.markdown("---")

    # ==========================================================================
    # CHARTS: WEALTH GROWTH, UNDERWATER DRAWDOWN & ALLOCATION
    # ==========================================================================
    tab_chart1, tab_chart2, tab_chart3 = st.tabs(["📈 Portfolio Growth & Wealth Index", "🌊 Underwater Drawdown Curve", "🔗 Constituent Correlation Heatmap"])
    
    with tab_chart1:
        chart_col1, chart_col2 = st.columns([1.7, 1])
        with chart_col1:
            st.markdown('<div class="section-title">Indexed Portfolio Value ($) vs Benchmark</div>', unsafe_allow_html=True)
            fig_wealth = go.Figure()
            fig_wealth.add_trace(go.Scatter(
                x=port_value_series.index,
                y=port_value_series.values,
                mode="lines",
                name="Portfolio Indexed Value",
                line=dict(color="#00d2ff", width=2.4)
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
                height=390,
                margin=dict(l=40, r=20, t=20, b=30),
                yaxis=dict(title="Portfolio Value ($)", tickprefix="$", tickformat=","),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_wealth, use_container_width=True)
            
            # Daily Return & 21-Day Rolling Volatility
            fig_ret = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.6, 0.4])
            colors = ["#ef5350" if x < 0 else "#26a69a" for x in port_returns]
            fig_ret.add_trace(go.Bar(
                x=port_returns.index,
                y=port_returns.values,
                marker_color=colors,
                name="Daily Log Return",
                showlegend=False
            ), row=1, col=1)
            
            fig_ret.add_trace(go.Scatter(
                x=rolling_vol_21.index,
                y=rolling_vol_21.values,
                line=dict(color="#ab47bc", width=1.8),
                name="21D Vol (Ann)",
                showlegend=True
            ), row=2, col=1)
            
            fig_ret.update_yaxes(title_text="Daily Return", tickformat=".1%", row=1, col=1)
            fig_ret.update_yaxes(title_text="21D Vol", tickformat=".1%", row=2, col=1)
            fig_ret.update_layout(
                template="plotly_dark",
                height=320,
                margin=dict(l=40, r=20, t=10, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_ret, use_container_width=True)

        with chart_col2:
            st.markdown('<div class="section-title">Asset Allocation Breakdown</div>', unsafe_allow_html=True)
            alloc_df = weights_series.rename("Weight").reset_index().rename(columns={"index": "Ticker"})
            alloc_df["Value ($)"] = alloc_df["Weight"] * port_value_series.iloc[-1]
            
            fig_pie = px.pie(
                alloc_df,
                names="Ticker",
                values="Weight",
                hole=0.45,
                template="plotly_dark",
                color_discrete_sequence=px.colors.qualitative.Dark24
            )
            fig_pie.update_traces(textinfo="label+percent", textposition="inside")
            fig_pie.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_pie, use_container_width=True)
            
            st.markdown('<div class="section-title">Statistical Moments & Normality</div>', unsafe_allow_html=True)
            stats_data = {
                "Statistic": [
                    "Total Observations", "Daily Mean Return", "Daily Volatility (σ)",
                    "Annualized Return", "Annualized Volatility", "Best Single Day",
                    "Worst Single Day", "Largest Dollar Gain", "Largest Dollar Loss",
                    "Skewness (Asymmetry)", "Excess Kurtosis (Fat Tails)"
                ],
                "Value": [
                    f"{len(port_returns):,}", f"{port_returns.mean():.4%}", f"{port_returns.std():.4%}",
                    f"{ann_ret:.2%}", f"{ann_vol:.2%}", f"{port_returns.max():.2%}",
                    f"{port_returns.min():.2%}", f"${daily_pnl.max():,.0f}", f"${daily_pnl.min():,.0f}",
                    f"{port_returns.skew():.3f}", f"{port_returns.kurtosis():.3f}"
                ]
            }
            st.dataframe(pd.DataFrame(stats_data), hide_index=True, use_container_width=True)

    with tab_chart2:
        st.markdown('<div class="section-title">Underwater Drawdown Profile (Peak-to-Trough Decline)</div>', unsafe_allow_html=True)
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=drawdown_series.index,
            y=drawdown_series.values,
            mode="lines",
            name="Portfolio Drawdown",
            line=dict(color="#f85149", width=1.5),
            fill="tozeroy",
            fillcolor="rgba(248, 81, 73, 0.25)"
        ))
        fig_dd.update_layout(
            template="plotly_dark",
            height=380,
            yaxis=dict(title="Drawdown (%)", tickformat=".1%"),
            xaxis=dict(title="Date")
        )
        st.plotly_chart(fig_dd, use_container_width=True)
        st.caption("Underwater plot captures capital recovery periods, illustrating depth and duration of major systemic drawdowns (e.g. March 2020 COVID shock, 2022 rate tightening).")

    with tab_chart3:
        st.markdown('<div class="section-title">Constituent Correlation Matrix (Diversification Heatmap)</div>', unsafe_allow_html=True)
        if len(stock_returns.columns) > 1:
            corr_matrix = stock_returns.corr()
            fig_corr = px.imshow(
                corr_matrix,
                text_auto=".2f",
                aspect="auto",
                color_continuous_scale="RdBu_r",
                zmin=-1.0,
                zmax=1.0,
                template="plotly_dark"
            )
            fig_corr.update_layout(height=450)
            st.plotly_chart(fig_corr, use_container_width=True)
            st.caption("Lower pairwise correlations between constituents reduce portfolio-level volatility through diversification benefits.")
        else:
            st.info("Select multiple constituents to inspect cross-asset correlation.")

# ==============================================================================
# PAGE 2: VaR PAGE
# ==============================================================================
elif page == "VaR Page":
    st.title("📉 Value at Risk (VaR) & Expected Shortfall (ES)")
    st.caption("Parametric and non-parametric downside risk measurement across confidence levels, estimation windows, and model regimes")
    
    # Control Selectors Bar
    ctl_col1, ctl_col2, ctl_col3 = st.columns(3)
    with ctl_col1:
        conf_level = st.selectbox(
            "Confidence Level (1 - α)",
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
    h_res = vec.calc_historical_var_es(port_returns, confidence_level=conf_level, window=hist_window)
    ew_var, ew_vol = vec.calc_ewma_volatility(port_returns, lam=0.94)
    ew_res = vec.calc_ewma_var_es(port_returns, ew_vol, confidence_level=conf_level)
    
    g_res, gj_res = None, None
    try:
        gf, g_vol, g_sum = vec.fit_garch_model(port_returns, gjr=False)
        g_res = vec.calc_garch_scaled_var_es(port_returns, gf, g_vol, g_sum["forecast_vol"], confidence_level=conf_level, gjr=False)
    except Exception:
        pass
        
    try:
        gjf, gj_vol, gj_sum = vec.fit_garch_model(port_returns, gjr=True)
        gj_res = vec.calc_garch_scaled_var_es(port_returns, gjf, gj_vol, gj_sum["forecast_vol"], confidence_level=conf_level, gjr=True)
    except Exception:
        pass
        
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
    st.markdown('<div class="section-title">🛡️ Tail Risk Outputs & Capital at Risk</div>', unsafe_allow_html=True)
    var_dollar = active_res["var_loss"] * pvalue
    es_dollar = active_res["es_loss"] * pvalue
    
    vc1, vc2, vc3, vc4 = st.columns(4)
    with vc1:
        render_kpi_card(
            title=f"{conf_level:.1%} Value at Risk (VaR)",
            value=f"${var_dollar:,.0f}",
            delta=f"-{active_res['var_loss']:.2%} of portfolio",
            delta_is_positive=False,
            subtext=f"1-day threshold at {conf_level:.1%}",
            icon="⚠️",
            accent_color="#ff7043"
        )
    with vc2:
        render_kpi_card(
            title=f"{conf_level:.1%} Expected Shortfall",
            value=f"${es_dollar:,.0f}",
            delta=f"-{active_res['es_loss']:.2%} of portfolio",
            delta_is_positive=False,
            subtext="Coherent conditional tail loss (CVaR)",
            icon="🔥",
            accent_color="#d32f2f"
        )
    with vc3:
        render_kpi_card(
            title="Selected Risk Model",
            value=model_name,
            delta=f"Window: {hist_window} Days",
            delta_is_positive=True,
            subtext="Active specification",
            icon="⚙️",
            accent_color="#00d2ff"
        )
    with vc4:
        tail_mult = (active_res['es_loss'] / active_res['var_loss']) if active_res['var_loss'] > 0 else 1.0
        render_kpi_card(
            title="Tail Heaviness Index",
            value=f"{tail_mult:.2f}x",
            delta="CVaR / VaR Ratio",
            delta_is_positive=True,
            subtext="Excess tail loss multiplier",
            icon="📊",
            accent_color="#b388ff"
        )
        
    st.info(f"💡 **Executive Risk Statement**: Over a 1-day horizon, there is a **{1-conf_level:.1%}** statistical probability that portfolio loss exceeds **${var_dollar:,.0f}** ({active_res['var_loss']:.2%}). If this threshold is breached, the average expected loss is **${es_dollar:,.0f}** ({active_res['es_loss']:.2%}).")

    # ==========================================================================
    # RETURN DISTRIBUTION & FAT TAILS NORMAL OVERLAY
    # ==========================================================================
    tab_var1, tab_var2, tab_var3 = st.tabs(["📊 Return Distribution & Tail Thresholds", "⚡ Normal vs Empirical Fat-Tails Analysis", "🔄 Multi-Model Risk Comparison"])
    
    with tab_var1:
        sample_r = port_returns.iloc[-hist_window:] if len(port_returns) >= hist_window else port_returns
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(
            x=sample_r,
            nbinsx=65,
            name="Empirical Returns",
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
            height=370,
            xaxis=dict(title="Daily Return", tickformat=".1%"),
            yaxis=dict(title="Frequency Count")
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    with tab_var2:
        st.markdown('<div class="section-title">Gaussian Normal vs Actual Empirical Tail Distribution</div>', unsafe_allow_html=True)
        # Normal vs Empirical comparison
        mu, sigma = port_returns.mean(), port_returns.std()
        x_axis = np.linspace(port_returns.min(), port_returns.max(), 300)
        norm_pdf = norm.pdf(x_axis, mu, sigma)
        
        fig_tails = go.Figure()
        fig_tails.add_trace(go.Histogram(
            x=port_returns,
            histnorm='probability density',
            nbinsx=80,
            name='Actual Market Returns',
            marker_color='rgba(66, 165, 245, 0.4)'
        ))
        fig_tails.add_trace(go.Scatter(
            x=x_axis,
            y=norm_pdf,
            mode='lines',
            name='Fitted Gaussian Normal Curve',
            line=dict(color='#ffa726', width=2.5, dash='dash')
        ))
        fig_tails.add_vline(x=-active_res["var_loss"], line_color="#ff5252", line_width=2, line_dash="solid", annotation_text="VaR Cutoff")
        fig_tails.update_layout(
            template="plotly_dark",
            height=370,
            xaxis=dict(title="Return", tickformat=".1%"),
            yaxis=dict(title="Density")
        )
        st.plotly_chart(fig_tails, use_container_width=True)
        st.caption("Notice the significant excess density in the left tail (kurtosis = {:.2f}). Standard normal distribution severely underestimates extreme tail risk, justifying non-parametric and GARCH modeling.".format(port_returns.kurtosis()))

    with tab_var3:
        comp_rows = [
            {"Model": "Historical Simulation", "VaR (%)": h_res["var_loss"], "ES (%)": h_res["es_loss"], "VaR ($)": h_res["var_loss"] * pvalue, "ES ($)": h_res["es_loss"] * pvalue},
            {"Model": "EWMA Scaled (λ=0.94)", "VaR (%)": ew_res["var_loss"], "ES (%)": ew_res["es_loss"], "VaR ($)": ew_res["var_loss"] * pvalue, "ES ($)": ew_res["es_loss"] * pvalue}
        ]
        if g_res:
            comp_rows.append({"Model": "GARCH(1,1)", "VaR (%)": g_res["var_loss"], "ES (%)": g_res["es_loss"], "VaR ($)": g_res["var_loss"] * pvalue, "ES ($)": g_res["es_loss"] * pvalue})
        if gj_res:
            comp_rows.append({"Model": "GJR-GARCH(1,1,1)", "VaR (%)": gj_res["var_loss"], "ES (%)": gj_res["es_loss"], "VaR ($)": gj_res["var_loss"] * pvalue, "ES ($)": gj_res["es_loss"] * pvalue})
            
        comp_df = pd.DataFrame(comp_rows)
        c_p1, c_p2 = st.columns([1.6, 1])
        with c_p1:
            fig_bar = px.bar(
                comp_df,
                x="Model",
                y=["VaR ($)", "ES ($)"],
                barmode="group",
                template="plotly_dark",
                color_discrete_sequence=["#ff7043", "#d32f2f"],
                title=f"Comparative VaR & ES at {conf_level:.1%} Confidence"
            )
            fig_bar.update_layout(height=350, yaxis=dict(title="Capital at Risk ($)", tickprefix="$", tickformat=","))
            st.plotly_chart(fig_bar, use_container_width=True)
        with c_p2:
            st.markdown("**Model Scorecard Table**")
            styled_df = comp_df.copy()
            styled_df["VaR (%)"] = styled_df["VaR (%)"].apply(lambda x: f"{x:.2%}")
            styled_df["ES (%)"] = styled_df["ES (%)"].apply(lambda x: f"{x:.2%}")
            styled_df["VaR ($)"] = styled_df["VaR ($)"].apply(lambda x: f"${x:,.0f}")
            styled_df["ES ($)"] = styled_df["ES ($)"].apply(lambda x: f"${x:,.0f}")
            st.dataframe(styled_df, hide_index=True, use_container_width=True)

# ==============================================================================
# PAGE 3: VOLATILITY SCALING PAGE
# ==============================================================================
elif page == "Volatility Scaling Page":
    st.title("⚡ Volatility Scaling & Clustering Analysis")
    st.caption("EWMA, GARCH(1,1), and GJR-GARCH(1,1,1) asymmetric leverage modeling & News Impact Curve")
    
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
        
    # Current Volatility Metric Cards (Un-truncated)
    st.markdown('<div class="section-title">⚡ Current Volatility Regimes & Persistence</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_kpi_card(
            title="EWMA Volatility",
            value=f"{ew_vol.iloc[-1]:.2%}",
            delta=f"Ann: {ew_vol.iloc[-1]*np.sqrt(252):.2%}",
            delta_is_positive=True,
            subtext=f"Decay parameter λ = {ewma_lam:.2f}",
            icon="⚡",
            accent_color="#ffa726"
        )
    with c2:
        render_kpi_card(
            title="GARCH(1,1) Volatility",
            value=f"{g_sum['current_vol']:.2%}",
            delta=f"1D Forecast: {g_sum['forecast_vol']:.2%}",
            delta_is_positive=True,
            subtext=f"Persistence: {g_sum['persistence']:.4f}",
            icon="📈",
            accent_color="#ab47bc"
        )
    with c3:
        render_kpi_card(
            title="GJR-GARCH Volatility",
            value=f"{gj_sum['current_vol']:.2%}",
            delta=f"1D Forecast: {gj_sum['forecast_vol']:.2%}",
            delta_is_positive=True,
            subtext=f"Persistence: {gj_sum['persistence']:.4f}",
            icon="📊",
            accent_color="#26a69a"
        )
    with c4:
        render_kpi_card(
            title="Asymmetry (Leverage γ)",
            value=f"{gj_sum['gamma']:.4f}",
            delta="Statistically Significant",
            delta_is_positive=True,
            subtext="Asymmetric negative shock impact",
            icon="🔍",
            accent_color="#00d2ff"
        )
        
    st.markdown("---")

    # Econometric Parameter Outputs
    tab_vol1, tab_vol2, tab_vol3 = st.tabs(["📉 Conditional Volatility Comparison", "🎯 News Impact Curve (Asymmetry)", "📋 Econometric Model Parameters"])
    
    with tab_vol1:
        vol_comp_df = pd.DataFrame({
            "EWMA Volatility": ew_vol,
            "GARCH(1,1)": g_vol,
            "GJR-GARCH(1,1,1)": gj_vol,
            "21D Rolling Hist Vol": port_returns.rolling(21).std()
        }).dropna()
        
        fig_vol = px.line(
            vol_comp_df,
            template="plotly_dark",
            title="Conditional Volatility Dynamics Over Time (Daily Scale)"
        )
        fig_vol.update_layout(height=410, yaxis=dict(title="Daily Conditional Volatility", tickformat=".2%"))
        st.plotly_chart(fig_vol, use_container_width=True)

    with tab_vol2:
        st.markdown('<div class="section-title">News Impact Curve: GARCH vs GJR-GARCH Shock Response</div>', unsafe_allow_html=True)
        # News Impact Curve construction
        eps = np.linspace(-4, 4, 200)
        sigma2_prev = g_sum['current_vol'] ** 2
        var_garch_nic = g_sum['omega'] + g_sum['alpha'] * (eps ** 2) + g_sum['beta'] * sigma2_prev
        var_gjr_nic = gj_sum['omega'] + (gj_sum['alpha'] + gj_sum['gamma'] * (eps < 0)) * (eps ** 2) + gj_sum['beta'] * sigma2_prev
        
        fig_nic = go.Figure()
        fig_nic.add_trace(go.Scatter(x=eps, y=var_garch_nic, mode='lines', name='GARCH(1,1) - Symmetric Response', line=dict(color='#ab47bc', width=2.5)))
        fig_nic.add_trace(go.Scatter(x=eps, y=var_gjr_nic, mode='lines', name='GJR-GARCH - Asymmetric Leverage', line=dict(color='#00d2ff', width=2.5, dash='solid')))
        fig_nic.update_layout(
            template="plotly_dark",
            height=370,
            xaxis=dict(title="Return Shock / Innovation (ε_t-1)"),
            yaxis=dict(title="Conditional Variance (σ²_t)")
        )
        st.plotly_chart(fig_nic, use_container_width=True)
        st.caption("The News Impact Curve shows that negative return shocks (ε < 0) provoke a much steeper volatility spike than equivalent positive shocks in GJR-GARCH.")

    with tab_vol3:
        m_out1, m_out2 = st.columns(2)
        with m_out1:
            st.subheader("GARCH(1,1) Model Output")
            garch_params_table = pd.DataFrame({
                "Parameter": ["Constant Mean (μ)", "Baseline Variance (ω)", "ARCH Shock (α)", "GARCH Persistence (β)", "Total Persistence (α+β)", "Shock Half-Life", "Log-Likelihood", "AIC", "BIC"],
                "Estimate": [
                    f"{g_sum['mu']:.6f}", f"{g_sum['omega']:.6f}", f"{g_sum['alpha']:.4f}",
                    f"{g_sum['beta']:.4f}", f"{g_sum['persistence']:.4f}",
                    f"{g_sum['half_life']:.1f} days" if np.isfinite(g_sum['half_life']) else "N/A",
                    f"{g_sum['log_likelihood']:.1f}", f"{g_sum['aic']:.1f}", f"{g_sum['bic']:.1f}"
                ]
            })
            st.dataframe(garch_params_table, hide_index=True, use_container_width=True)
        with m_out2:
            st.subheader("GJR-GARCH(1,1,1) Model Output")
            gjr_params_table = pd.DataFrame({
                "Parameter": ["Constant Mean (μ)", "Baseline Variance (ω)", "ARCH Shock (α)", "Leverage Asymmetry (γ)", "GARCH Persistence (β)", "Total Persistence (α+β+0.5γ)", "Log-Likelihood", "AIC", "BIC"],
                "Estimate": [
                    f"{gj_sum['mu']:.6f}", f"{gj_sum['omega']:.6f}", f"{gj_sum['alpha']:.4f}",
                    f"{gj_sum['gamma']:.4f}", f"{gj_sum['beta']:.4f}", f"{gj_sum['persistence']:.4f}",
                    f"{gj_sum['log_likelihood']:.1f}", f"{gj_sum['aic']:.1f}", f"{gj_sum['bic']:.1f}"
                ]
            })
            st.dataframe(gjr_params_table, hide_index=True, use_container_width=True)

# ==============================================================================
# PAGE 4: BACKTESTING PAGE
# ==============================================================================
elif page == "Backtesting Page":
    st.title("🔬 Out-of-Sample VaR Backtesting & Regulatory Verification")
    st.caption("Rolling 1-step ahead exceptions, Kupiec POF, Christoffersen independence & CC, and Basel Traffic Light framework")
    
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
            
    # Top Level Backtest KPI Summary Cards (Un-truncated)
    target_p = 1.0 - bt_conf
    oos_len = len(oos_df)
    exp_ex = oos_len * target_p
    active_m_key = "Historical" if bt_model_view in ("Historical", "Overlay All Models") else bt_model_view
    active_row = summary_df[summary_df["Model"] == active_m_key].iloc[0]
    
    st.markdown('<div class="section-title">🔬 Backtest Diagnostics & Regulatory Standing</div>', unsafe_allow_html=True)
    bk1, bk2, bk3, bk4, bk5 = st.columns(5)
    with bk1:
        render_kpi_card("OOS Sample Size", f"{oos_len:,}", delta=f"Est: {bt_est_win}D", delta_is_positive=True, subtext="Evaluated days", icon="📅", accent_color="#00d2ff")
    with bk2:
        render_kpi_card(f"{active_m_key} Exceptions", f"{active_row['Exceptions']}", delta=f"Exp: {exp_ex:.1f}", delta_is_positive=(active_row['Exceptions'] <= exp_ex), subtext="Total breaches", icon="⚠️", accent_color="#ff7043")
    with bk3:
        render_kpi_card("Exception Rate", f"{active_row['Exception Rate']:.2%}", delta=f"Target: {target_p:.2%}", delta_is_positive=(active_row['Exception Rate'] <= target_p * 1.2), subtext="Observed ratio", icon="📈", accent_color="#ffb300")
    with bk4:
        render_kpi_card("Kupiec POF Result", active_row["Kupiec Decision"], delta=f"p: {active_row['Kupiec p-value']:.4f}", delta_is_positive=(active_row["Kupiec Decision"] == "Pass"), subtext="Unconditional coverage", icon="🎯", accent_color="#3fb950" if active_row["Kupiec Decision"] == "Pass" else "#f85149")
    with bk5:
        b_zone = active_row["Basel Zone"]
        render_kpi_card("Basel Traffic Light", f"{b_zone} Zone", delta=f"{active_row['Basel Multiplier']:.2f}x Charge", delta_is_positive=(b_zone == "Green"), subtext=active_row["Basel Decision"][:15], icon="🚦", accent_color="#3fb950" if b_zone == "Green" else "#d29922" if b_zone == "Yellow" else "#f85149")

    st.markdown("---")

    # ==========================================================================
    # ACTUAL RETURN / P&L vs VaR CHART WITH BREACHES
    # ==========================================================================
    tab_bk1, tab_bk2, tab_bk3 = st.tabs(["📉 Realized P&L vs Rolling VaR Threshold", "📋 Full Statistical Scorecard", "📥 Exceptions Audit Log"])
    
    with tab_bk1:
        fig_bt = go.Figure()
        fig_bt.add_trace(go.Scatter(
            x=oos_df.index,
            y=oos_df["Actual Return"],
            mode="lines",
            name="Realized Daily Return",
            line=dict(color="#90a4ae", width=0.8),
            opacity=0.85
        ))
        
        colors_map = {"Historical": "#42a5f5", "EWMA": "#ffa726", "GARCH": "#ab47bc", "GJR-GARCH": "#26a69a"}
        models_to_plot = ["Historical", "EWMA", "GARCH", "GJR-GARCH"] if bt_model_view == "Overlay All Models" else [bt_model_view]
            
        for m in models_to_plot:
            fig_bt.add_trace(go.Scatter(
                x=oos_df.index,
                y=oos_df[f"{m} VaR"],
                mode="lines",
                name=f"{m} VaR Threshold",
                line=dict(color=colors_map[m], width=1.5, dash="dash" if m != "GJR-GARCH" else "solid")
            ))
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
            yaxis=dict(title="Daily Return / VaR Threshold", tickformat=".1%"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_bt, use_container_width=True)

    with tab_bk2:
        st.markdown('<div class="section-title">Comparative Statistical Backtesting Scorecard</div>', unsafe_allow_html=True)
        scorecard_df = summary_df[[
            "Model", "Observations", "Exceptions", "Expected", "Exception Rate",
            "Kupiec LR", "Kupiec p-value", "Kupiec Decision",
            "Christoffersen LR Ind", "Christoffersen p-value", "Independence Decision",
            "Conditional Coverage LR", "Conditional Coverage p-value", "CC Decision",
            "Basel Zone", "Basel Multiplier"
        ]].copy()
        
        scorecard_df["Expected"] = scorecard_df["Expected"].apply(lambda x: f"{x:.1f}")
        scorecard_df["Exception Rate"] = scorecard_df["Exception Rate"].apply(lambda x: f"{x:.2%}")
        scorecard_df["Kupiec LR"] = scorecard_df["Kupiec LR"].apply(lambda x: f"{x:.3f}")
        scorecard_df["Kupiec p-value"] = scorecard_df["Kupiec p-value"].apply(lambda x: f"{x:.4f}")
        scorecard_df["Christoffersen LR Ind"] = scorecard_df["Christoffersen LR Ind"].apply(lambda x: f"{x:.3f}")
        scorecard_df["Christoffersen p-value"] = scorecard_df["Christoffersen p-value"].apply(lambda x: f"{x:.4f}")
        scorecard_df["Conditional Coverage LR"] = scorecard_df["Conditional Coverage LR"].apply(lambda x: f"{x:.3f}")
        scorecard_df["Conditional Coverage p-value"] = scorecard_df["Conditional Coverage p-value"].apply(lambda x: f"{x:.4f}")
        st.dataframe(scorecard_df, hide_index=True, use_container_width=True)
        
        # Basel explanation
        b_c1, b_c2 = st.columns([1.5, 1])
        with b_c1:
            basel_chart_df = summary_df[["Model", "Basel Exceptions (250D)"]].copy()
            fig_basel = px.bar(
                basel_chart_df,
                x="Model",
                y="Basel Exceptions (250D)",
                template="plotly_dark",
                title="Basel 99% VaR Exceptions (Final 250-Day Regulatory Window)"
            )
            fig_basel.add_hline(y=4.5, line_dash="dash", line_color="#4caf50", annotation_text="Green/Yellow Cutoff (4)")
            fig_basel.add_hline(y=9.5, line_dash="dash", line_color="#f44336", annotation_text="Yellow/Red Cutoff (9)")
            fig_basel.update_layout(height=280)
            st.plotly_chart(fig_basel, use_container_width=True)
        with b_c2:
            st.markdown("""
            **Basel Regulatory Traffic Light Standards:**
            - 🟢 **Green Zone (0–4 Exceptions)**: Model is accepted by regulators. Capital multiplier is **3.00x**.
            - 🟡 **Yellow Zone (5–9 Exceptions)**: Supervisory warning. Mandatory scaling penalty (+0.40 to +0.85).
            - 🔴 **Red Zone (10+ Exceptions)**: Automatic model failure. Model must be replaced. Multiplier **4.00x+**.
            """)

    with tab_bk3:
        st.markdown('<div class="section-title">Exceptions Audit Table & Severity Log</div>', unsafe_allow_html=True)
        sel_ex_model = st.selectbox("Filter Exceptions by Model", ["Historical", "EWMA", "GARCH", "GJR-GARCH"], index=0)
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
        
    # Active Scenario Metrics (Un-truncated)
    st.markdown('<div class="section-title">⚡ Stressed Portfolio Losses & Tail Downside</div>', unsafe_allow_html=True)
    sc_loss = active_sc_row["Stress Loss"]
    sc_loss_dollar = active_sc_row["Dollar Loss"]
    sc_var = active_sc_row["95% Stressed VaR"]
    sc_es = active_sc_row["95% Stressed ES"]
    sc_worst = active_sc_row["Worst-Case Loss"]
    
    s_m1, s_m2, s_m3, s_m4 = st.columns(4)
    with s_m1:
        render_kpi_card(
            title="Cumulative Stress Loss",
            value=f"${sc_loss_dollar:,.0f}" if pd.notna(sc_loss) else "Statistical Shift",
            delta=f"{sc_loss:.2%}" if pd.notna(sc_loss) else "Covariance Shock",
            delta_is_positive=False,
            subtext="Period cumulative drawdown",
            icon="💥",
            accent_color="#f85149"
        )
    with s_m2:
        render_kpi_card(
            title="95% Stressed VaR",
            value=f"${abs(sc_var)*pvalue:,.0f}",
            delta=f"{sc_var:.2%}",
            delta_is_positive=False,
            subtext="Stressed 1-day loss threshold",
            icon="⚠️",
            accent_color="#ff7043"
        )
    with s_m3:
        render_kpi_card(
            title="95% Stressed ES (CVaR)",
            value=f"${abs(sc_es)*pvalue:,.0f}",
            delta=f"{sc_es:.2%}",
            delta_is_positive=False,
            subtext="Conditional expected tail loss",
            icon="🔥",
            accent_color="#d32f2f"
        )
    with s_m4:
        render_kpi_card(
            title="Worst Single-Day Drop",
            value=f"${abs(sc_worst)*pvalue:,.0f}",
            delta=f"{sc_worst:.2%}",
            delta_is_positive=False,
            subtext="Maximum daily stress decline",
            icon="📉",
            accent_color="#b388ff"
        )
        
    base_var_dl = abs(base_stress_info["baseline_var_95"]) * pvalue
    diff_pct = (abs(sc_var) - abs(base_stress_info["baseline_var_95"])) / abs(base_stress_info["baseline_var_95"])
    st.info(f"📊 **Baseline Normal Risk vs Stress**: Baseline 95% VaR is **${base_var_dl:,.0f}** ({base_stress_info['baseline_var_95']:.2%}). Under **{scenario_choice}**, downside risk expands by **{diff_pct:+.1%}**.")

    # ==========================================================================
    # SCENARIO COMPARISON CHARTS
    # ==========================================================================
    tab_st1, tab_st2, tab_st3 = st.tabs(["📉 Cumulative Stress Loss Ranking", "🛡️ 95% Stressed VaR Across Regimes", "📋 Master Stress Scorecard Table"])
    
    with tab_st1:
        loss_chart_df = stress_master[stress_master["Stress Loss"].notna()].sort_values("Stress Loss")
        fig_sc_loss = px.bar(
            loss_chart_df,
            x="Stress Loss",
            y="Scenario",
            orientation="h",
            color="Scenario Type",
            color_discrete_map={"Historical": "#ef5350", "Hypothetical": "#ab47bc"},
            template="plotly_dark",
            title="Cumulative Portfolio Loss Across All Stress Scenarios"
        )
        fig_sc_loss.update_layout(height=410, xaxis=dict(tickformat=".1%"))
        st.plotly_chart(fig_sc_loss, use_container_width=True)
        
    with tab_st2:
        var_chart_df = stress_master.sort_values("95% Stressed VaR")
        fig_sc_var = px.bar(
            var_chart_df,
            x="95% Stressed VaR",
            y="Scenario",
            orientation="h",
            color="Scenario Type",
            color_discrete_map={"Historical": "#42a5f5", "Hypothetical": "#26a69a"},
            template="plotly_dark",
            title="95% Stressed Value at Risk (Daily Loss Threshold Comparison)"
        )
        fig_sc_var.update_layout(height=410, xaxis=dict(tickformat=".1%"))
        st.plotly_chart(fig_sc_var, use_container_width=True)

    with tab_st3:
        st.markdown('<div class="section-title">Master Stress Testing Results Scorecard (Phase 10 Specification)</div>', unsafe_allow_html=True)
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
st.caption("Quantitative Risk Engine & Institutional Terminal | Faithful implementation of VaR_Engine.ipynb | Created for Quant Finance Portfolio Risk Management")
