# Portfolio VaR Engine: Equity Portfolio Risk Management

A Python-based portfolio risk analysis project exploring **Value at Risk (VaR), Expected Shortfall (ES), volatility modelling, backtesting, and stress testing**, with an interactive dashboard deployed on Streamlit Cloud.

The project compares historical loss-based risk measurement with conditional volatility models to examine how portfolio risk estimates respond to changing market conditions.

## Interactive Risk Dashboard

**[Launch the Portfolio VaR Engine Dashboard](https://portfolio-var-engine-nqev2xi4efhdfvax2mvzqp.streamlit.app/)**

The dashboard provides an interactive interface for exploring portfolio risk measures, volatility estimates, backtesting results, and stress-testing scenarios.

Key dashboard features include:

* **Portfolio Analysis:** Configure portfolio assets, weights, portfolio value, and analysis dates.
* **VaR and Expected Shortfall:** Explore risk estimates across different confidence levels, historical windows, and supported models.
* **Volatility Modelling:** Examine EWMA, GARCH, and GJR-GARCH volatility estimates and compare their behaviour.
* **Backtesting:** Inspect realised portfolio outcomes against VaR thresholds, review exceptions, and examine Kupiec, Christoffersen, and Basel Traffic Light results where implemented.
* **Stress Testing:** Explore adverse market scenarios, estimated stress losses, stress VaR, and scenario comparisons.

The dashboard is designed to make the project's risk analysis more accessible through interactive controls and visualisations.

## Project Overview

Market risk cannot be understood from a single risk estimate. Historical losses, changing volatility, and periods of market stress can all affect the reliability of a model.

This project develops a portfolio risk analysis workflow to investigate these challenges using Python. It combines historical simulation with conditional volatility models and backtesting techniques to evaluate portfolio downside risk.

The Jupyter notebook contains the core research and analysis, while the deployed dashboard provides an interactive interface for exploring portfolio risk.

## Key Areas of Analysis

* **Historical Simulation VaR:** Estimates potential portfolio losses using historical return observations.
* **Expected Shortfall (ES):** Examines the average loss in the tail beyond the VaR threshold.
* **EWMA Volatility:** Models time-varying volatility using exponentially weighted observations.
* **GARCH Modelling:** Analyses conditional volatility clustering in financial returns.
* **GJR-GARCH Modelling:** Extends volatility analysis to account for potential asymmetry in the response to positive and negative shocks.
* **VaR Backtesting:** Compares risk estimates with realised portfolio outcomes and examines exceptions.
* **Statistical Backtesting:** Uses tests such as Kupiec's unconditional coverage test and Christoffersen's conditional coverage framework, where implemented.
* **Stress Testing:** Explores portfolio sensitivity to adverse market conditions.

## Methodology

The analysis follows a risk-modelling workflow:

1. Prepare financial market data and calculate returns.
2. Construct portfolio returns and profit-and-loss measures.
3. Estimate portfolio downside risk using historical simulation.
4. Model conditional volatility using EWMA, GARCH, and GJR-GARCH approaches.
5. Compare risk estimates with realised outcomes through backtesting.
6. Examine statistical test results and assess model limitations.
7. Explore adverse scenarios and compare portfolio stress outcomes through the dashboard.

The precise implementation, estimation windows, assumptions, and available model outputs are defined by the notebook and dashboard code.

## Why This Project Matters

The objective is not simply to calculate a VaR number. It is to understand how different modelling assumptions influence risk estimates and how effectively those estimates capture realised losses.

The project brings together portfolio analytics, time-series volatility modelling, tail-risk measurement, model validation, and interactive risk visualisation in a single research workflow.

## Repository Structure

```text
portfolio-var-engine/
├── VaR_Engine.ipynb
├── README.md
└── portfolio_var_dashboard/
    ├── app.py
    ├── requirements.txt
    └── README.md
```

* `VaR_Engine.ipynb` — Jupyter notebook containing the portfolio risk analysis and modelling workflow.
* `portfolio_var_dashboard/app.py` — Streamlit application for interactive portfolio risk analysis and visualisation.
* `portfolio_var_dashboard/requirements.txt` — Python dependencies required by the dashboard.
* `portfolio_var_dashboard/README.md` — Dashboard-specific setup and usage instructions.

## Technology

* Python
* Jupyter Notebook
* Streamlit
* Plotly
* pandas and NumPy
* Financial time-series analysis
* Statistical volatility modelling
* Portfolio risk measurement and backtesting

Refer to the notebook's imports and setup cells, along with the dashboard's requirements file, for the respective package dependencies.

## Limitations

VaR and Expected Shortfall estimates depend on the data, model assumptions, confidence level, and estimation window. Historical observations may not represent future market conditions, and statistical backtesting results should be interpreted rather than treated as automatic evidence that a model is reliable.

Dashboard outputs depend on the selected portfolio inputs, data availability, model configuration, and implementation. Interactive results should be interpreted in the context of the underlying methodology and its assumptions.

The findings are specific to the portfolio and sample examined in the relevant analysis. The dashboard's available calculations and controls should be understood in conjunction with the implemented code.

## Disclaimer

This project is for educational and analytical purposes only. It is not investment advice, a trading recommendation, or a production-ready risk management system.

## Live Application

**[Open the Interactive Portfolio VaR Engine Dashboard](https://portfolio-var-engine-nqev2xi4efhdfvax2mvzqp.streamlit.app/)**
