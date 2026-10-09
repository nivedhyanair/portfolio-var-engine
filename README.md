# Portfolio VaR Engine: Equity Portfolio Risk Management

A Python-based portfolio risk analysis project exploring **Value at Risk (VaR), Expected Shortfall (ES), volatility modelling, backtesting, and stress testing**.

The project compares historical loss-based risk measurement with volatility models to examine how portfolio risk estimates respond to changing market conditions.

## Project Overview

Market risk cannot be understood from a single risk estimate. Historical losses, changing volatility, and periods of market stress can all affect the reliability of a model.

This project develops a portfolio risk analysis workflow to investigate these challenges using Python. It combines historical simulation with conditional volatility models and backtesting techniques to evaluate portfolio downside risk.

## Key Areas of Analysis

* **Historical Simulation VaR:** Estimates potential portfolio losses using historical return observations.
* **Expected Shortfall (ES):** Examines the average loss in the tail beyond the VaR threshold.
* **EWMA Volatility:** Models time-varying volatility using exponentially weighted observations.
* **GARCH Modelling:** Analyses conditional volatility clustering in financial returns.
* **GJR-GARCH Modelling:** Extends volatility analysis to account for potential asymmetry in the response to positive and negative shocks.
* **VaR Backtesting:** Compares risk estimates with realised portfolio outcomes and examines exceptions.
* **Statistical Backtesting:** Uses tests such as Kupiec's unconditional coverage test and Christoffersen's conditional coverage framework, where implemented in the notebook.
* **Stress Testing:** Explores portfolio sensitivity to adverse market conditions.

## Methodology

The analysis follows a risk-modelling workflow:

1. Prepare financial market data and calculate returns.
2. Construct portfolio returns and profit-and-loss measures.
3. Estimate portfolio downside risk using historical simulation.
4. Model conditional volatility using EWMA, GARCH, and GJR-GARCH approaches.
5. Compare risk estimates with realised outcomes through backtesting.
6. Examine statistical test results and assess model limitations.

## Why This Project Matters

The objective is not simply to calculate a VaR number. It is to understand how different modelling assumptions influence risk estimates and how effectively those estimates capture realised losses.

The project brings together portfolio analytics, time-series volatility modelling, tail-risk measurement, and model validation in a single research workflow.

## Repository Structure

```text
portfolio-var-engine/
└── VaR_Engine.ipynb
```

The notebook contains the current analysis. The project will be expanded with a written research report and an interactive risk dashboard as those components are completed.

## Technology

* Python
* Jupyter Notebook
* Financial time-series analysis
* Statistical volatility modelling
* Portfolio risk measurement and backtesting

Refer to the notebook's imports and setup cells for the exact package requirements.

## Limitations

VaR and Expected Shortfall estimates depend on the data, model assumptions, confidence level, and estimation window. Historical observations may not represent future market conditions, and statistical backtesting results should be interpreted rather than treated as automatic evidence that a model is reliable.

The findings are specific to the portfolio and sample examined in the notebook.

## Disclaimer

This project is for educational and analytical purposes only. It is not investment advice, a trading recommendation, or a production-ready risk management system.
