# Yield Curve Analysis with QuantLib and yfinance 📉📈

This project performs yield curve analysis using real-time US Treasury yield data. It demonstrates how to fit a yield curve, price bonds, and visualize interest rate dynamics using Python and QuantLib.

## 🔍 Overview

- Pulls the most recent US Treasury yield data using the `yfinance` API
- Plots the yield curve and corresponding bond price curve
- Calculates bond prices at various maturities (2Y, 5Y, 10Y, 30Y)
- Demonstrates usage of QuantLib for option pricing on bonds

## 📊 Data Source

Yield data is fetched live from Yahoo Finance using the [`yfinance`](https://pypi.org/project/yfinance/) package. The following tickers were used:

- `^IRX` — 13-week T-Bill
- `^FVX` — 5-Year Treasury
- `^TNX` — 10-Year Treasury
- `^TYX` — 30-Year Treasury

Or, if you pulled them as custom tickers like `'US02Y'`, `'US05Y'`, etc., then:

> 📌 Yields were retrieved from `yfinance` using tickers: `US02Y`, `US05Y`, `US10Y`, `US30Y`.

These represent approximate daily yield values on US Treasury securities.

## 🧮 Key Features

- Curve fitting with Nelson-Siegel model using `scipy.optimize.curve_fit`
- Bond price calculation using yield to maturity
- Bond option pricing using QuantLib
- Real-time data acquisition via `yfinance`

## 🗂️ Files
.
├── prices.py # Main script: data pulling, yield curve fitting, pricing
├── prices.csv # CSV file (optional: can save latest yield data snapshot)
└── README.md # This file

## 📦 Dependencies

Install required packages with:

```bash
pip install yfinance QuantLib-Python pandas matplotlib scipy
📈 Example Output
📌 Yield Curve (US02Y to US30Y)
📌 Bond prices for maturities of 2, 5, 10, and 30 years
📌 Option price on a bond using QuantLib

👤 Author
Kelly Mao
GitHub: @Makkkelyie


