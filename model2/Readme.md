# ₿ Bitcoin Next-Day Closing Price Predictor

A Machine Learning web application built with **Scikit-Learn** and **Streamlit** that fetches real-time Bitcoin (`BTC-USD`) market data using the Yahoo Finance API (`yfinance`) and predicts the next day's closing price.

---

## 🚀 Features
- **Live Market Data:** Automatically downloads the last 10 days of Bitcoin daily price action (`Open`, `High`, `Low`, `Close`, `Volume`).
- **Smart Completed-Candle Detection:** Uses yesterday's completed candle to predict today's closing price, with an automatic fallback if Yahoo Finance hasn't updated the latest daily candle yet.
- **Interactive Dashboard:** Displays a 10-day historical closing price chart, the automatic input feature table, and predicted price metrics with delta comparison.

---

## 🛠️ Tech Stack
- **Language:** Python
- **Machine Learning:** Scikit-Learn, Joblib
- **Data Fetching & Processing:** `yfinance`, Pandas, NumPy
- **Web Framework:** Streamlit

---

## 📂 Project Structure
```text
model2/
├── bitcoin_app.py        # Streamlit web application
├── model.py              # Model training script
├── bitcoin_model.pkl     # Trained regression model
├── requirements.txt      # Required Python libraries
└── README.md             # Project documentation