import streamlit as st
import pandas as pd
import numpy as np
import pickle
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# -------------------------------
# Load the trained model
# -------------------------------
with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# -------------------------------
# Fetch stock data (CACHED)
# -------------------------------
@st.cache_data(ttl=300)   # 5 minutes cache
def fetch_stock_data(stock_symbol):
    stock = yf.Ticker(stock_symbol)
    df = stock.history(period="1mo")
    return df

@st.cache_data(ttl=300)
def download_stock_data(stock_symbol):
    return yf.download(stock_symbol, period="6mo")

# -------------------------------
# Compute RSI
# -------------------------------
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# -------------------------------
# Prepare Features
# -------------------------------
def prepare_features(df):
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['RSI'] = compute_rsi(df['Close'], 14)

    df = df.dropna()

    latest_features = df[['Close', 'EMA20', 'EMA50', 'RSI']].iloc[-1].values

    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform([latest_features])

    return scaled_features[0]

# -------------------------------
# Predict stock movement
# -------------------------------
def predict_stock_movement(features):
    prediction = model.predict([features])
    return "BUY 📈" if prediction[0] == 1 else "SELL 📉"

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("📊 Predictive AI Model for Stock Market Trends and Analysis")

st.sidebar.header("Enter Stock Symbol")
stock_symbol = st.sidebar.text_input(
    "Stock Symbol (e.g., AAPL, TSLA, GOOG)", "AAPL"
)

# -------------------------------
# Predict Button
# -------------------------------
if st.sidebar.button("Predict"):

    try:
        df = fetch_stock_data(stock_symbol.upper())

        if df.empty:
            st.sidebar.error("No data found. Try another stock symbol.")
        else:
            features = prepare_features(df)
            result = predict_stock_movement(features)
            st.sidebar.success(f"Prediction for {stock_symbol.upper()}: {result}")

    except Exception:
        st.sidebar.error("Too many requests. Please wait 10-15 minutes and try again.")

# -------------------------------
# Plot Stock Data
# -------------------------------
def plot_stock_prices(df, stock_symbol):
    st.subheader(f"Stock Price Trend for {stock_symbol.upper()}")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df.index, df['Close'])
    ax.set_xlabel("Date")
    ax.set_ylabel("Stock Price (USD)")
    st.pyplot(fig)

# -------------------------------
# Show Data & Graph Button
# -------------------------------
if st.button("Show Stock Data & Graph"):

    try:
        df = download_stock_data(stock_symbol.upper())

        if df.empty:
            st.error("No data found. Try another stock symbol.")
        else:
            st.write(df)
            plot_stock_prices(df, stock_symbol)

    except Exception:
        st.error("Too many requests. Please wait some time and try again.")