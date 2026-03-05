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
model = pickle.load(open("model.pkl", "rb"))

# -------------------------------
# Fetch stock data
# -------------------------------
@st.cache_data(ttl=300)
def fetch_stock_data(stock_symbol):
    stock = yf.Ticker(stock_symbol)
    df = stock.history(period="1mo")
    return df

@st.cache_data(ttl=300)
def download_stock_data(stock_symbol):
    return yf.download(stock_symbol, period="6mo")

# -------------------------------
# RSI Calculation
# -------------------------------
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# -------------------------------
# Feature Preparation
# -------------------------------
def prepare_features(df):
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['RSI'] = compute_rsi(df['Close'], 14)

    df = df.dropna()

    latest = df[['Close','EMA20','EMA50','RSI']].iloc[-1].values

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform([latest])

    return scaled[0]

# -------------------------------
# Prediction
# -------------------------------
def predict_stock(features):
    prediction = model.predict([features])
    return "BUY 📈" if prediction[0] == 1 else "SELL 📉"

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="AI Stock Predictor", layout="wide")

st.title("📊 AI Stock Market Prediction App")

st.sidebar.header("Enter Stock Symbol")
stock_symbol = st.sidebar.text_input(
    "Example: AAPL, TSLA, GOOG",
    "AAPL"
)

# -------------------------------
# Prediction Button
# -------------------------------
if st.sidebar.button("Predict Stock"):

    try:
        df = fetch_stock_data(stock_symbol.upper())

        if df.empty:
            st.sidebar.error("Invalid stock symbol")
        else:
            features = prepare_features(df)
            result = predict_stock(features)
            st.sidebar.success(f"Prediction: {result}")

    except:
        st.sidebar.error("API request limit reached. Try again later.")

# -------------------------------
# Graph
# -------------------------------
def plot_stock(df, stock_symbol):

    st.subheader(f"{stock_symbol.upper()} Stock Price Trend")

    fig, ax = plt.subplots()

    ax.plot(df.index, df["Close"])
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")

    st.pyplot(fig)

# -------------------------------
# Show Data Button
# -------------------------------
if st.button("Show Stock Data & Graph"):

    try:
        df = download_stock_data(stock_symbol.upper())

        if df.empty:
            st.error("No stock data found")
        else:
            st.write(df)
            plot_stock(df, stock_symbol)

    except:
        st.error("Too many requests. Try again later.")
