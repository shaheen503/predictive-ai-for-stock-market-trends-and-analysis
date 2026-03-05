import streamlit as st
import pandas as pd
import numpy as np
import pickle
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# ---------------------------------
# PAGE CONFIG
# ---------------------------------
st.set_page_config(
    page_title="AI Stock Predictor",
    page_icon="📈",
    layout="wide"
)

st.title("📊 AI Stock Market Prediction App")
st.markdown("Machine Learning based **Stock Buy/Sell Prediction System**")

# ---------------------------------
# LOAD MODEL
# ---------------------------------
@st.cache_resource
def load_model():
    return pickle.load(open("model.pkl", "rb"))

model = load_model()

# ---------------------------------
# FETCH STOCK DATA
# ---------------------------------
@st.cache_data(ttl=900)
def fetch_stock_data(symbol):
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(period="3mo")
        return df
    except:
        return pd.DataFrame()

@st.cache_data(ttl=900)
def download_stock_data(symbol):
    try:
        df = yf.download(symbol, period="6mo", progress=False)
        return df
    except:
        return pd.DataFrame()

# ---------------------------------
# RSI CALCULATION
# ---------------------------------
def compute_rsi(series, period=14):

    delta = series.diff()

    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()

    rs = gain / loss

    rsi = 100 - (100 / (1 + rs))

    return rsi

# ---------------------------------
# FEATURE ENGINEERING
# ---------------------------------
def prepare_features(df):

    df["EMA20"] = df["Close"].ewm(span=20).mean()
    df["EMA50"] = df["Close"].ewm(span=50).mean()

    df["RSI"] = compute_rsi(df["Close"])

    df = df.dropna()

    if len(df) == 0:
        return None

    latest = df[["Close","EMA20","EMA50","RSI"]].iloc[-1].values

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform([latest])

    return scaled[0]

# ---------------------------------
# PREDICTION
# ---------------------------------
def predict_stock(features):

    prediction = model.predict([features])[0]

    if prediction == 1:
        return "BUY 📈"
    else:
        return "SELL 📉"

# ---------------------------------
# SIDEBAR INPUT
# ---------------------------------
st.sidebar.header("Stock Input")

stock_symbol = st.sidebar.text_input(
    "Enter Stock Symbol",
    value="AAPL"
)

# ---------------------------------
# PREDICT BUTTON
# ---------------------------------
if st.sidebar.button("Predict Stock"):

    with st.spinner("Analyzing Stock Data..."):

        df = fetch_stock_data(stock_symbol.upper())

        if df.empty:
            st.sidebar.error("❌ Invalid stock symbol or API limit reached")
        else:

            features = prepare_features(df)

            if features is None:
                st.sidebar.error("Not enough data for prediction")
            else:

                result = predict_stock(features)

                st.sidebar.success(f"Prediction: **{result}**")

# ---------------------------------
# PLOT GRAPH
# ---------------------------------
def plot_stock(df, symbol):

    st.subheader(f"{symbol.upper()} Stock Price Trend")

    fig, ax = plt.subplots()

    ax.plot(df.index, df["Close"])

    ax.set_xlabel("Date")
    ax.set_ylabel("Price")

    st.pyplot(fig)

# ---------------------------------
# SHOW DATA BUTTON
# ---------------------------------
if st.button("Show Stock Data & Graph"):

    with st.spinner("Loading Stock Data..."):

        df = download_stock_data(stock_symbol.upper())

        if df.empty:
            st.error("❌ Unable to fetch stock data")
        else:

            st.dataframe(df)

            plot_stock(df, stock_symbol)
