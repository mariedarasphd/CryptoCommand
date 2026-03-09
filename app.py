import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
from textblob import TextBlob
import numpy as np

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Crypto Command",
    page_icon="logo.png",
    layout="wide",
)

# --- LOGO ---
st.image("logo.png", width=250)

# --- TITLE ---
st.title("Crypto Command: BTC & ETH Dashboard")

# --- FETCH DATA ---
coins = ["BTC-USD", "ETH-USD"]
price_frames = []

for coin in coins:
    data = yf.download(coin, start="2022-01-01", end="2026-03-01", auto_adjust=True)
    if data.empty:
        st.warning(f"No data for {coin}")
    else:
        df_coin = data[['Close']].rename(columns={'Close': coin})
        price_frames.append(df_coin)

df = pd.concat(price_frames, axis=1)
df = df.ffill()  # forward fill missing values

# --- PLOT PRICES ---
fig_prices = px.line(
    df,
    x=df.index,
    y=df.columns,
    labels={'value': 'Price (USD)', 'variable': 'Coin', 'index': 'Date'},
    title="BTC & ETH Historical Prices"
)
st.plotly_chart(fig_prices, use_container_width=True)

# --- SENTIMENT ANALYSIS ---
st.subheader("Crypto News Sentiment Analysis")

# Placeholder headlines (replace with API later)
news_headlines = [
    "Bitcoin hits new all-time high",
    "Ethereum network faces congestion issues",
    "Regulators consider stricter crypto rules",
]

for headline in news_headlines:
    sentiment = TextBlob(headline).sentiment.polarity
    sentiment_label = "Positive" if sentiment > 0 else "Negative" if sentiment < 0 else "Neutral"
    st.write(f"**{headline}** — Sentiment: {sentiment_label} ({sentiment:.2f})")

# --- SIMULATION PLACEHOLDER ---
st.subheader("Crypto Price Simulation (Coming Soon)")

st.write("This section will allow you to simulate future BTC/ETH prices using Monte Carlo or other predictive models.")
st.slider("Simulation horizon (days)", min_value=1, max_value=365, value=30)

# --- FOOTER ---
st.markdown("---")
st.markdown("© 2026 Crypto Command | Black & Gold Theme")
