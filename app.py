# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px

# ---------- Page Config ----------
st.set_page_config(
    page_title="Crypto Command",
    page_icon="🪙",
    layout="wide"
)

# ---------- Custom CSS for Black/Gold Theme ----------
st.markdown(
    """
    <style>
    .css-18e3th9 {background-color: #000000;}  /* main background */
    .css-1d391kg {color: #FFD700;}  /* text color gold */
    .stButton>button {background-color: #FFD700; color: black;}
    .stPlotlyChart {background-color: #000000;}
    </style>
    """,
    unsafe_allow_html=True
)

# ---------- Logo ----------
st.image("logo.png", width=200)

st.title("Crypto Command")
st.write("Predict, simulate, and analyze your crypto portfolio.")

# ---------- Load Historical Prices ----------
coins = ["BTC-USD", "ETH-USD"]
price_frames = []

for coin in coins:
    data = yf.download(coin, start="2022-01-01", end="2026-03-01", auto_adjust=True, interval='1d')
    if data.empty:
        st.warning(f"No data for {coin}")
    else:
        df_coin = data[['Close']].rename(columns={'Close': coin})
        price_frames.append(df_coin)

# Merge all coins
df = pd.concat(price_frames, axis=1)
df = df.ffill()

# Flatten MultiIndex if needed
if isinstance(df.columns, pd.MultiIndex):
    df.columns = [col[-1] if isinstance(col, tuple) else col for col in df.columns]

# Reset index and melt for Plotly
df_reset = df.reset_index()
df_long = df_reset.melt(id_vars='Date', var_name='Coin', value_name='Price')

# ---------- Historical Prices Chart ----------
st.subheader("Historical Prices")
fig_prices = px.line(
    df_long,
    x='Date',
    y='Price',
    color='Coin',
    title="BTC & ETH Historical Prices",
    labels={'Price': 'Price (USD)', 'Date': 'Date', 'Coin': 'Coin'},
    template="plotly_dark"
)
st.plotly_chart(fig_prices, use_container_width=True)

# ---------- Portfolio Simulation ----------
st.subheader("Portfolio Simulation")
st.write("Test hypothetical changes to your crypto holdings:")

# Simple widget for simulation
holdings = {}
for coin in coins:
    holdings[coin] = st.number_input(f"{coin} Holdings:", min_value=0.0, value=1.0, step=0.1)

st.write("Simulate price change (%):")
price_change = {}
for coin in coins:
    price_change[coin] = st.slider(f"{coin} change %", -50, 100, 0)

# Compute simulated portfolio
simulated_value = 0
for coin in coins:
    last_price = df[coin].iloc[-1]
    simulated_price = last_price * (1 + price_change[coin]/100)
    simulated_value += simulated_price * holdings[coin]

st.success(f"Simulated Portfolio Value: ${simulated_value:,.2f}")

# ---------- Sentiment Analysis Placeholder ----------
st.subheader("Sentiment Analysis")
st.write("This section will integrate crypto news and social media sentiment.")

# ---------- Footer ----------
st.markdown("---")
st.markdown("© 2026 Crypto Command | Black & Gold Theme")
