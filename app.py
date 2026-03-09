# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
from textblob import TextBlob
import requests

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="Crypto Command",
    page_icon="logo.png",
    layout="wide"
)

# -------------------------
# CUSTOM CSS
# -------------------------
st.markdown(
    """
    <style>
    .css-18e3th9 {background-color: black; color: gold;}
    .st-bb {background-color: black; color: gold;}
    .stApp {background-color: black; color: gold;}
    .stButton>button {background-color: gold; color: black;}
    </style>
    """, unsafe_allow_html=True
)

# -------------------------
# LOGO
# -------------------------
st.image("logo.png", width=250)

st.title("Crypto Command")
st.markdown("**Predict, simulate, and manage your crypto portfolio with intelligence.**")

# -------------------------
# DATA FETCHING
# -------------------------
coins = ["BTC-USD", "ETH-USD"]
price_frames = []

for coin in coins:
    data = yf.download(coin, start="2022-01-01", end="2026-03-01", auto_adjust=True, interval='1d')
    if data.empty:
        st.warning(f"No data for {coin}")
    else:
        df_coin = data[['Close']].rename(columns={'Close': coin})
        price_frames.append(df_coin)

# Merge all coins on the index (Date)
df = pd.concat(price_frames, axis=1)
df = df.ffill()

# -------------------------
# HISTORICAL PRICE CHART
# -------------------------
st.header("Historical Prices")
fig_prices = px.line(
    df,
    x=df.index,
    y=df.columns,
    labels={'value': 'Price (USD)', 'Date': 'Date', 'variable': 'Coin'},
    title="BTC & ETH Historical Prices"
)
st.plotly_chart(fig_prices, use_container_width=True)

# -------------------------
# PORTFOLIO SIMULATOR
# -------------------------
st.header("Portfolio Simulator")
st.write("Simulate how market movements affect your portfolio.")

col1, col2 = st.columns(2)
with col1:
    btc_qty = st.number_input("BTC Holdings", min_value=0.0, value=1.0)
with col2:
    eth_qty = st.number_input("ETH Holdings", min_value=0.0, value=5.0)

col3, col4 = st.columns(2)
with col3:
    btc_change = st.slider("BTC % Price Change", -50, 50, 0)
with col4:
    eth_change = st.slider("ETH % Price Change", -50, 50, 0)

portfolio = {"BTC-USD": btc_qty, "ETH-USD": eth_qty}
new_values = {}

for coin, qty in portfolio.items():
    pct = btc_change/100 if coin == "BTC-USD" else eth_change/100
    new_values[coin] = qty * df[coin].iloc[-1] * (1 + pct)

total_value = sum(new_values.values())
st.metric("Simulated Portfolio Value", f"${total_value:,.2f}")

sim_df = pd.DataFrame(list(new_values.items()), columns=["Coin","Value"])
fig_sim = px.bar(
    sim_df,
    x="Coin",
    y="Value",
    color="Coin",
    title="Simulated Portfolio Allocation"
)
st.plotly_chart(fig_sim, use_container_width=True)

# -------------------------
# NEWS & SENTIMENT PLACEHOLDER
# -------------------------
st.header("News & Sentiment")
st.write("This section will show news sentiment and AI predictions in Phase 2.")

# Example: placeholder table
st.table(pd.DataFrame({
    "Source": ["CoinDesk", "CryptoNews", "Twitter"],
    "Headline": ["BTC rallies 5%", "ETH hits new ATH", "Whales accumulating BTC"],
    "Sentiment": ["Positive", "Positive", "Neutral"]
}))
