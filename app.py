import streamlit as st 
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from textblob import TextBlob
import plotly.graph_objects as go
import plotly.express as px

# -----------------------------------
# PAGE CONFIG
# -----------------------------------

st.set_page_config(
    page_title="Crypto Command",
    page_icon="logo.png",
    layout="wide"
)

# -----------------------------------
# CUSTOM BLACK + GOLD THEME
# -----------------------------------

st.markdown("""
<style>

.stApp {
    background-color: #000000;
    color: gold;
}

h1, h2, h3 {
    color: gold;
}

[data-testid="stSidebar"] {
    background-color: #111111;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------------
# LOGO + TITLE
# -----------------------------------

st.image("logo.png", width=300)

st.title("Crypto Command")

st.write("AI-powered crypto analytics dashboard")

# -----------------------------------
# HISTORICAL PRICE DATA
# -----------------------------------

st.header("Crypto Price Trends")

coins = [
    "BTC-USD",
    "ETH-USD",
    "SOL-USD",
    "ADA-USD",
    "XRP-USD"
]

price_frames = []

for coin in coins:

    data = yf.download(
        coin,
        start="2022-01-01",
        auto_adjust=True,
        interval="1d"
    )

    df_coin = data[['Close']].rename(columns={'Close': coin})

    price_frames.append(df_coin)

df = pd.concat(price_frames, axis=1)
df = df.ffill()

# -----------------------------------
# PRICE CHART
# -----------------------------------

fig = go.Figure()

for coin in coins:

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df[coin],
            mode="lines",
            name=coin
        )
    )

fig.update_layout(
    title="Crypto Price Trends",
    template="plotly_dark"
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------------
# LIVE MARKET DATA
# -----------------------------------

st.header("Live Crypto Market Data")

url = "https://api.coingecko.com/api/v3/coins/markets"

params = {
    "vs_currency": "usd",
    "order": "market_cap_desc",
    "per_page": 50,
    "page": 1
}

data = requests.get(url, params=params).json()

market_df = pd.DataFrame(data)[[
    "name",
    "symbol",
    "current_price",
    "market_cap",
    "total_volume",
    "price_change_percentage_24h"
]]

market_df.columns = [
    "Coin",
    "Symbol",
    "Price",
    "Market Cap",
    "Volume",
    "24h Change %"
]

st.dataframe(market_df)

# -----------------------------------
# CRYPTO HEATMAP
# -----------------------------------

st.header("Crypto Market Heatmap")

heatmap_df = market_df.copy()

fig2 = px.treemap(
    heatmap_df,
    path=["Coin"],
    values="Market Cap",
    color="24h Change %",
    color_continuous_scale=["red", "black", "gold"]
)

st.plotly_chart(fig2, use_container_width=True)

# -----------------------------------
# PORTFOLIO SIMULATOR
# -----------------------------------

st.header("Portfolio Simulator")

btc_change = st.slider("BTC % Change", -50, 50, 0)

eth_change = st.slider("ETH % Change", -50, 50, 0)

portfolio = {
    "BTC-USD": 1,
    "ETH-USD": 5
}

new_values = {}

for coin, qty in portfolio.items():

    pct = btc_change/100 if coin == "BTC-USD" else eth_change/100

    new_values[coin] = qty * df[coin].iloc[-1] * (1 + pct)

total = sum(new_values.values())

st.metric("Simulated Portfolio Value", f"${total:,.2f}")

sim_df = pd.DataFrame(list(new_values.items()), columns=["Coin","Value"])

st.bar_chart(sim_df.set_index("Coin"))
