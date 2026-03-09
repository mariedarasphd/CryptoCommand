import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from textblob import TextBlob
import plotly.graph_objects as go

# -----------------------------
# PAGE CONFIG
# -----------------------------

st.set_page_config(
    page_title="Crypto Command",
    page_icon="logo.png",
    layout="wide"
)

# -----------------------------
# CUSTOM STYLE (BLACK + GOLD)
# -----------------------------

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

# -----------------------------
# LOGO + TITLE
# -----------------------------

st.image("logo.png", width=300)
st.title("Crypto Command")

st.write("AI-powered crypto analytics dashboard")

# -----------------------------
# DOWNLOAD PRICE DATA
# -----------------------------

coins = ["BTC-USD", "ETH-USD"]

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

# -----------------------------
# PRICE CHART
# -----------------------------

fig = go.Figure()

for coin in coins:
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df[coin],
        mode="lines",
        name=coin
    ))

fig.update_layout(
    title="Crypto Price Trends",
    template="plotly_dark"
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# SAMPLE SENTIMENT
# -----------------------------

st.header("Market Sentiment")

sample_news = [
    "Bitcoin surges after institutional investment",
    "Ethereum upgrade improves network efficiency",
    "Crypto regulation fears shake the market",
    "Altcoins rally as Bitcoin stabilizes"
]

sentiment_scores = [TextBlob(x).sentiment.polarity for x in sample_news]

sentiment_df = pd.DataFrame({
    "headline": sample_news,
    "sentiment": sentiment_scores
})

st.dataframe(sentiment_df)

# -----------------------------
# PORTFOLIO SIMULATOR
# -----------------------------

st.header("Portfolio Simulator")

btc_change = st.slider("BTC % Change", -50, 50, 0)
eth_change = st.slider("ETH % Change", -50, 50, 0)

portfolio = {"BTC-USD": 1, "ETH-USD": 5}

new_values = {}

for coin, qty in portfolio.items():

    pct = btc_change/100 if coin == "BTC-USD" else eth_change/100

    new_values[coin] = qty * df[coin].iloc[-1] * (1 + pct)

total = sum(new_values.values())

st.metric("Simulated Portfolio Value", f"${total:,.2f}")

sim_df = pd.DataFrame(list(new_values.items()), columns=["Coin","Value"])

st.bar_chart(sim_df.set_index("Coin"))
