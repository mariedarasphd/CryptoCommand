import streamlit as st
import yfinance as yf
import pandas as pd
from textblob import TextBlob
import plotly.express as px

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Crypto Command",
    page_icon="logo.png",
    layout="wide",
)

# --- LOGO ---
st.image("logo.png", width=250)

# --- STYLE ---
st.markdown(
    """
    <style>
    body {background-color: black; color: gold;}
    .stButton>button {background-color: gold; color: black;}
    .stSlider>div>div>div>div {color: gold;}
    </style>
    """,
    unsafe_allow_html=True
)

# --- DATA FETCH ---
coins = ["BTC-USD", "ETH-USD"]
price_frames = []

for coin in coins:
    data = yf.download(coin, start="2022-01-01", end="2026-03-01", auto_adjust=True, interval='1d')
    if data.empty:
        st.warning(f"No data for {coin}")
    else:
        df_coin = data[['Close']].rename(columns={'Close': coin})
        price_frames.append(df_coin)

df = pd.concat(price_frames, axis=1)
df = df.ffill()  # forward-fill missing values

# --- PRICE PLOT ---
df_plot = df.reset_index().melt(id_vars='Date', value_vars=df.columns,
                                var_name='Coin', value_name='Price')

fig_prices = px.line(
    df_plot,
    x='Date',
    y='Price',
    color='Coin',
    title="BTC & ETH Historical Prices",
    labels={'Price': 'Price (USD)', 'Date': 'Date', 'Coin': 'Coin'}
)
st.plotly_chart(fig_prices, use_container_width=True)

# --- SENTIMENT ANALYSIS ---
st.header("Example Crypto News Sentiment")
example_news = [
    "Bitcoin hits new all-time high",
    "Ethereum network faces congestion issues",
    "Crypto market volatility continues"
]

sentiments = [TextBlob(n).sentiment.polarity for n in example_news]
df_sentiment = pd.DataFrame({
    "News": example_news,
    "Sentiment": sentiments
})

st.table(df_sentiment)

# --- SIMULATION PLACEHOLDER ---
st.header("Simulation Placeholder")
days = st.slider("Forecast Days", min_value=1, max_value=30, value=7)
st.write(f"Simulation for next {days} days will appear here once streaming data services are enabled.")

# --- END ---
st.markdown("<br><br><p style='color:gold;'>Crypto Command © 2026</p>", unsafe_allow_html=True)
