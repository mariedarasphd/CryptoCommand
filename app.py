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

# --- CUSTOM STYLE: BLACK/GOLD ---
st.markdown(
    """
    <style>
    body {background-color: black; color: gold;}
    .stApp {background-color: black; color: gold;}
    .stButton>button {background-color: gold; color: black; font-weight: bold;}
    .stSlider>div>div>div>div {color: gold;}
    .stTable {color: gold; background-color: black;}
    .stDataFrame thead th {color: gold; background-color: black;}
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {color: gold;}
    </style>
    """,
    unsafe_allow_html=True
)

# --- LOGO ---
st.image("logo.png", width=250)

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

# Flatten columns if MultiIndex
if isinstance(df.columns, pd.MultiIndex):
    df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in df.columns]

df = df.ffill()

# --- PRICE PLOT ---
df_plot = df.reset_index().melt(
    id_vars='Date',
    value_vars=df.columns.tolist(),
    var_name='Coin',
    value_name='Price'
)

fig_prices = px.line(
    df_plot,
    x='Date',
    y='Price',
    color='Coin',
    title="BTC & ETH Historical Prices",
    labels={'Price': 'Price (USD)', 'Date': 'Date', 'Coin': 'Coin'},
    template='plotly_dark'
)
fig_prices.update_layout(
    plot_bgcolor='black',
    paper_bgcolor='black',
    font_color='gold'
)
st.plotly_chart(fig_prices, use_container_width=True)

# --- SENTIMENT ANALYSIS ---
st.header("Crypto News Sentiment")
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

st.dataframe(df_sentiment.style.set_properties(**{'color': 'gold', 'background-color': 'black'}))

# --- SIMULATION PLACEHOLDER ---
st.header("Simulation Placeholder")
days = st.slider("Forecast Days", min_value=1, max_value=30, value=7)
st.markdown(f"<p style='color:gold;'>Simulation for next {days} days will appear here once streaming data services are enabled.</p>", unsafe_allow_html=True)

# --- FOOTER ---
st.markdown("<br><br><p style='color:gold;'>Crypto Command © 2026</p>", unsafe_allow_html=True)
