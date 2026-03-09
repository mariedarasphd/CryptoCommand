import streamlit as st
import yfinance as yf
import pandas as pd
from textblob import TextBlob
import plotly.express as px
from PIL import Image

# --- Page config ---
st.set_page_config(
    page_title="Crypto Command",
    page_icon="logo.png",
    layout="wide"
)

# --- Custom black & gold theme ---
st.markdown("""
    <style>
    body { background-color: #000000; color: #FFD700; }
    .stButton>button { background-color: #FFD700; color: black; font-weight: bold; }
    .stSlider>div>div>div>div { color: #FFD700; }
    .stTextInput>div>div>input { background-color: #111111; color: #FFD700; }
    .stMarkdown { color: #FFD700; }
    .stHeader, h1, h2, h3 { color: #FFD700; }
    </style>
""", unsafe_allow_html=True)

# --- Logo ---
logo = Image.open("logo.png")
st.image(logo, width=250)

st.title("Crypto Command")
st.markdown("Dashboard for cryptocurrency insights, sentiment, and simulations.")

# --- Fetch historical price data ---
coins = ["BTC-USD", "ETH-USD"]
price_frames = []

for coin in coins:
    data = yf.download(coin, start="2022-01-01", end="2026-03-01", auto_adjust=True, interval="1d")
    if data.empty:
        st.warning(f"No data for {coin}")
    else:
        df_coin = data[['Close']].rename(columns={'Close': coin})
        price_frames.append(df_coin)

df = pd.concat(price_frames, axis=1).ffill()

# Flatten columns if MultiIndex exists
if isinstance(df.columns, pd.MultiIndex):
    df.columns = [col[-1] for col in df.columns]

# --- Melt for Plotly ---
df_plot = df.reset_index().melt(
    id_vars='Date',
    value_vars=df.columns.tolist(),
    var_name='Coin',
    value_name='Price'
)

# --- Historical Prices Plot ---
fig_prices = px.line(
    df_plot,
    x='Date',
    y='Price',
    color='Coin',
    labels={'Price':'Price (USD)', 'Date':'Date'},
    title="BTC & ETH Historical Prices"
)
st.plotly_chart(fig_prices, use_container_width=True)

# --- Sentiment Analysis ---
st.header("Sentiment Analysis")
user_text = st.text_area("Paste news article or social media text here for sentiment scoring:", "")
if user_text:
    blob = TextBlob(user_text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    st.markdown(f"**Polarity:** {polarity:.2f} (negative to positive)")
    st.markdown(f"**Subjectivity:** {subjectivity:.2f} (objective to subjective)")
else:
    st.markdown("Enter text above to perform sentiment analysis.")

# --- Simulation Placeholder ---
st.header("Simulation")
forecast_days = st.slider("Forecast Days", 1, 30, 7)
st.markdown(f"Simulation for next {forecast_days} days will appear here once streaming data services are enabled.")

# --- Footer ---
st.markdown("---")
st.markdown("Crypto Command © 2026 | Black & Gold Theme")
