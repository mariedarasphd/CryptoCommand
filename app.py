import streamlit as st
import yfinance as yf
import pandas as pd
from textblob import TextBlob
import plotly.express as px

# --------------------------
# Page config
# --------------------------
st.set_page_config(
    page_title="Crypto Command",
    page_icon=":money_with_wings:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------
# Custom CSS for black & gold theme
# --------------------------
st.markdown(
    """
    <style>
    .stApp {
        background-color: #000000;
        color: #FFD700;
    }
    .stButton>button {
        background-color: #FFD700;
        color: #000000;
        font-weight: bold;
    }
    .stSlider>div>div>div>div {
        color: #000000;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --------------------------
# Logo
# --------------------------
st.image("logo.png", width=200)

st.title("Crypto Command")
st.write("Historical Prices, Sentiment, and Simulation Preview")

# --------------------------
# Data Fetching
# --------------------------
coins = ["BTC-USD", "ETH-USD"]
price_frames = []

for coin in coins:
    data = yf.download(coin, start="2022-01-01", end="2026-03-01", auto_adjust=True)
    if not data.empty:
        df_coin = data[['Close']].rename(columns={'Close': coin})
        price_frames.append(df_coin)

# Merge into a single DataFrame
df = pd.concat(price_frames, axis=1).ffill()

# --------------------------
# Price Chart
# --------------------------
df_plot = df.reset_index().melt(id_vars='Date', value_vars=df.columns,
                                var_name='Coin', value_name='Price')

fig_prices = px.line(
    df_plot, x='Date', y='Price', color='Coin',
    title="BTC & ETH Historical Prices",
    labels={'Price': 'Price (USD)'}
)
st.plotly_chart(fig_prices, use_container_width=True)

# --------------------------
# Sentiment Analysis (mock)
# --------------------------
st.subheader("Crypto Sentiment Analysis")
sample_news = [
    "Bitcoin surges as institutional investors pour in.",
    "Ethereum struggles amid regulatory concerns.",
    "Market volatility continues, traders cautious."
]

sentiments = []
for text in sample_news:
    blob = TextBlob(text)
    sentiments.append({'text': text, 'polarity': blob.sentiment.polarity})

sentiment_df = pd.DataFrame(sentiments)
st.table(sentiment_df)

# --------------------------
# Simulation Preview
# --------------------------
st.subheader("Simulation Placeholder")
forecast_days = st.slider("Forecast Days", min_value=1, max_value=30, value=7)

st.write(
    f"Simulation for next {forecast_days} days will appear here once streaming data services are enabled."
)
