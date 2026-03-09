# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from textblob import TextBlob
import plotly.express as px
from datetime import datetime, timedelta

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="Crypto Command",
    page_icon=":moneybag:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------
# STYLE
# ----------------------------
st.markdown(
    """
    <style>
    body {
        background-color: #000000;
        color: #FFD700;
    }
    .stButton>button {
        background-color: #FFD700;
        color: black;
    }
    .stSlider>div>div>div>div {
        color: #FFD700;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------------------
# LOGO
# ----------------------------
st.image("logo.png", width=300)

st.title("Crypto Command Dashboard")

# ----------------------------
# DATA
# ----------------------------
coins = ["BTC-USD", "ETH-USD"]
price_frames = []

for coin in coins:
    data = yf.download(coin, start="2022-01-01", end=datetime.today().strftime("%Y-%m-%d"), auto_adjust=True)
    if data.empty:
        st.warning(f"No data found for {coin}")
    else:
        df_coin = data[['Close']].rename(columns={'Close': coin})
        price_frames.append(df_coin)

df = pd.concat(price_frames, axis=1)
df = df.ffill()  # fill missing values

# ----------------------------
# PLOT HISTORICAL PRICES
# ----------------------------
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
fig_prices.update_layout(plot_bgcolor='black', paper_bgcolor='black', font_color='gold')

st.plotly_chart(fig_prices, use_container_width=True)

# ----------------------------
# SENTIMENT ANALYSIS
# ----------------------------
st.subheader("Sentiment Analysis Placeholder")

sample_text = st.text_area(
    "Enter cryptocurrency news text or social media text here to score sentiment:",
    "Bitcoin is surging and investors are excited..."
)

if sample_text:
    blob = TextBlob(sample_text)
    sentiment_score = blob.sentiment.polarity
    st.write(f"Sentiment score: {sentiment_score:.2f}")
    if sentiment_score > 0:
        st.success("Positive sentiment")
    elif sentiment_score < 0:
        st.error("Negative sentiment")
    else:
        st.info("Neutral sentiment")

# ----------------------------
# SIMULATION
# ----------------------------
st.subheader("Monte Carlo Simulation (Placeholder)")

forecast_days = st.slider("Forecast Days", min_value=1, max_value=30, value=7)

st.info(
    f"Simulation for next {forecast_days} days will appear here once streaming/live data services are enabled.\n\n"
    "Currently using simple Monte Carlo simulation based on historical volatility."
)

simulate_button = st.button("Run Simulation (Sample)")

if simulate_button:
    # Simple Monte Carlo
    last_prices = df.iloc[-1]
    num_simulations = 10
    simulation_results = pd.DataFrame(index=range(forecast_days))

    for coin in coins:
        daily_returns = df[coin].pct_change().dropna()
        mu = daily_returns.mean()
        sigma = daily_returns.std()

        simulations = []
        for _ in range(num_simulations):
            price_series = [last_prices[coin]]
            for _ in range(forecast_days):
                price_series.append(price_series[-1] * (1 + np.random.normal(mu, sigma)))
            simulations.append(price_series[1:])
        simulation_results[coin] = np.mean(simulations, axis=0)

    # Plot simulation
    sim_plot = px.line(
        simulation_results,
        y=coins,
        labels={'value': 'Simulated Price (USD)', 'index': 'Forecast Day', 'variable': 'Coin'},
        title="Monte Carlo Simulation Forecast"
    )
    sim_plot.update_layout(plot_bgcolor='black', paper_bgcolor='black', font_color='gold')
    st.plotly_chart(sim_plot, use_container_width=True)
