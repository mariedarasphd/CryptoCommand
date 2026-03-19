# BLOOMBERG-STYLE UI UPGRADE: CryptoCommand

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import requests
import os
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Crypto Command", page_icon="🚀", layout="wide")

# -----------------------------
# 🎨 TERMINAL-STYLE THEME
# -----------------------------
st.markdown("""
<style>
    .stApp {
        background-color: #000000;
        color: #FFD700;
        font-family: 'Courier New', monospace;
    }
    h1, h2, h3 {
        color: #FFD700;
        letter-spacing: 1px;
    }
    [data-testid="stSidebar"] {
        background-color: #050505;
        border-right: 1px solid #222;
    }
    .stMetric {
        background-color: #111;
        padding: 10px;
        border-radius: 10px;
        border: 1px solid #333;
    }
    .stButton>button {
        background-color: #FFD700;
        color: black;
        font-weight: bold;
        border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# DATA FETCH
# -----------------------------

def fetch_coin_data(coin, retries=3):
    for _ in range(retries):
        try:
            data = yf.download(coin, start="2022-01-01", interval="1d", auto_adjust=True, progress=False, threads=False)
            if not data.empty:
                return data
        except:
            pass
        time.sleep(2)
    return pd.DataFrame()

@st.cache_data
def load_price_data():
    coins = ["BTC-USD", "ETH-USD"]
    frames = []

    for coin in coins:
        data = fetch_coin_data(coin)

        if data.empty and coin == "BTC-USD":
            dates = pd.date_range(start="2022-01-01", periods=500)
            prices = np.cumsum(np.random.normal(0, 1, 500)) + 20000
            data = pd.DataFrame({"Close": prices}, index=dates)

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(-1)

        series = data['Close'] if 'Close' in data.columns else data.iloc[:, 0]
        frames.append(series.rename(coin).to_frame())

    return pd.concat(frames, axis=1).ffill()

# -----------------------------
# FEATURES + MODEL
# -----------------------------

@st.cache_data
def generate_features(df):
    df = df.copy()
    returns = df['BTC-USD'].pct_change().fillna(0)

    sentiment = pd.Series(np.random.normal(0, 0.3, len(df)), index=df.index)
    sentiment += 0.3 * returns
    sentiment = np.clip(sentiment, -1, 1)

    df['Sentiment'] = sentiment
    df['Volume_Proxy'] = np.abs(returns) * 100
    df['Momentum_5D'] = df['BTC-USD'].rolling(5).mean().pct_change().fillna(0)
    df['Momentum_20D'] = df['BTC-USD'].rolling(20).mean().pct_change().fillna(0)

    df['Target'] = pd.cut(returns, bins=[-1, -0.01, 0.01, 1], labels=['Down', 'Stable', 'Up'])

    return df.dropna(), sentiment


def train_model(df):
    if df.empty:
        return None, None

    X = df[['Sentiment', 'Volume_Proxy', 'Momentum_5D', 'Momentum_20D']]
    y = df['Target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    clf = DecisionTreeClassifier(max_depth=5, min_samples_leaf=10)
    clf.fit(X_train, y_train)

    return clf, accuracy_score(y_test, clf.predict(X_test))

# -----------------------------
# LOAD
# -----------------------------

st.title("CRYPTO COMMAND TERMINAL")

with st.spinner("Initializing market engine..."):
    df_raw = load_price_data()

if df_raw.empty or 'BTC-USD' not in df_raw.columns:
    st.error("DATA FEED ERROR: BTC unavailable")
    st.stop()


df_clean, sentiment = generate_features(df_raw)
clf, acc = train_model(df_clean)

# -----------------------------
# TOP DASHBOARD STRIP (BLOOMBERG STYLE)
# -----------------------------

latest_btc = df_raw['BTC-USD'].iloc[-1]
latest_eth = df_raw['ETH-USD'].iloc[-1]

col1, col2, col3 = st.columns(3)
col1.metric("BTC-USD", f"${latest_btc:,.0f}")
col2.metric("ETH-USD", f"${latest_eth:,.0f}")
col3.metric("MODEL ACCURACY", f"{acc*100:.1f}%" if acc else "N/A")

# -----------------------------
# MAIN GRID
# -----------------------------

left, right = st.columns([2,1])

with left:
    st.subheader("MARKET GRAPH")
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df_raw.index, y=df_raw['BTC-USD'], name='BTC', line=dict(width=2)))
    fig.add_trace(go.Scatter(x=df_raw.index, y=df_raw['ETH-USD'], name='ETH', line=dict(width=2)))

    if sentiment is not None:
        fig.add_trace(go.Scatter(x=df_raw.index, y=sentiment, name='Sentiment', yaxis='y2', line=dict(dash='dot')))

    fig.update_layout(template="plotly_dark", height=500,
                      yaxis2=dict(overlaying='y', side='right'))

    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("SIGNAL ENGINE")

    s1 = st.slider("Sentiment", -1.0, 1.0, 0.0)
    s2 = st.slider("Volume", 0.0, 5.0, 1.0)
    s3 = st.slider("Momentum 5D", -0.1, 0.1, 0.0)
    s4 = st.slider("Momentum 20D", -0.1, 0.1, 0.0)

    if clf:
        pred = clf.predict([[s1, s2, s3, s4]])[0]
        color = "green" if pred == "Up" else "red" if pred == "Down" else "gray"
        st.markdown(f"## SIGNAL: :{color}[{pred}]")

# -----------------------------
# FOOTER
# -----------------------------

st.markdown("---")
st.caption("SYSTEM STATUS: OPERATIONAL | DEMO MODE | NOT FINANCIAL ADVICE")
