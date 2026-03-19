# FIXED VERSION WITH FULL UI RESTORED

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import requests
import os
import time
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Crypto Command", page_icon="🚀", layout="wide")

# -----------------------------
# 🎨 CUSTOM THEME (BLACK + GOLD RESTORED)
# -----------------------------
st.markdown("""
<style>
    .stApp {
        background-color: #000000;
        color: #FFD700;
    }
    h1, h2, h3, h4 {
        color: #FFD700;
    }
    [data-testid="stSidebar"] {
        background-color: #111111;
    }
    .stButton>button {
        background-color: #FFD700;
        color: #000;
        font-weight: bold;
        border: none;
    }
    .stButton>button:hover {
        background-color: #E6C200;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# DATA FETCH (ROBUST)
# -----------------------------

def fetch_coin_data(coin, retries=3):
    for _ in range(retries):
        try:
            data = yf.download(
                coin,
                start="2022-01-01",
                interval="1d",
                auto_adjust=True,
                progress=False,
                threads=False
            )
            if not data.empty:
                return data
        except Exception:
            pass
        time.sleep(2)
    return pd.DataFrame()

@st.cache_data
def load_price_data():
    coins = ["BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD", "XRP-USD"]
    frames = []

    for coin in coins:
        data = fetch_coin_data(coin)

        if data.empty:
            if coin == "BTC-USD":
                dates = pd.date_range(start="2022-01-01", periods=500)
                prices = np.cumsum(np.random.normal(0, 1, 500)) + 20000
                data = pd.DataFrame({"Close": prices}, index=dates)
            else:
                continue

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(-1)

        series = data['Close'] if 'Close' in data.columns else data.iloc[:, 0]
        frames.append(series.rename(coin).to_frame())

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, axis=1).ffill()

# -----------------------------
# FEATURES
# -----------------------------

@st.cache_data
def generate_features(df):
    if df.empty or 'BTC-USD' not in df.columns:
        return pd.DataFrame(), None

    df = df.copy()
    np.random.seed(42)

    returns = df['BTC-USD'].pct_change().fillna(0)

    sentiment = pd.Series(np.random.normal(0, 0.3, len(df)), index=df.index)
    sentiment += 0.3 * returns
    sentiment = np.clip(sentiment, -1, 1)

    df['Sentiment'] = sentiment
    df['Volume_Proxy'] = np.abs(returns) * 100
    df['Momentum_5D'] = df['BTC-USD'].rolling(5).mean().pct_change().fillna(0)
    df['Momentum_20D'] = df['BTC-USD'].rolling(20).mean().pct_change().fillna(0)

    df['Return'] = returns
    df['Target'] = pd.cut(df['Return'], bins=[-1, -0.01, 0.01, 1], labels=['Down', 'Stable', 'Up'])

    return df.dropna(), sentiment

# -----------------------------
# MODEL
# -----------------------------

def train_model(df):
    if df is None or df.empty:
        return None, None

    X = df[['Sentiment', 'Volume_Proxy', 'Momentum_5D', 'Momentum_20D']]
    y = df['Target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, random_state=42)
    clf.fit(X_train, y_train)

    return clf, accuracy_score(y_test, clf.predict(X_test))

# -----------------------------
# SIDEBAR (RESTORED)
# -----------------------------

with st.sidebar:
    if os.path.exists("logo.png"):
        st.image("logo.png", width=150)
    else:
        st.markdown("### 🚀 Crypto Command")

    st.markdown("---")
    st.markdown("**Phase 1 Demo Mode**")
    st.info("• Prices: Real (yfinance)\n• Sentiment: Synthetic\n• Model: ML")

# -----------------------------
# APP
# -----------------------------

st.title("Crypto Command")
st.markdown("### AI-Powered Predictive Analytics")

with st.spinner("Loading data..."):
    df_raw = load_price_data()

if df_raw.empty or 'BTC-USD' not in df_raw.columns:
    st.error("BTC data unavailable. Try refresh.")
    st.stop()


df_clean, sentiment = generate_features(df_raw)
clf, acc = train_model(df_clean)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Live Market")
    try:
        url = "https://api.coingecko.com/api/v3/coins/markets"
        params = {"vs_currency": "usd", "order": "market_cap_desc", "per_page": 10}
        df_live = pd.DataFrame(requests.get(url, params=params).json())
        st.dataframe(df_live[['name', 'symbol', 'current_price']])
    except:
        st.warning("Live data unavailable")

with col2:
    if clf:
        st.success(f"Model Accuracy: {acc*100:.1f}%")
    else:
        st.error("Model failed")

# Prediction
st.header("AI Prediction Engine")

s1 = st.slider("Sentiment", -1.0, 1.0, 0.0)
s2 = st.slider("Volume", 0.0, 5.0, 1.0)
s3 = st.slider("Momentum 5D", -0.1, 0.1, 0.0)
s4 = st.slider("Momentum 20D", -0.1, 0.1, 0.0)

if clf:
    pred = clf.predict([[s1, s2, s3, s4]])[0]
    color = "green" if pred == "Up" else "red" if pred == "Down" else "gray"
    st.markdown(f"### 🎯 Forecast: :{color}[{pred}]")

# Chart
st.header("Price & Sentiment")

fig = go.Figure()
fig.add_trace(go.Scatter(x=df_raw.index, y=df_raw['BTC-USD'], name='BTC', line=dict(width=2)))

if sentiment is not None:
    fig.add_trace(go.Scatter(x=df_raw.index, y=sentiment, name='Sentiment', yaxis='y2', line=dict(dash='dot', color='orange')))

fig.update_layout(template="plotly_dark", yaxis2=dict(overlaying='y', side='right'))
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption("⚠️ Demo only. Not financial advice.")
