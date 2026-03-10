import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import requests
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings

# Suppress warnings for cleaner UI
warnings.filterwarnings('ignore')

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Crypto Command",
    page_icon="🚀", # Using emoji as fallback if logo.png missing
    layout="wide"
)

# -----------------------------
# CUSTOM THEME (Black + Gold)
# -----------------------------
st.markdown("""
<style>
    .stApp {
        background-color: #000000;
        color: #FFD700; /* Gold */
    }
    h1, h2, h3, h4, h5, h6 {
        color: #FFD700;
        font-family: 'Helvetica Neue', sans-serif;
    }
    .metric-value {
        color: #FFD700 !important;
    }
    [data-testid="stSidebar"] {
        background-color: #111111;
        border-right: 1px solid #333;
    }
    .css-1r6slb0 {
        color: #FFD700;
    }
    /* Custom button style */
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
# HELPER FUNCTIONS
# -----------------------------

@st.cache_data
def load_price_data():
    """Fetches historical price data for demo coins."""
    coins = ["BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD", "XRP-USD"]
    price_frames = []
    
    for coin in coins:
        try:
            data = yf.download(coin, start="2022-01-01", auto_adjust=True, interval="1d", progress=False)
            if not data.empty:
                df_coin = data[['Close']].rename(columns={'Close': coin})
                price_frames.append(df_coin)
        except Exception as e:
            st.warning(f"Could not load {coin}: {e}")
            
    if not price_frames:
        st.error("No price data loaded. Check internet connection.")
        return pd.DataFrame()
        
    df = pd.concat(price_frames, axis=1).ffill()
    return df

@st.cache_data
def generate_synthetic_features(df):
    """Generates synthetic sentiment and features for the C5.0 model."""
    if df.empty:
        return df, None
        
    df = df.copy()
    np.random.seed(42)
    
    # 1. Create synthetic sentiment correlated with price returns
    price_returns = df['BTC-USD'].pct_change().fillna(0)
    base_sentiment = np.random.normal(0, 0.3, len(df))
    sentiment_correlation = 0.3 * price_returns
    sentiment_series = base_sentiment + sentiment_correlation
    sentiment_series = np.clip(sentiment_series, -1, 1)
    
    df['Sentiment'] = sentiment_series
    
    # 2. Create other features
    df['Volume_Proxy'] = np.abs(price_returns) * 100
    df['Momentum_5D'] = df['BTC-USD'].rolling(window=5).mean().pct_change().fillna(0)
    df['Momentum_20D'] = df['BTC-USD'].rolling(window=20).mean().pct_change().fillna(0)
    
    # 3. Create Target (Direction)
    df['Return'] = df['BTC-USD'].pct_change().fillna(0)
    df['Target'] = pd.cut(df['Return'], bins=[-1, -0.01, 0.01, 1], labels=['Down', 'Stable', 'Up'])
    
    # Drop NaNs
    df_clean = df.dropna(subset=['Sentiment', 'Volume_Proxy', 'Momentum_5D', 'Momentum_20D', 'Target'])
    
    return df_clean, sentiment_series

def train_model(df_clean):
    """Trains the C5.0-style Decision Tree."""
    if df_clean is None or len(df_clean) < 10:
        return None, None
        
    feature_cols = ['Sentiment', 'Volume_Proxy', 'Momentum_5D', 'Momentum_20D']
    X = df_clean[feature_cols]
    y = df_clean['Target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # C5.0 Proxy: DecisionTreeClassifier with limited depth for interpretability
    clf = DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, random_state=42)
    clf.fit(X_train, y_train)
    
    acc = accuracy_score(y_test, clf.predict(X_test))
    return clf, acc

def explain_prediction(clf, features_dict, feature_names):
    """Generates human-readable explanation from the tree path."""
    if clf is None:
        return "Model not trained yet."
        
    try:
        # Get prediction and confidence
        pred = clf.predict([list(features_dict.values())])[0]
        proba = clf.predict_proba([list(features_dict.values())]).max()
        
        # Get rules
        rules = export_text(clf, feature_names=feature_names)
        
        explanation = f"""
        **Prediction:** {pred}
        **Confidence:** {proba*100:.1f}%
        
        **Key Factors (Decision Path):**
        {rules[:400]} 
        
        **Interpretation:**
        • Sentiment > 0.5 suggests bullish market mood.
        • Momentum > 0 indicates upward price pressure.
        • High volume proxy confirms trader interest.
        """
        return explanation
    except Exception as e:
        return f"Error generating explanation: {str(e)}"

def fetch_live_market_data():
    """Fetches live market data from CoinGecko."""
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {"vs_currency": "usd", "order": "market_cap_desc", "per_page": 20, "page": 1}
    try:
        response = requests.get(url, params=params, timeout=5)
        data = response.json()
        df = pd.DataFrame(data)
        if 'name' in df.columns:
            return df[['name', 'symbol', 'current_price', 'market_cap', 'total_volume', 'price_change_percentage_24h']]
    except Exception:
        pass
    return pd.DataFrame()

# -----------------------------
# MAIN APP LOGIC
# -----------------------------

# Sidebar
with st.sidebar:
    st.image("logo.png", width=150) if os.path.exists("logo.png") else st.markdown("### 🚀 Crypto Command")
    st.markdown("---")
    st.markdown("**Phase 1 Demo Mode**")
    st.info("• Prices: Real (yfinance)\n• Sentiment: Synthetic (Demo)\n• Model: C5.0-Style Decision Tree")
    
    st.markdown("---")
    st.markdown("### Controls")
    scenario = st.selectbox("Market Scenario", ["Neutral", "Bullish", "Bearish", "Volatile"])
    st.markdown("---")
    st.caption("Powered by Streamlit & Scikit-Learn")

# Header
st.title("Crypto Command")
st.markdown("### AI-Powered Predictive Analytics & Scenario Simulator")

# Load Data
with st.spinner("Loading market data and training model..."):
    df_raw = load_price_data()
    df_clean, sentiment_series = generate_synthetic_features(df_raw)
    clf, model_acc = train_model(df_clean)

if df_raw.empty:
    st.error("Failed to load data. Please check your connection.")
    st.stop()

# -----------------------------
# SECTION 1: LIVE MARKET DATA
# -----------------------------
col1, col2 = st.columns([2, 1])

with col1:
    st.header("Live Market Overview")
    live_df = fetch_live_market_data()
    if not live_df.empty:
        # Format columns
        live_df['Price'] = live_df['current_price'].apply(lambda x: f"${x:,.2f}")
        live_df['24h Change'] = live_df['price_change_percentage_24h'].apply(lambda x: f"{x:.2f}%")
        live_df['Market Cap'] = live_df['market_cap'].apply(lambda x: f"${x/1e9:.2f}B")
        
        display_cols = ['name', 'symbol', 'Price', 'Market Cap', '24h Change']
        st.dataframe(live_df[display_cols].head(10), use_container_width=True, hide_index=True)
    else:
        st.warning("Live market data unavailable (API limit or network issue).")

with col2:
    st.header("Model Status")
    if clf:
        st.success(f"✅ Model Trained (Accuracy: {model_acc*100:.1f}%)")
        st.markdown("**Algorithm:** C5.0-Style Decision Tree")
        st.markdown("**Features:** Sentiment, Volume, Momentum")
        st.markdown("**Explainability:** Full Decision Path Available")
    else:
        st.error("❌ Model Training Failed")

# -----------------------------
# SECTION 2: PREDICTION & EXPLANATION
# -----------------------------
st.markdown("---")
st.header("🤖 AI Prediction Engine")

col_pred1, col_pred2 = st.columns(2)

with col_pred1:
    st.subheader("Input Parameters (Manual Override)")
    # Default values based on latest data
    latest_sentiment = df_clean['Sentiment'].iloc[-1] if not df_clean.empty else 0.0
    latest_vol = df_clean['Volume_Proxy'].iloc[-1] if not df_clean.empty else 0.0
    latest_mom5 = df_clean['Momentum_5D'].iloc[-1] if not df_clean.empty else 0.0
    latest_mom20 = df_clean['Momentum_20D'].iloc[-1] if not df_clean.empty else 0.0
    
    s_sentiment = st.slider("Sentiment Score (-1 to 1)", -1.0, 1.0, float(latest_sentiment))
    s_vol = st.slider("Volume Proxy (Volatility)", 0.0, 5.0, float(latest_vol))
    s_mom5 = st.slider("5-Day Momentum", -0.1, 0.1, float(latest_mom5))
    s_mom20 = st.slider("20-Day Momentum", -0.1, 0.1, float(latest_mom20))

with col_pred2:
    st.subheader("Prediction & Reasoning")
    if clf:
        features = {
            'Sentiment': s_sentiment,
            'Volume_Proxy': s_vol,
            'Momentum_5D': s_mom5,
            'Momentum_20D': s_mom20
        }
        
        explanation = explain_prediction(clf, features, ['Sentiment', 'Volume_Proxy', 'Momentum_5D', 'Momentum_20D'])
        st.markdown(explanation)
        
        # Visual indicator
        pred_val = clf.predict([[s_sentiment, s_vol, s_mom5, s_mom20]])[0]
        color = "green" if pred_val == "Up" else "red" if pred_val == "Down" else "gray"
        st.markdown(f"### 🎯 Forecast: :{color}[{pred_val}]")
    else:
        st.warning("Model not available.")

# -----------------------------
# SECTION 3: SCENARIO SIMULATOR
# -----------------------------
st.markdown("---")
st.header("🎮 Portfolio Scenario Simulator")

# Portfolio Setup
st.markdown("**Current Holdings:** 1 BTC, 5 ETH")
portfolio = {"BTC-USD": 1, "ETH-USD": 5}

col_sim1, col_sim2 = st.columns(2)

with col_sim1:
    st.subheader("Adjust Market Conditions")
    btc_change = st.slider("BTC % Change", -50, 50, 0)
    eth_change = st.slider("ETH % Change", -50, 50, 0)
    
    # Scenario Multipliers
    scenario_map = {
        "Neutral": {"sentiment_boost": 1.0, "volatility": 1.0},
        "Bullish": {"sentiment_boost": 1.5, "volatility": 0.8},
        "Bearish": {"sentiment_boost": 0.5, "volatility": 1.2},
        "Volatile": {"sentiment_boost": 1.0, "volatility": 1.5}
    }
    multipliers = scenario_map[scenario]
    
    # Apply sentiment adjustment to the simulation
    # If user sets sentiment slider, use that, else use latest
    sim_sentiment = s_sentiment if 's_sentiment' in locals() else 0.0
    sentiment_adj = sim_sentiment * 0.02 * multipliers['sentiment_boost']
    
    # Calculate new values
    new_values = {}
    total_current = 0
    total_new = 0
    
    for coin, qty in portfolio.items():
        current_price = df_raw[coin].iloc[-1]
        total_current += qty * current_price
        
        pct_change = btc_change/100 if coin == "BTC-USD" else eth_change/100
        adjusted_pct = pct_change + sentiment_adj
        
        new_val = qty * current_price * (1 + adjusted_pct)
        new_values[coin] = new_val
        total_new += new_val

with col_sim2:
    st.subheader("Simulation Results")
    st.metric("Current Portfolio Value", f"${total_current:,.2f}")
    st.metric("Simulated Value", f"${total_new:,.2f}", delta=f"{((total_new - total_current)/total_current)*100:.2f}%")
    
    # Chart
    sim_df = pd.DataFrame(list(new_values.items()), columns=["Coin", "Value"])
    fig_bar = px.bar(sim_df, x="Coin", y="Value", color="Coin", 
                     color_discrete_map={"BTC-USD": "#F7931A", "ETH-USD": "#627EEA"},
                     title="Portfolio Composition", template="plotly_dark")
    fig_bar.update_layout(showlegend=False, height=300)
    st.plotly_chart(fig_bar, use_container_width=True)

# -----------------------------
# SECTION 4: PRICE CHART WITH SENTIMENT
# -----------------------------
st.markdown("---")
st.header("📈 Price Trends & Sentiment Overlay")

fig_line = go.Figure()

# Add Price Lines
for coin in ["BTC-USD", "ETH-USD"]:
    if coin in df_raw.columns:
        fig_line.add_trace(go.Scatter(x=df_raw.index, y=df_raw[coin], mode='lines', name=coin, line=dict(width=2)))

# Add Sentiment (Secondary Axis)
if sentiment_series is not None:
    fig_line.add_trace(go.Scatter(
        x=df_raw.index, 
        y=sentiment_series,
        name='Sentiment',
        mode='lines',
        yaxis='y2',
        line=dict(dash='dot', color='orange', width=2)
    ))

fig_line.update_layout(
    title="Crypto Command: Price & Sentiment Analysis",
    xaxis_title="Date",
    yaxis_title="Price (USD)",
    yaxis2=dict(title="Sentiment Polarity (-1 to 1)", overlaying='y', side='right', range=[-1, 1]),
    legend=dict(x=0.01, y=0.99),
    template="plotly_dark",
    height=600
)
st.plotly_chart(fig_line, use_container_width=True)

# Footer
st.markdown("---")
st.caption("⚠️ **Disclaimer:** This is a Phase 1 Demo. Sentiment data is synthetic. Not financial advice.")
