import requests
import plotly.express as px

# -----------------------------
# LIVE CRYPTO MARKET HEATMAP
# -----------------------------

st.header("Crypto Market Heatmap")

url = "https://api.coingecko.com/api/v3/coins/markets"

params = {
    "vs_currency": "usd",
    "order": "market_cap_desc",
    "per_page": 20,
    "page": 1,
    "sparkline": False
}

data = requests.get(url, params=params).json()

coins = []
prices = []
changes = []

for coin in data:
    coins.append(coin["symbol"].upper())
    prices.append(coin["current_price"])
    changes.append(coin["price_change_percentage_24h"])

heatmap_df = pd.DataFrame({
    "Coin": coins,
    "Price": prices,
    "24h Change": changes
})

fig = px.treemap(
    heatmap_df,
    path=["Coin"],
    values="Price",
    color="24h Change",
    color_continuous_scale=["red", "black", "gold"]
)

st.plotly_chart(fig, use_container_width=True)
