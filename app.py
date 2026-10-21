"""MandiMitra AI - Mandi comparison and net-return calculator."""
from dataclasses import asdict

import pandas as pd
import streamlit as st

from modules.mandi_data import get_prices, load_mandi_data
from modules.net_return import MarketOption, compare_markets

st.set_page_config(page_title="MandiMitra AI", page_icon="🌾", layout="wide")
st.title("🌾 MandiMitra AI")
st.caption("Diagnose Smarter. Sell Better. Farm Sustainably.")

df = load_mandi_data()

if (df["source"] == "DEMO").any():
    st.warning("Showing DEMO data. These are sample prices, not real market prices.")

st.header("Mandi Price Comparison")

col1, col2 = st.columns(2)
with col1:
    commodity = st.selectbox("Crop", sorted(df["commodity"].unique()))
with col2:
    quantity = st.number_input("Quantity (quintals)", min_value=1.0, value=20.0, step=1.0)

prices = get_prices(df, commodity)
market_names = prices["market"].tolist()

st.subheader("Reported prices")
st.dataframe(
    prices[["market", "district", "min_price", "max_price", "modal_price", "arrival_date", "source"]],
    use_container_width=True,
)

st.subheader("Net-return calculator")
selected = st.multiselect("Markets to compare", market_names, default=market_names[:2])

col3, col4 = st.columns(2)
with col3:
    handling = st.number_input("Handling cost per quintal (INR)", min_value=0.0, value=10.0)
with col4:
    commission = st.number_input("Commission (%)", min_value=0.0, max_value=100.0, value=1.0)

options = []
for market in selected:
    row = prices[prices["market"] == market].iloc[0]
    transport = st.number_input(
        f"Transport cost to {market} (INR, total)",
        min_value=0.0, value=1500.0, step=100.0, key=f"transport_{market}",
    )
    options.append(
        MarketOption(market, float(row["modal_price"]), transport, handling, commission)
    )

if options:
    results = compare_markets(quantity, options)
    table = pd.DataFrame([asdict(r) for r in results])
    table = table.rename(columns={
        "market": "Market",
        "price_per_quintal": "Price (INR/quintal)",
        "gross_sale": "Gross sale (INR)",
        "total_costs": "Total costs (INR)",
        "net_return": "Net return (INR)",
    })
    st.dataframe(
        table[["Market", "Price (INR/quintal)", "Gross sale (INR)",
               "Total costs (INR)", "Net return (INR)"]],
        use_container_width=True,
    )
    st.bar_chart(table.set_index("Market")["Net return (INR)"])

    best = results[0]
    st.success(f"Highest estimated net return: {best.market} (INR {best.net_return:,.0f})")
    st.caption(
        "Estimates only. Actual returns depend on quality, buyer price, "
        "transport and commission. Edit the costs above to match your situation."
    )
else:
    st.info("Select at least one market to compare.")