"""Load and filter mandi (market) price data.

Prices are in INR per quintal. Rows with source == "DEMO" are sample data
and must be labelled as demo data in the user interface.
"""
import pandas as pd

REQUIRED_COLUMNS = [
    "state", "district", "market", "commodity", "variety",
    "min_price", "max_price", "modal_price", "arrival_date", "source",
]


def load_mandi_data(path="data/mandi_sample.csv"):
    """Read the price file and check that all required columns exist."""
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in price file: {missing}")
    df["arrival_date"] = pd.to_datetime(df["arrival_date"])
    return df


def get_prices(df, commodity, state=None):
    """Return prices for one commodity, highest modal price first."""
    result = df[df["commodity"].str.lower() == commodity.lower()]
    if state:
        result = result[result["state"].str.lower() == state.lower()]
    return result.sort_values("modal_price", ascending=False).reset_index(drop=True)