import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_FILE = SCRIPT_DIR / "nifty_daily_data.csvnot in "

parser = argparse.ArgumentParser(description="Analyze NIFTY daily market data.")
parser.add_argument(
    "input_file",
    nargs="?",
    type=Path,
    help="CSV file with Date and Close columns (relative paths are resolved beside this script)",
)
args = parser.parse_args()
DATA_FILE = (
    args.input_file
    if args.input_file is not None and args.input_file.is_absolute()
    else SCRIPT_DIR / args.input_file
    if args.input_file is not None
    else DEFAULT_DATA_FILE
)


def load_market_data(path: Path, download_if_missing: bool = True) -> pd.DataFrame:
    if not path.exists() and not download_if_missing:
        raise FileNotFoundError(f"Input CSV not found: {path}")

    if path.exists():
        try:
            df = pd.read_csv(path)
            if not df.empty and {"Date", "Close"}.issubset(df.columns):
                df = df.copy()
                df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
                df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
                valid_rows = df[["Date", "Close"]].notna().all(axis=1)
                if valid_rows.any():
                    return df.loc[valid_rows].reset_index(drop=True)
        except pd.errors.ParserError as exc:
            if not download_if_missing:
                raise ValueError(f"{path.name} could not be parsed as a CSV.") from exc

    if not download_if_missing:
        raise ValueError(f"{path.name} must be a CSV containing valid Date and Close columns.")

    try:
        market = yf.download("^NSEI", period="1y", interval="1d", auto_adjust=False, progress=False, actions=False)
        if market.empty:
            raise ValueError("Yahoo Finance returned no data.")

        market = market.copy()
        if isinstance(market.columns, pd.MultiIndex):
            market.columns = [col[0] if isinstance(col, tuple) else col for col in market.columns]

        df = market.reset_index().rename(columns={"index": "Date"})
        df["Date"] = pd.to_datetime(df["Date"])
        df = df[["Date", "Close"]].dropna().reset_index(drop=True)
        df.to_csv(path, index=False)
        return df
    except Exception as exc:
        raise ValueError(
            f"{path.name} is not a valid CSV export and live data could not be downloaded. "
            "Please save a valid NIFTY 50 historical CSV as dataset.csv or ensure internet access."
        ) from exc


df = load_market_data(DATA_FILE, download_if_missing=args.input_file is None)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").dropna(subset=["Close"]).reset_index(drop=True)

df["Return"] = df["Close"].pct_change() * 100
df = df.dropna(subset=["Return"]).reset_index(drop=True)


def classify_market(r):
    if r > 0.5:
        return "Up"
    elif r < -0.5:
        return "Down"
    return "Stable"


df["State"] = df["Return"].apply(classify_market)
print(df["State"].value_counts())

states = ["Up", "Stable", "Down"]
counts = pd.DataFrame(0, index=states, columns=states)
for i in range(len(df) - 1):
    current = df.loc[i, "State"]
    next_state = df.loc[i + 1, "State"]
    counts.loc[current, next_state] += 1
row_totals = counts.sum(axis=1).replace(0, np.nan)
transition = counts.div(row_totals, axis=0).fillna(0)
print(transition.round(3))

start_price = float(df["Close"].iloc[-1])
returns_by_state = {
    s: df.loc[df["State"] == s, "Return"].to_numpy() / 100
    for s in states
}
for s in states:
    if len(returns_by_state[s]) == 0:
        raise ValueError(f"No returns available for {s}")


def simulate_path(days=30):
    current = df["State"].iloc[-1]
    price = start_price
    path = [price]
    for _ in range(days):
        probs = transition.loc[current].to_numpy()
        if not np.isclose(probs.sum(), 1.0):
            raise ValueError(f"Incomplete row: {current}")
        nxt = np.random.choice(states, p=probs)
        r = np.random.choice(returns_by_state[nxt])
        price *= (1 + r)
        path.append(price)
        current = nxt
    return path


paths = [simulate_path(30) for _ in range(100)]

plt.figure(figsize=(9, 4))
plt.plot(df["Date"], df["Close"])
plt.title("NIFTY 50 Historical Closing Prices")
plt.xlabel("Date")
plt.ylabel("Index value")
plt.tight_layout()
plt.savefig(SCRIPT_DIR / "nifty_historical_prices.png", dpi=150)
plt.close()

plt.figure(figsize=(9, 4))
for p in paths:
    plt.plot(range(len(p)), p, alpha=0.25)
plt.title("Monte Carlo Simulated Price Paths")
plt.xlabel("Future trading days")
plt.ylabel("Simulated index value")
plt.tight_layout()
plt.savefig(SCRIPT_DIR / "nifty_simulated_paths.png", dpi=150)
plt.close()