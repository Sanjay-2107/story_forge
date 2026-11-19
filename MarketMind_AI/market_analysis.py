from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

STATES = ["Up", "Stable", "Down"]


def prepare_market_data(frame: pd.DataFrame, threshold_pct: float = 0.5) -> pd.DataFrame:
    """Validate and normalize a CSV dataframe with Date and Close columns."""
    if not np.isfinite(threshold_pct) or threshold_pct < 0:
        raise ValueError("The state threshold must be a finite, non-negative percentage.")

    columns = {str(column).strip().lower(): column for column in frame.columns}
    if "date" not in columns or "close" not in columns:
        raise ValueError("Your CSV must contain columns named Date and Close.")

    data = frame[[columns["date"], columns["close"]]].copy()
    data.columns = ["Date", "Close"]
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data["Close"] = pd.to_numeric(data["Close"], errors="coerce")
    data = data.dropna(subset=["Date", "Close"])
    data = data[np.isfinite(data["Close"]) & (data["Close"] > 0)]
    data = data.drop_duplicates(subset=["Date"], keep="last")
    data = data.sort_values("Date").reset_index(drop=True)

    if len(data) < 2:
        raise ValueError("At least two rows with valid dates and positive closing prices are required.")

    data["Return"] = data["Close"].pct_change() * 100
    data["State"] = pd.Series(
        np.select(
            [data["Return"] > threshold_pct, data["Return"] < -threshold_pct],
            ["Up", "Down"],
            default="Stable",
        ),
        index=data.index,
        dtype="string",
    ).where(data["Return"].notna())
    return data


def analyze_market_data(data: pd.DataFrame) -> dict[str, Any]:
    """Calculate historical metrics and a state-transition probability matrix."""
    returns = data.dropna(subset=["Return", "State"])
    counts = pd.DataFrame(0, index=STATES, columns=STATES, dtype=int)
    for current, following in zip(returns["State"].iloc[:-1], returns["State"].iloc[1:]):
        counts.loc[current, following] += 1

    row_totals = counts.sum(axis=1)
    probabilities = counts.div(row_totals.replace(0, np.nan), axis=0)

    state_counts = returns["State"].value_counts().reindex(STATES, fill_value=0)
    observed_state_probabilities = state_counts / state_counts.sum()
    probabilities = probabilities.fillna(observed_state_probabilities, axis=0)

    close = data["Close"]
    drawdown = close / close.cummax() - 1
    elapsed_years = (data["Date"].iloc[-1] - data["Date"].iloc[0]).days / 365.25
    cagr = (
        (close.iloc[-1] / close.iloc[0]) ** (1 / elapsed_years) - 1
        if elapsed_years > 0
        else None
    )
    metrics = {
        "observations": len(data),
        "start_date": data["Date"].iloc[0],
        "end_date": data["Date"].iloc[-1],
        "start_close": float(close.iloc[0]),
        "latest_close": float(close.iloc[-1]),
        "total_return_pct": float((close.iloc[-1] / close.iloc[0] - 1) * 100),
        "cagr_pct": float(cagr * 100) if cagr is not None else None,
        "annualized_volatility_pct": float(returns["Return"].std(ddof=1) * np.sqrt(252))
        if len(returns) > 1
        else None,
        "max_drawdown_pct": float(drawdown.min() * 100),
        "state_counts": state_counts,
        "state_percentages": state_counts / state_counts.sum() * 100,
    }

    return {
        "metrics": metrics,
        "transition_counts": counts,
        "transition_probabilities": probabilities,
        "returns_by_state": {
            state: returns.loc[returns["State"] == state, "Return"].to_numpy() / 100
            for state in STATES
        },
    }


def simulate_paths(
    data: pd.DataFrame,
    analysis: dict[str, Any],
    days: int = 30,
    path_count: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """Simulate prices by resampling observed returns within transition states."""
    if days < 1 or path_count < 1:
        raise ValueError("Simulation days and path count must be positive.")

    probabilities = analysis["transition_probabilities"]
    returns_by_state = analysis["returns_by_state"]
    available_states = [state for state in STATES if len(returns_by_state[state])]
    if not available_states:
        raise ValueError("No market returns are available for the simulation.")

    rng = np.random.default_rng(seed)
    starting_state = data["State"].iloc[-1]
    paths = np.empty((days + 1, path_count), dtype=float)
    paths[0] = data["Close"].iloc[-1]

    for path_index in range(path_count):
        state = starting_state
        price = paths[0, path_index]
        for day in range(1, days + 1):
            state_probabilities = probabilities.loc[state].to_numpy(dtype=float)
            next_state = rng.choice(STATES, p=state_probabilities)
            price *= 1 + rng.choice(returns_by_state[next_state])
            paths[day, path_index] = price
            state = next_state

    return pd.DataFrame(
        {
            "Trading day": np.arange(days + 1),
            "10th percentile": np.percentile(paths, 10, axis=1),
            "Median": np.percentile(paths, 50, axis=1),
            "90th percentile": np.percentile(paths, 90, axis=1),
        }
    ).set_index("Trading day")
