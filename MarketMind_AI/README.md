# MarketMind AI — Stock Market Research Agent

This project upgrades the original Market Lens CSV dashboard into a small, tool-using educational market research agent. It preserves the original analytical approach: Date/Close validation, Up/Stable/Down states using configurable symmetric daily-return thresholds (default ±0.5%), Markov transition probabilities, historical risk metrics, and Monte Carlo return resampling.

## Features
- Upload a CSV with `Date` and `Close` columns.
- Adjust the symmetric Up/Down daily-return state threshold in the sidebar (0.1%–5.0%; default 0.5%).
- Validate and clean price history.
- Calculate total return, CAGR, annualized volatility and maximum drawdown.
- Estimate Markov state-transition probabilities.
- Run reproducible Monte Carlo scenario simulations.
- Produce a written research explanation and downloadable Markdown/JSON reports.
- Optional Gemini explanation through an API key. Numerical calculations are always performed locally in Python.

## Setup (Windows PowerShell)
```powershell
py -3.13 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
# Optional: open .env and add your Gemini API key
streamlit run app.py
```
Open the local URL printed by Streamlit (normally http://localhost:8501).

## Input format
CSV must contain `Date` and `Close`. Additional columns are ignored. Example:
```csv
Date,Close
2025-01-02,24000.5
2025-01-03,24120.0
```

## Optional Gemini setup
Set `GEMINI_API_KEY` in `.env`. Without it, the agent still runs and creates a deterministic local explanation. Never commit `.env` or share your API key. Gemini is only used to explain already-computed statistics; it does not calculate or invent the numerical results.

## Test
```powershell
pytest -q
```

## Agent workflow
`CSV → validation → historical metrics → Markov state model → Monte Carlo simulation → explanation → downloadable report`

## Important limitations
For education and data-analysis practice only; not investment advice, a trading signal, or a reliable forecast. The model resamples the uploaded historical return sample and does not capture all market drivers, structural changes, transaction costs, liquidity, or unexpected events. Simulated percentiles are not confidence guarantees.
