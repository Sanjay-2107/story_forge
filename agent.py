from __future__ import annotations
import os
from typing import Any
import numpy as np
import pandas as pd
from market_analysis import analyze_market_data, simulate_paths

class MarketResearchAgent:
    """A small tool-using educational research agent. Calculations stay in Python."""
    def __init__(self, model: str | None = None):
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.api_key = os.getenv("GEMINI_API_KEY")

    def run(
        self,
        data: pd.DataFrame,
        horizon: int = 30,
        path_count: int = 500,
        threshold_pct: float = 0.5,
    ) -> dict[str, Any]:
        analysis = analyze_market_data(data)
        scenarios = simulate_paths(data, analysis, days=horizon, path_count=path_count)
        m = analysis["metrics"]
        median_end = float(scenarios["Median"].iloc[-1])
        low_end = float(scenarios["10th percentile"].iloc[-1])
        high_end = float(scenarios["90th percentile"].iloc[-1])
        current = float(m["latest_close"])
        summary = {
            "state_threshold_pct": threshold_pct,
            "latest_close": current,
            "total_return_pct": m["total_return_pct"],
            "cagr_pct": m["cagr_pct"],
            "annualized_volatility_pct": m["annualized_volatility_pct"],
            "max_drawdown_pct": m["max_drawdown_pct"],
            "state_counts": {k: int(v) for k, v in m["state_counts"].items()},
            "transition_probabilities": analysis["transition_probabilities"].round(4).to_dict(),
            "simulation": {"horizon_trading_days": horizon, "paths": path_count,
                "10th_percentile_end": low_end, "median_end": median_end,
                "90th_percentile_end": high_end,
                "median_change_pct": (median_end/current-1)*100},
        }
        commentary = self._comment(summary)
        return {"summary": summary, "commentary": commentary, "scenarios": scenarios,
                "analysis": analysis, "agent_steps": [
                    "Validated and prepared the uploaded price history",
                    "Calculated historical performance and risk metrics",
                    "Estimated Up / Stable / Down state transitions",
                    f"Simulated {path_count:,} illustrative paths over {horizon} trading days",
                    "Generated an evidence-based research explanation" ]}

    def _comment(self, s: dict[str, Any]) -> str:
        sim=s["simulation"]
        lines=[
          f"Daily return states use a symmetric ±{s['state_threshold_pct']:.2f}% threshold.",
          f"The dataset's latest closing value is {s['latest_close']:,.2f}.",
          f"Across the uploaded history, total price change was {s['total_return_pct']:+.2f}%.",
          f"Historical maximum drawdown was {s['max_drawdown_pct']:.2f}%.",
          f"Annualized historical volatility was {s['annualized_volatility_pct']:.2f}%." if s['annualized_volatility_pct'] is not None else "Annualized volatility could not be estimated from this sample.",
          f"In the {sim['horizon_trading_days']}-trading-day resampling exercise, the simulated ending-value 10th/50th/90th percentiles were {sim['10th_percentile_end']:,.2f}, {sim['median_end']:,.2f}, and {sim['90th_percentile_end']:,.2f}.",
          "These are summaries of a historical-resampling experiment, not predictions. The model assumes the historical return patterns remain informative; real markets can behave differently."]
        if self.api_key:
            try:
                from google import genai
                client = genai.Client(api_key=self.api_key)
                prompt = (
                    "You are an educational market-data explainer. Explain the following computed statistics in plain language. "
                    "Do not recommend buying/selling/holding, do not predict certainty, and clearly state limitations. "
                    "Do not invent data. Computed data:\n" + str(s)
                )
                response = client.models.generate_content(model=self.model, contents=prompt)
                if response.text:
                    return response.text.strip()
            except Exception as exc:
                return "\n".join(lines) + f"\n\nOptional Gemini explanation unavailable ({type(exc).__name__}); showing the local analysis instead."
        return "\n".join(lines) + "\n\nAI language explanation is optional: add GEMINI_API_KEY to .env to enable Gemini. All numerical results above are calculated locally in Python."
