from __future__ import annotations
import json
from datetime import datetime

def make_markdown_report(result: dict, filename: str = "uploaded dataset") -> str:
    s=result["summary"]; sim=s["simulation"]
    def pct(v): return "Not available" if v is None else f"{v:+.2f}%"
    lines=["# MarketMind AI — Educational Market Research Report", "", f"**Source:** {filename}", f"**Generated:** {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M %Z')}", "",
      "> Educational data analysis only. Not investment advice, a trading signal, or a reliable forecast.", "", "## Historical overview",
      f"- Latest close: {s['latest_close']:,.2f}", f"- Total return: {pct(s['total_return_pct'])}", f"- CAGR: {pct(s['cagr_pct'])}",
      f"- Annualized volatility: {pct(s['annualized_volatility_pct'])}", f"- Maximum drawdown: {s['max_drawdown_pct']:.2f}%",
      f"- State classification threshold: ±{s['state_threshold_pct']:.2f}%", "", "## Observed market states"]
    lines += [f"- {k}: {v:,}" for k,v in s['state_counts'].items()]
    lines += ["", "## Scenario simulation", f"- Horizon: {sim['horizon_trading_days']} trading days", f"- Paths: {sim['paths']:,}",
      f"- 10th percentile ending value: {sim['10th_percentile_end']:,.2f}", f"- Median ending value: {sim['median_end']:,.2f}",
      f"- 90th percentile ending value: {sim['90th_percentile_end']:,.2f}", "", "## Agent explanation", result['commentary'], "", "## Workflow"]
    lines += [f"{i}. {step}" for i,step in enumerate(result['agent_steps'],1)]
    lines += ["", "## Limitations", "The Markov transition model and Monte Carlo resampling are based on the uploaded historical sample. They do not account for all market events, changing regimes, fees, liquidity, or future information. Percentile ranges are simulated outcomes, not confidence guarantees."]
    return "\n".join(lines)

def make_json_report(result: dict) -> bytes:
    return json.dumps(result['summary'], indent=2, default=str).encode('utf-8')
