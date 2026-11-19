import pandas as pd
from market_analysis import analyze_market_data, prepare_market_data
from agent import MarketResearchAgent

def sample():
    return pd.DataFrame({"Date":pd.date_range("2024-01-01",periods=80,freq="B"),"Close":[100+i*0.25+((i%5)-2)*0.4 for i in range(80)]})

def test_agent_pipeline_returns_expected_sections():
    data=prepare_market_data(sample())
    result=MarketResearchAgent().run(data,horizon=10,path_count=100)
    assert "summary" in result and "commentary" in result and "scenarios" in result
    assert len(result["scenarios"]) == 11
    assert result["summary"]["simulation"]["paths"] == 100

def test_csv_column_names_case_insensitive():
    data=prepare_market_data(pd.DataFrame({"date":pd.date_range("2024-01-01",periods=4),"CLOSE":[10,11,10.5,12]}))
    assert len(data)==4

def test_state_threshold_can_be_changed():
    prices=pd.DataFrame({"Date":pd.date_range("2024-01-01",periods=3),"Close":[100,100.4,100.8]})
    default_states=prepare_market_data(prices)["State"]
    lower_threshold_data=prepare_market_data(prices,threshold_pct=0.3)
    lower_threshold_states=lower_threshold_data["State"]

    assert pd.isna(default_states.iloc[0])
    assert default_states.iloc[1:].tolist() == ["Stable","Stable"]
    assert pd.isna(lower_threshold_states.iloc[0])
    assert lower_threshold_states.iloc[1:].tolist() == ["Up","Up"]
    assert analyze_market_data(lower_threshold_data)["metrics"]["state_counts"]["Up"] == 2

def test_state_threshold_must_be_non_negative_and_finite():
    prices=pd.DataFrame({"Date":pd.date_range("2024-01-01",periods=2),"Close":[100,101]})

    for threshold in (-0.1,float("nan"),float("inf")):
        try:
            prepare_market_data(prices,threshold_pct=threshold)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid thresholds should raise ValueError")
