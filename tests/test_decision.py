from src.decision.decision_engine import make_decision


def test_anomalous_visual_goes_to_inspect():
    result = make_decision(
        {"status": "ANOMALOUS"},
        {"downgrade_probability": 0.1, "risk_band": "LOW"},
        {"downgrade_probability": 0.2, "risk_band": "LOW"},
    )

    assert result["recommendation"] == "INSPECT"


def test_high_wait_risk_goes_to_ship_now():
    result = make_decision(
        {"status": "NORMAL"},
        {"downgrade_probability": 0.2, "risk_band": "LOW"},
        {"downgrade_probability": 0.7, "risk_band": "HIGH"},
    )

    assert result["recommendation"] == "SHIP_NOW"
