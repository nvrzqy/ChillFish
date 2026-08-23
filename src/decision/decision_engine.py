def make_decision(visual_result: dict, ship_result: dict, wait_result: dict) -> dict:
    visual_status = visual_result.get("status", "UNKNOWN")
    ship_risk = float(ship_result["downgrade_probability"])
    wait_risk = float(wait_result["downgrade_probability"])
    wait_band = wait_result.get("risk_band", "UNKNOWN")

    reasons = []
    if visual_status == "ANOMALOUS":
        reasons.append("Visual appearance is outside the reference distribution.")
        if wait_band == "HIGH":
            reasons.append("Thermal risk is also high, so the lot should be prioritized for inspection.")
        return {"recommendation": "INSPECT", "reasons": reasons}

    if visual_status == "CHECK":
        reasons.append("Visual appearance is near the anomaly threshold.")

    if wait_band == "HIGH" and wait_risk > ship_risk:
        reasons.append("Waiting increases predicted thermal downgrade risk.")
        return {"recommendation": "SHIP_NOW", "reasons": reasons}

    if ship_result.get("risk_band") == "HIGH":
        reasons.append("Thermal risk is already high at the shipping horizon.")
        return {"recommendation": "SHIP_NOW", "reasons": reasons}

    reasons.append("Visual and thermal signals do not indicate urgent action in this demo threshold.")
    return {"recommendation": "WAIT_POSSIBLE", "reasons": reasons}
