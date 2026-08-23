import argparse
import json
from pathlib import Path

from src.config import PROJECT_ROOT
from src.decision.decision_engine import make_decision
from src.thermal.predict_thermal import predict_thermal
from src.visual.predict_visual import predict_visual


def run_demo(
    image: str | Path,
    temperature: str | Path,
    lot_id: str,
    age_hours: float,
    wait_hours: float,
    eta_hours: float,
) -> dict:
    print("loading visual model")
    visual = predict_visual(image)

    print("loading temperature history")
    print("running XGBoost for SHIP NOW")
    ship = predict_thermal(temperature, lot_id, age_hours, eta_hours)

    print("running XGBoost for WAIT")
    wait_horizon = wait_hours + eta_hours
    wait = predict_thermal(temperature, lot_id, age_hours, wait_horizon)

    print("making recommendation")
    decision = make_decision(visual, ship, wait)

    return {
        "lot_id": lot_id,
        "visual": visual,
        "thermal": {
            "ship_horizon_hours": eta_hours,
            "ship": ship,
            "wait_horizon_hours": wait_horizon,
            "wait": wait,
        },
        "decision": decision,
    }


def print_report(result: dict) -> None:
    visual = result["visual"]
    ship = result["thermal"]["ship"]
    wait = result["thermal"]["wait"]
    decision = result["decision"]

    print("\n========================================")
    print("LEMURU COLD-CHAIN AI")
    print("========================================\n")
    print("LOT")
    print(f"ID            : {result['lot_id']}\n")
    print("VISUAL CONDITION")
    print(f"Status        : {visual['status']}")
    print(f"Anomaly score : {visual['anomaly_score']:.4f}")
    print(f"Check thr.    : {visual['check_threshold']:.4f}")
    print(f"Anomaly thr.  : {visual['anomaly_threshold']:.4f}\n")
    print("THERMAL RISK")
    print("SHIP NOW")
    print(f"Horizon       : {result['thermal']['ship_horizon_hours']:.0f} hours")
    print(f"Risk          : {ship['downgrade_probability'] * 100:.1f}%")
    print(f"Band          : {ship['risk_band']}\n")
    print("WAIT")
    print(f"Horizon       : {result['thermal']['wait_horizon_hours']:.0f} hours")
    print(f"Risk          : {wait['downgrade_probability'] * 100:.1f}%")
    print(f"Band          : {wait['risk_band']}\n")
    print("----------------------------------------\n")
    print("RECOMMENDATION")
    print(decision["recommendation"])
    print("\nREASON")
    for reason in decision["reasons"]:
        print(f"- {reason}")
    print("\n========================================")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--temperature", required=True)
    parser.add_argument("--lot-id", required=True)
    parser.add_argument("--age-hours", type=float, required=True)
    parser.add_argument("--wait-hours", type=float, default=24)
    parser.add_argument("--eta-hours", type=float, default=48)
    parser.add_argument("--json-output", default=None)
    args = parser.parse_args()

    result = run_demo(
        args.image,
        args.temperature,
        args.lot_id,
        args.age_hours,
        args.wait_hours,
        args.eta_hours,
    )
    print_report(result)

    if args.json_output:
        output_path = Path(args.json_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)


if __name__ == "__main__":
    main()
