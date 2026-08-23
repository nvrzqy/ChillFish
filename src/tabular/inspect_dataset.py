import json

from src.tabular.dataset import load_competition_table


def main() -> None:
    df = load_competition_table()
    summary = {
        "rows": int(len(df)),
        "recommended_action_counts": df["recommended_action"].value_counts().to_dict(),
        "condition_status_counts": df["condition_status"].value_counts().to_dict(),
        "risk_score_min": float(df["risk_score_0_100"].min()),
        "risk_score_max": float(df["risk_score_0_100"].max()),
        "risk_score_mean": float(df["risk_score_0_100"].mean()),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
