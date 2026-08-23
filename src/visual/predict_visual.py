import argparse
import json
from pathlib import Path

import numpy as np

from src.config import PROJECT_ROOT, load_config
from src.visual.feature_extractor import MobileNetFeatureExtractor
from src.visual.preprocess import load_image


def predict_visual(
    image_path: str | Path,
    models_dir: str | Path = PROJECT_ROOT / "models",
    config_path: str | Path | None = None,
) -> dict:
    config = load_config(config_path or PROJECT_ROOT / "config" / "config.yaml")
    image_size = int(config["visual"]["image_size"])
    models_dir = Path(models_dir)

    centroid_path = models_dir / "reference_centroid.npy"
    threshold_path = models_dir / "visual_threshold.json"
    if not centroid_path.exists() or not threshold_path.exists():
        raise FileNotFoundError(
            "Visual reference model is missing. Run: python -m src.visual.build_reference"
        )

    centroid = np.load(centroid_path)
    with open(threshold_path, "r", encoding="utf-8") as handle:
        threshold = json.load(handle)

    extractor = MobileNetFeatureExtractor()
    tensor = load_image(image_path, image_size=image_size)
    embedding = extractor.extract(tensor)
    anomaly_score = float(np.linalg.norm(embedding - centroid))

    if anomaly_score > float(threshold["anomaly_threshold"]):
        status = "ANOMALOUS"
    elif anomaly_score > float(threshold["check_threshold"]):
        status = "CHECK"
    else:
        status = "NORMAL"

    return {
        "anomaly_score": anomaly_score,
        "status": status,
        "check_threshold": float(threshold["check_threshold"]),
        "anomaly_threshold": float(threshold["anomaly_threshold"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--models-dir", default=PROJECT_ROOT / "models")
    args = parser.parse_args()
    print(json.dumps(predict_visual(args.image, args.models_dir), indent=2))


if __name__ == "__main__":
    main()
