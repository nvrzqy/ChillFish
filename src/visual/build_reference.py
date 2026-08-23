import argparse
import json
from pathlib import Path

import numpy as np

from src.config import PROJECT_ROOT, load_config
from src.visual.feature_extractor import MobileNetFeatureExtractor
from src.visual.preprocess import list_images, load_image


def build_reference(
    reference_dir: str | Path,
    models_dir: str | Path = PROJECT_ROOT / "models",
    config_path: str | Path | None = None,
) -> dict:
    config = load_config(config_path or PROJECT_ROOT / "config" / "config.yaml")
    image_size = int(config["visual"]["image_size"])
    multiplier = float(config["visual"]["threshold_std_multiplier"])

    image_paths = list_images(reference_dir)
    if not image_paths:
        raise ValueError(f"No reference images found in {reference_dir}")

    extractor = MobileNetFeatureExtractor()
    embeddings = []
    for image_path in image_paths:
        tensor = load_image(image_path, image_size=image_size)
        embeddings.append(extractor.extract(tensor))

    embeddings_array = np.vstack(embeddings)
    centroid = embeddings_array.mean(axis=0)
    centroid = centroid / max(np.linalg.norm(centroid), 1e-12)

    distances = np.linalg.norm(embeddings_array - centroid, axis=1)
    mean_distance = float(distances.mean())
    std_distance = float(distances.std(ddof=0))

    threshold = {
        "reference_count": int(len(image_paths)),
        "mean_distance": mean_distance,
        "std_distance": std_distance,
        "check_threshold": mean_distance + std_distance,
        "anomaly_threshold": mean_distance + (multiplier * std_distance),
        "method": "centroid_euclidean",
    }

    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    np.save(models_dir / "reference_embeddings.npy", embeddings_array)
    np.save(models_dir / "reference_centroid.npy", centroid)
    with open(models_dir / "visual_threshold.json", "w", encoding="utf-8") as handle:
        json.dump(threshold, handle, indent=2)

    return threshold


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-dir", default=PROJECT_ROOT / "data" / "images" / "reference")
    parser.add_argument("--models-dir", default=PROJECT_ROOT / "models")
    args = parser.parse_args()

    threshold = build_reference(args.reference_dir, args.models_dir)
    print(json.dumps(threshold, indent=2))


if __name__ == "__main__":
    main()
