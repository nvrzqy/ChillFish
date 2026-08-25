import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from src.config import PROJECT_ROOT


MODEL_DIR = PROJECT_ROOT / "models" / "visual_lite"
REFERENCE_PATH = MODEL_DIR / "reference.npz"
THRESHOLD_PATH = MODEL_DIR / "threshold.json"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".avif"}


def list_reference_images(directory: str | Path) -> list[Path]:
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Image directory not found: {directory}")
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def extract_simple_features(image_path: str | Path) -> np.ndarray:
    image = Image.open(image_path).convert("RGB").resize((224, 224))
    arr = np.asarray(image, dtype=np.float32) / 255.0

    channel_means = arr.mean(axis=(0, 1))
    channel_stds = arr.std(axis=(0, 1))

    histograms = []
    for channel in range(3):
        hist, _ = np.histogram(arr[:, :, channel], bins=16, range=(0.0, 1.0), density=True)
        histograms.append(hist.astype(np.float32))

    gray = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    edge = np.asarray(image.convert("L").filter(ImageFilter.FIND_EDGES), dtype=np.float32) / 255.0
    texture = np.array(
        [
            gray.mean(),
            gray.std(),
            np.percentile(gray, 10),
            np.percentile(gray, 50),
            np.percentile(gray, 90),
            edge.mean(),
            edge.std(),
        ],
        dtype=np.float32,
    )

    feature = np.concatenate([channel_means, channel_stds, *histograms, texture])
    norm = np.linalg.norm(feature)
    return (feature / max(norm, 1e-12)).astype(np.float32)


def build_simple_reference(
    reference_dir: str | Path = PROJECT_ROOT / "data" / "images" / "reference",
    model_dir: str | Path = MODEL_DIR,
) -> dict:
    image_paths = list_reference_images(reference_dir)
    if not image_paths:
        raise ValueError(f"No reference images found in {reference_dir}")

    features = np.vstack([extract_simple_features(path) for path in image_paths])
    centroid = features.mean(axis=0)
    centroid = centroid / max(np.linalg.norm(centroid), 1e-12)
    distances = np.linalg.norm(features - centroid, axis=1)

    mean_distance = float(distances.mean())
    std_distance = float(distances.std(ddof=0))
    threshold = {
        "reference_count": int(len(image_paths)),
        "mean_distance": mean_distance,
        "std_distance": std_distance,
        "check_threshold": mean_distance + std_distance,
        "anomaly_threshold": mean_distance + (2.0 * std_distance),
        "method": "simple_color_texture_centroid",
        "note": "Prototype visual anomaly screening from normal lemuru reference photos.",
    }

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    np.savez(model_dir / "reference.npz", features=features, centroid=centroid)
    with open(model_dir / "threshold.json", "w", encoding="utf-8") as handle:
        json.dump(threshold, handle, indent=2)
    return threshold


def predict_simple_visual(
    image_path: str | Path,
    model_dir: str | Path = MODEL_DIR,
) -> dict:
    model_dir = Path(model_dir)
    if not (model_dir / "reference.npz").exists() or not (model_dir / "threshold.json").exists():
        raise FileNotFoundError("Visual lite reference is missing. Run: python -m src.visual.simple_anomaly --build")

    reference = np.load(model_dir / "reference.npz")
    centroid = reference["centroid"]
    with open(model_dir / "threshold.json", "r", encoding="utf-8") as handle:
        threshold = json.load(handle)

    feature = extract_simple_features(image_path)
    score = float(np.linalg.norm(feature - centroid))
    if score > float(threshold["anomaly_threshold"]):
        status = "ANOMALOUS"
    elif score > float(threshold["check_threshold"]):
        status = "CHECK"
    else:
        status = "NORMAL"

    return {
        "anomaly_score": score,
        "status": status,
        "check_threshold": float(threshold["check_threshold"]),
        "anomaly_threshold": float(threshold["anomaly_threshold"]),
        "method": threshold["method"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--reference-dir", default=PROJECT_ROOT / "data" / "images" / "reference")
    parser.add_argument("--image", default=None)
    args = parser.parse_args()

    if args.build:
        print(json.dumps(build_simple_reference(args.reference_dir), indent=2))
    elif args.image:
        print(json.dumps(predict_simple_visual(args.image), indent=2))
    else:
        parser.error("Use --build or --image")


if __name__ == "__main__":
    main()
