from pathlib import Path

from PIL import Image
from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_image_transform(image_size: int = 224):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def load_image(path: str | Path, image_size: int = 224):
    image_path = Path(path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as exc:
        raise ValueError(f"Invalid image file: {image_path}") from exc

    transform = get_image_transform(image_size)
    return transform(image).unsqueeze(0)


def list_images(directory: str | Path) -> list[Path]:
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Image directory not found: {directory}")

    extensions = {".jpg", ".jpeg", ".png", ".webp", ".avif"}
    return sorted(path for path in directory.iterdir() if path.suffix.lower() in extensions)
