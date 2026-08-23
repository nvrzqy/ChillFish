import os
from pathlib import Path

os.environ.setdefault("TORCH_HOME", str(Path(__file__).resolve().parents[2] / "models" / "torch_cache"))

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


class MobileNetFeatureExtractor:
    def __init__(self, device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        weights = MobileNet_V3_Small_Weights.DEFAULT
        model = mobilenet_v3_small(weights=weights)
        model.classifier = torch.nn.Identity()
        model.eval()
        self.model = model.to(self.device)

    @torch.no_grad()
    def extract(self, image_tensor: torch.Tensor) -> np.ndarray:
        image_tensor = image_tensor.to(self.device)
        embedding = self.model(image_tensor)
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding.squeeze(0).detach().cpu().numpy().astype(np.float32)
