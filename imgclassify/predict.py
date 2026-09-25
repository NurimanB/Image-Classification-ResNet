"""Core inference: load a pretrained torchvision model and classify images."""

from __future__ import annotations

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

from .labels import load_labels

PREPROCESS = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Model name -> (constructor, weights enum). Add more here to extend --model.
MODEL_REGISTRY = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1),
    "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V1),
    "resnet101": (models.resnet101, models.ResNet101_Weights.IMAGENET1K_V1),
}


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(name: str = "resnet50", device: torch.device | None = None) -> torch.nn.Module:
    """Load a pretrained, eval-mode model by name (see MODEL_REGISTRY)."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Choose from: {list(MODEL_REGISTRY)}")
    fn, weights = MODEL_REGISTRY[name]
    model = fn(weights=weights)
    model.eval()
    if device is not None:
        model.to(device)
    return model


def load_image_tensor(image_path: str, device: torch.device | None = None) -> torch.Tensor:
    """Preprocess a single image file into a (1, 3, 224, 224) batch tensor."""
    img = Image.open(image_path).convert("RGB")
    tensor = PREPROCESS(img).unsqueeze(0)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def classify_image(
    image_path: str,
    model: torch.nn.Module,
    labels: list[str] | None = None,
    top_k: int = 5,
    device: torch.device | None = None,
) -> list[dict]:
    """Return the top-k {class, probability} predictions for one image."""
    labels = labels or load_labels()
    tensor = load_image_tensor(image_path, device)

    with torch.no_grad():
        output = model(tensor)

    probs = torch.nn.functional.softmax(output[0], dim=0)
    top_probs, top_idxs = torch.topk(probs, min(top_k, len(labels)))
    return [
        {"class": labels[idx.item()], "probability": prob.item() * 100}
        for prob, idx in zip(top_probs, top_idxs)
    ]
