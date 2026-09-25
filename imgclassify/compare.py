"""Compare ResNet variants on size, CPU inference speed, and published accuracy."""

from __future__ import annotations

import time

import torch

from .predict import MODEL_REGISTRY, load_image_tensor, load_labels, load_model

# Published ImageNet-1k top-1 accuracy for the IMAGENET1K_V1 weights (torchvision docs).
PUBLISHED_TOP1_ACC = {
    "resnet18": 69.8,
    "resnet50": 76.1,
    "resnet101": 77.4,
}


def compare_models(image_paths: dict[str, str], model_names: list[str] | None = None) -> dict:
    """Run each model over every image; return params, avg inference time, and predictions."""
    model_names = model_names or list(MODEL_REGISTRY)
    labels = load_labels()
    comparison: dict[str, dict] = {}

    for name in model_names:
        model = load_model(name)
        n_params = sum(p.numel() for p in model.parameters())

        times: list[float] = []
        preds: dict[str, str] = {}
        for img_name, path in image_paths.items():
            tensor = load_image_tensor(path)
            t0 = time.perf_counter()
            with torch.no_grad():
                out = model(tensor)
            times.append((time.perf_counter() - t0) * 1000)
            _, idx = torch.max(out, 1)
            preds[img_name] = labels[idx.item()]

        comparison[name] = {
            "params_M": round(n_params / 1e6, 1),
            "avg_inference_ms": round(sum(times) / len(times), 1),
            "per_image_ms": [round(t, 1) for t in times],
            "top1_predictions": preds,
            "published_top1_acc": PUBLISHED_TOP1_ACC.get(name),
        }

    return comparison
