"""Batch vs. one-at-a-time inference timing on a single model."""

from __future__ import annotations

import time

import torch
from PIL import Image

try:
    from .predict import PREPROCESS, load_model
except ImportError:  # pragma: no cover - allows running as a script directly
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from imgclassify.predict import PREPROCESS, load_model


def benchmark_batch_vs_individual(image_paths: list[str], model_name: str = "resnet50") -> dict:
    """Time N images processed one-by-one vs. stacked into a single batched forward pass."""
    model = load_model(model_name)

    individual_times: list[float] = []
    for path in image_paths:
        tensor = PREPROCESS(Image.open(path).convert("RGB")).unsqueeze(0)
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(tensor)
        individual_times.append((time.perf_counter() - t0) * 1000)

    total_individual = sum(individual_times)

    tensors = [PREPROCESS(Image.open(p).convert("RGB")) for p in image_paths]
    batch = torch.stack(tensors)

    t0 = time.perf_counter()
    with torch.no_grad():
        _ = model(batch)
    total_batch = (time.perf_counter() - t0) * 1000

    return {
        "n_images": len(image_paths),
        "individual_total_ms": round(total_individual, 1),
        "individual_per_img_ms": round(total_individual / len(image_paths), 1),
        "individual_times_ms": [round(t, 1) for t in individual_times],
        "batch_total_ms": round(total_batch, 1),
        "batch_per_img_ms": round(total_batch / len(image_paths), 1),
        "speedup": round(total_individual / total_batch, 2),
    }
