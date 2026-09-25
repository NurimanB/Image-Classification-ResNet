"""ImageNet-1k class labels, cached locally after the first download."""

from __future__ import annotations

import os

import requests

_LABELS_URL = (
    "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
)
_CACHE_PATH = os.path.join(os.path.dirname(__file__), "..", ".cache", "imagenet_classes.txt")


def load_labels(cache_path: str = _CACHE_PATH) -> list[str]:
    """Return the 1000 ImageNet class names, downloading once and caching to disk."""
    cache_path = os.path.abspath(cache_path)
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            return f.read().strip().split("\n")

    response = requests.get(_LABELS_URL, timeout=15)
    response.raise_for_status()
    labels = response.text.strip().split("\n")

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        f.write(response.text.strip())

    return labels
