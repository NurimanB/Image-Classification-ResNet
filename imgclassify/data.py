"""Sample image set: two are checked into data/images/, the rest download on demand."""

from __future__ import annotations

import os

import requests

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "images")

# Images not already in data/images/ are fetched from these URLs the first time
# they're needed and cached alongside the checked-in samples.
IMAGE_URLS = {
    "dog": "https://upload.wikimedia.org/wikipedia/commons/2/26/YellowLabradorLooking_new.jpg",
    "cat": "https://upload.wikimedia.org/wikipedia/commons/4/4d/Cat_November_2010-1a.jpg",
    "car": "https://images.unsplash.com/photo-1492144534655-ae79c964c9d7?auto=format&fit=crop&w=1200&q=80",
    "airplane": "https://images.unsplash.com/photo-1436491865332-7a61a109cc05?auto=format&fit=crop&w=1200&q=80",
    "banana": "https://images.unsplash.com/photo-1571771894821-ce9b6c11b08e?auto=format&fit=crop&w=1200&q=80",
}


def get_sample_images(names: list[str] | None = None, save_dir: str = SAMPLE_DIR) -> dict[str, str]:
    """Return {name: local_path} for the requested sample images (default: all of them).

    Anything already present in save_dir is used as-is; anything missing is
    downloaded from IMAGE_URLS.
    """
    names = names or list(IMAGE_URLS.keys())
    os.makedirs(save_dir, exist_ok=True)
    headers = {"User-Agent": "Mozilla/5.0"}

    paths: dict[str, str] = {}
    for name in names:
        path = os.path.join(save_dir, f"{name}.jpg")
        if os.path.exists(path):
            paths[name] = path
            continue
        if name not in IMAGE_URLS:
            raise KeyError(f"No sample image or URL registered for '{name}'")
        response = requests.get(IMAGE_URLS[name], headers=headers, timeout=20)
        response.raise_for_status()
        with open(path, "wb") as f:
            f.write(response.content)
        paths[name] = path

    return paths
