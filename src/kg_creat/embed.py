"""Local, torch-free sentence embeddings via mlx-embeddings (Apple Silicon).

Provides the ``embed: Callable[[str], np.ndarray]`` that ``scoring.novelty_R`` needs,
memoized per string (entity/triple labels repeat heavily across paths). Model default is
MiniLM (the same family CREATE uses for diversity), swappable for an ablation.
"""

from __future__ import annotations

import numpy as np

_DEFAULT_MODEL = "mlx-community/all-MiniLM-L6-v2-4bit"
_state: dict = {"model": None, "tok": None, "name": None, "cache": {}}


def get_embedder(model_name: str = _DEFAULT_MODEL):
    """Return a memoized ``embed(text) -> np.ndarray`` backed by a local MLX model."""
    if _state["model"] is None or _state["name"] != model_name:
        from mlx_embeddings import load
        _state["model"], _state["tok"] = load(model_name)
        _state["name"] = model_name
        _state["cache"] = {}

    def embed(text: str) -> np.ndarray:
        cache = _state["cache"]
        if text not in cache:
            from mlx_embeddings import generate
            out = generate(_state["model"], _state["tok"], texts=[text])
            cache[text] = np.asarray(out.text_embeds[0], dtype=np.float32)
        return cache[text]

    return embed
