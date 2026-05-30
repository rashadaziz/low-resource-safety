"""Reusable accessors for stored activation tensors."""

from pathlib import Path

import torch
from safetensors import safe_open

from multilingual_latent_safety.analysis import load_subset_activations, token_index
from multilingual_latent_safety.paths import activations_file


def activation_token_index(
    activations_root: str | Path,
    language: str,
    split: str,
    subset: str,
    layer: int,
    token_position: str | int,
) -> int:
    path = activations_file(activations_root, language, split, subset, layer)
    with safe_open(path, framework="pt") as f:
        metadata = f.metadata() or {}
    return token_index(metadata.get("token_positions", "-1"), token_position)


class ActivationCache:
    """Small cache for harmful/harmless activation tensors."""

    def __init__(
        self,
        root: str | Path,
        layer: int,
        token_position: str | int,
        device: torch.device | None = None,
    ) -> None:
        self.root = Path(root)
        self.layer = int(layer)
        self.token_position = token_position
        self.device = device
        self.pair_cache: dict[tuple[str, str], tuple[torch.Tensor, torch.Tensor]] = {}
        self.subset_cache: dict[tuple[str, str, str], torch.Tensor] = {}

    def token_index(self, language: str, split: str, subset: str = "harmful") -> int:
        return activation_token_index(
            self.root, language, split, subset, self.layer, self.token_position
        )

    def subset(self, language: str, split: str, subset: str) -> torch.Tensor:
        key = (language, split, subset)
        if key not in self.subset_cache:
            tok_idx = self.token_index(language, split, subset)
            activations = load_subset_activations(
                self.root, language, split, subset, self.layer, tok_idx
            )
            if self.device is not None:
                activations = activations.to(self.device)
            self.subset_cache[key] = activations
        return self.subset_cache[key]

    def pair(self, language: str, split: str) -> tuple[torch.Tensor, torch.Tensor]:
        key = (language, split)
        if key not in self.pair_cache:
            self.pair_cache[key] = (
                self.subset(language, split, "harmful"),
                self.subset(language, split, "harmless"),
            )
        return self.pair_cache[key]
