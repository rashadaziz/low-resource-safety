from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from safetensors import safe_open
from safetensors.torch import save_file

from multilingual_latent_safety.analysis import (
    layer_ids,
    load_subset_activations,
    token_index,
)
from multilingual_latent_safety.paths import (
    activations_file,
    direction_dir,
    direction_file,
)

DIRECTION_METHODS = ("diff_of_means", "pca_top1")


def estimate_direction(harmful: torch.Tensor, harmless: torch.Tensor, method: str) -> torch.Tensor:
    """Estimate a 1-D harmful direction from per-class activation matrices ``(N_h, d)``.

    - ``diff_of_means``: μ_harmful − μ_harmless (Arditi et al. baseline).
    - ``pca_top1``: top right-singular vector of the joint, jointly-mean-centered activations,
      sign-aligned with the diff-of-means direction (so a positive projection still encodes
      "more harmful").
    """
    if method == "diff_of_means":
        return harmful.mean(dim=0) - harmless.mean(dim=0)
    if method == "pca_top1":
        combined = torch.cat([harmful, harmless], dim=0)
        combined = combined - combined.mean(dim=0, keepdim=True)
        _, _, vh = torch.linalg.svd(combined, full_matrices=False)
        direction = vh[0]
        anchor = harmful.mean(dim=0) - harmless.mean(dim=0)
        if direction @ anchor < 0:
            direction = -direction
        return direction
    raise ValueError(f"method={method!r}; supported: {DIRECTION_METHODS}")


@hydra.main(version_base=None, config_path="../../configs", config_name="hrl_direction/compute_harmful_direction")
def main(cfg: DictConfig) -> None:
    """Estimate a harmful-vs-harmless direction per (language, layer) from saved activations."""
    acts_root = Path(cfg.activations_root)
    out_root = Path(cfg.output_root)

    if cfg.method not in DIRECTION_METHODS:
        raise ValueError(f"method={cfg.method!r}; supported: {DIRECTION_METHODS}")

    for language in cfg.languages:
        harmful_layers = layer_ids(acts_root, language, cfg.split, "harmful")
        harmless_layers = layer_ids(acts_root, language, cfg.split, "harmless")
        if harmful_layers != harmless_layers:
            raise ValueError(
                f"[{language}] layer mismatch: harmful={harmful_layers[:3]}... harmless={harmless_layers[:3]}..."
            )

        sample_meta_path = activations_file(acts_root, language, cfg.split, "harmless", harmful_layers[0])
        with safe_open(sample_meta_path, framework="pt") as f:
            meta = f.metadata() or {}
        tok_idx = token_index(meta.get("token_positions", "-1"), cfg.token_position)

        out_dir = direction_dir(out_root, language, cfg.split, cfg.token_position)
        out_dir.mkdir(parents=True, exist_ok=True)

        for layer in harmful_layers:
            harmful = load_subset_activations(acts_root, language, cfg.split, "harmful", layer, tok_idx)
            harmless = load_subset_activations(acts_root, language, cfg.split, "harmless", layer, tok_idx)
            direction = estimate_direction(harmful, harmless, cfg.method)
            if cfg.normalize:
                direction = direction / direction.norm().clamp(min=1e-8)
            save_file(
                {"direction": direction.contiguous()},
                direction_file(out_root, language, cfg.split, cfg.token_position, layer),
                metadata={
                    "model": cfg.model.name,
                    "language": language,
                    "split": cfg.split,
                    "layer": str(layer),
                    "token_position": str(cfg.token_position),
                    "method": cfg.method,
                    "normalize": str(cfg.normalize),
                    "n_harmful": str(harmful.shape[0]),
                    "n_harmless": str(harmless.shape[0]),
                },
            )
        print(f"[done] {language}: wrote {len(harmful_layers)} directions to {out_dir}")


if __name__ == "__main__":
    main()
