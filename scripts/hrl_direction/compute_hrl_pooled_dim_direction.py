"""Pooled HRL difference-in-means direction, per layer.

For each layer, this script concatenates all HRL harmful train activations and
all HRL harmless train activations, computes their mean difference, normalizes
it, and writes the result under ``pooled_direction_file``.
"""

from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from safetensors import safe_open
from safetensors.torch import save_file

from multilingual_latent_safety.analysis import layer_ids, load_subset_activations, token_index
from multilingual_latent_safety.paths import activations_file, pooled_direction_file


def estimate_pooled_dim_direction(harmful_chunks: list[torch.Tensor], harmless_chunks: list[torch.Tensor]) -> torch.Tensor:
    """Compute pooled ``mean(harmful) - mean(harmless)`` across language chunks."""
    if not harmful_chunks or not harmless_chunks:
        raise ValueError("pooled DIM requires at least one harmful and one harmless activation chunk")
    harmful = torch.cat(harmful_chunks, dim=0).to(torch.float32)
    harmless = torch.cat(harmless_chunks, dim=0).to(torch.float32)
    if harmful.shape[1] != harmless.shape[1]:
        raise ValueError(f"activation width mismatch: harmful={harmful.shape}, harmless={harmless.shape}")
    return harmful.mean(dim=0) - harmless.mean(dim=0)


@hydra.main(version_base=None, config_path="../../configs", config_name="hrl_direction/compute_hrl_pooled_dim_direction")
def main(cfg: DictConfig) -> None:
    acts_root = Path(cfg.activations_root)
    output_root = Path(cfg.output_root)
    languages = [str(language) for language in cfg.languages]
    if not languages:
        raise ValueError("cfg.languages is empty")

    layers = layer_ids(acts_root, languages[0], cfg.split, "harmful")
    if not layers:
        raise ValueError(f"no harmful activation layers found for {languages[0]} at {acts_root}")

    sample_meta_path = activations_file(acts_root, languages[0], cfg.split, "harmful", layers[0])
    with safe_open(sample_meta_path, framework="pt") as f:
        meta = f.metadata() or {}
    tok_idx = token_index(meta.get("token_positions", "-1"), cfg.token_position)

    first_out = pooled_direction_file(output_root, cfg.pool, cfg.token_position, layers[0])
    first_out.parent.mkdir(parents=True, exist_ok=True)

    for layer in layers:
        harmful_chunks: list[torch.Tensor] = []
        harmless_chunks: list[torch.Tensor] = []
        for language in languages:
            harmful_layers = layer_ids(acts_root, language, cfg.split, "harmful")
            harmless_layers = layer_ids(acts_root, language, cfg.split, "harmless")
            if layer not in harmful_layers or layer not in harmless_layers:
                raise ValueError(f"[{language}] missing layer {layer} for split={cfg.split}")
            harmful_chunks.append(load_subset_activations(acts_root, language, cfg.split, "harmful", layer, tok_idx))
            harmless_chunks.append(load_subset_activations(acts_root, language, cfg.split, "harmless", layer, tok_idx))

        direction = estimate_pooled_dim_direction(harmful_chunks, harmless_chunks)
        if cfg.normalize:
            direction = direction / direction.norm().clamp(min=1e-8)

        save_file(
            {"direction": direction.contiguous()},
            pooled_direction_file(output_root, cfg.pool, cfg.token_position, layer),
            metadata={
                "model": cfg.model.name,
                "pool": cfg.pool,
                "languages": ",".join(languages),
                "split": cfg.split,
                "layer": str(layer),
                "token_position": str(cfg.token_position),
                "method": "pooled_dim",
                "normalize": str(cfg.normalize),
                "n_languages": str(len(languages)),
                "n_harmful": str(sum(chunk.shape[0] for chunk in harmful_chunks)),
                "n_harmless": str(sum(chunk.shape[0] for chunk in harmless_chunks)),
            },
        )
    print(f"[done] pool={cfg.pool}: wrote {len(layers)} pooled-DIM directions to {first_out.parent}")


if __name__ == "__main__":
    main()
