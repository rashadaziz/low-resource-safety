"""PC-1 of the HRL per-language diff-of-means stack, per layer.

For each layer L, load each HRL language's ``v_lang(L) = mu_harmful - mu_harmless`` direction
(as already written by ``scripts/hrl_direction/compute_harmful_direction.py``), stack them into ``(n_langs, d)``,
take the top right-singular vector of the **uncentered** stack, sign-align with the English direction,
and save under ``pooled_direction_file``.

Why uncentered SVD (rather than mean-centered PCA): the stacked HRL diff-of-means vectors are highly
correlated (pairwise cosines near 1), so mean-centering extracts the direction of maximum
disagreement across languages. The dominant shared direction, which is what we want, is the top
right-singular vector of the uncentered stack.

Sign convention: v_HRL is explicitly sign-aligned with v_en (English diff-of-means) so that the two
anchors point in the same direction — positive projection means "more harmful" for both. Without
this alignment, SVD's sign choice is arbitrary, and that would corrupt the v_en-vs-v_HRL comparison
under directional ablation ``h <- h - (h . v_hat) v_hat``: ablation is sign-invariant at the
single-direction level (because the outer product ``v v^T`` is the same under ``v -> -v``), but any
downstream projection / logging that treats "positive" as "harmful" would silently flip.
"""

from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from safetensors.torch import save_file

from multilingual_latent_safety.analysis import direction_layer_ids, load_direction_stack
from multilingual_latent_safety.paths import pooled_direction_file


def estimate_pc1_of_stack(directions: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
    """Top right-singular vector of the (uncentered) stack, sign-aligned with ``anchor``."""
    _, _, vh = torch.linalg.svd(directions, full_matrices=False)
    v = vh[0]
    if v @ anchor < 0:
        v = -v
    return v


@hydra.main(version_base=None, config_path="../../configs", config_name="hrl_direction/compute_hrl_pooled_direction")
def main(cfg: DictConfig) -> None:
    input_root = Path(cfg.input_root)
    output_root = Path(cfg.output_root)
    languages = list(cfg.languages)
    if not languages:
        raise ValueError("cfg.languages is empty")
    if "en" not in languages:
        raise ValueError(
            f"cfg.languages must contain 'en' so v_HRL can be sign-aligned with v_en; got {languages}"
        )
    en_idx = languages.index("en")
    layers = direction_layer_ids(input_root, languages[0], cfg.split, cfg.token_position)
    if not layers:
        raise FileNotFoundError(
            f"No per-language directions at {input_root}/{languages[0]}/{cfg.split}/tok={cfg.token_position}"
        )
    first_out = pooled_direction_file(output_root, cfg.pool, cfg.token_position, layers[0])
    first_out.parent.mkdir(parents=True, exist_ok=True)

    for layer in layers:
        stack = load_direction_stack(input_root, languages, cfg.split, cfg.token_position, layer)
        direction = estimate_pc1_of_stack(stack, anchor=stack[en_idx])
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
                "method": "pc1_of_diffs",
                "input_method": str(cfg.input_method),
                "normalize": str(cfg.normalize),
                "n_languages": str(len(languages)),
                "sign_anchor_language": "en",
            },
        )
    print(f"[done] pool={cfg.pool}: wrote {len(layers)} directions to {first_out.parent}")


if __name__ == "__main__":
    main()
