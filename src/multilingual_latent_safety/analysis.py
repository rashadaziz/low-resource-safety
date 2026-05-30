"""Artifact loading and numerical analysis helpers."""

from collections.abc import Sequence
import math
from pathlib import Path

import torch
import yaml
from safetensors.torch import load_file

from multilingual_latent_safety.paths import (
    activations_dir,
    activations_file,
    direction_dir,
    direction_file,
)


def mean(values: Sequence[float]) -> float:
    """Arithmetic mean for small in-memory metric groups."""
    return sum(values) / len(values)


def stderr(values: Sequence[float]) -> float:
    """Unbiased standard error; returns 0 for singleton groups."""
    if len(values) <= 1:
        return 0.0
    mu = mean(values)
    var = sum((value - mu) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(var) / math.sqrt(len(values))


def load_resource_tiers(dataset_config: str | Path = "configs/dataset/polyrefuse.yaml") -> dict[str, str]:
    """Read the ``resource_tier`` mapping from a dataset config."""
    with open(dataset_config) as f:
        return dict(yaml.safe_load(f)["resource_tier"])


def token_index(extracted_positions_csv: str, requested_position: str | int) -> int:
    """Map a requested token position spec to its index inside a stored activation's token-position axis.

    Positions may be symbolic (``t_inst``, ``t_post_inst``) or numeric strings (``"-1"``); the comparison
    is by string equality so callers don't need to know the encoding used at extraction time.
    """
    positions = [p.strip() for p in extracted_positions_csv.split(",")]
    requested = str(requested_position)
    if requested not in positions:
        raise ValueError(
            f"Requested token position {requested!r} not present in extracted positions {positions}"
        )
    return positions.index(requested)


def load_subset_activations(
    acts_root: str | Path,
    language: str,
    split: str,
    subset: str,
    layer: int,
    tok_idx: int,
) -> torch.Tensor:
    """Load activations at a single token-position index. Returns shape ``(N, d)`` as float32."""
    path = activations_file(acts_root, language, split, subset, layer)
    loaded = load_file(path)
    return loaded["activations"][:, tok_idx, :].to(torch.float32)


def layer_ids(acts_root: str | Path, language: str, split: str, subset: str) -> list[int]:
    """Layer IDs present in an activations directory, sorted ascending."""
    d = activations_dir(acts_root, language, split, subset)
    return sorted(int(p.stem.split("_")[-1]) for p in d.glob("layer_*.safetensors"))


def direction_layer_ids(
    dir_root: str | Path, language: str, split: str, token_position: str | int
) -> list[int]:
    """Layer IDs present in a directions directory, sorted ascending."""
    d = direction_dir(dir_root, language, split, token_position)
    return sorted(int(p.stem.split("_")[-1]) for p in d.glob("layer_*.safetensors"))


def load_direction(
    dir_root: str | Path,
    language: str,
    split: str,
    token_position: str | int,
    layer: int,
) -> torch.Tensor:
    """Load a single harmful direction. Returns shape ``(d,)`` as float32."""
    path = direction_file(dir_root, language, split, token_position, layer)
    return load_file(path)["direction"].to(torch.float32)


def load_direction_stack(
    dir_root: str | Path,
    languages: list[str],
    split: str,
    token_position: str | int,
    layer: int,
) -> torch.Tensor:
    """Stack directions for the given languages into a ``(n_langs, d)`` tensor. Fails if any is missing."""
    return torch.stack(
        [load_direction(dir_root, lang, split, token_position, layer) for lang in languages],
        dim=0,
    )


def cosine_similarity_matrix(dirs: torch.Tensor) -> torch.Tensor:
    """Standard cosine-similarity matrix for a ``(n, d)`` stack of directions."""
    normed = dirs / dirs.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    return normed @ normed.T


def mahalanobis_cosine_similarity(dirs: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Cosine similarity after whitening by ``Σ^{-1/2}``.

    Uses Cholesky ``Σ = L L^T`` and triangular solve to whiten each direction (``w_i = L^{-1} d_i``),
    then takes plain cosine on the whitened stack. This is more numerically stable than computing ``Σ^{-1}``
    via ``linalg.inv`` and guarantees the output is exactly symmetric.
    """
    l = torch.linalg.cholesky(sigma)
    whitened = torch.linalg.solve_triangular(l, dirs.T, upper=False).T
    return cosine_similarity_matrix(whitened)


def covariance(x: torch.Tensor, shrinkage: float = 1e-3) -> torch.Tensor:
    """Sample covariance of ``x`` of shape ``(N, d)`` with ridge shrinkage ``Σ + λ·(tr(Σ)/d)·I`` for stable inversion."""
    x = x.to(torch.float32)
    centered = x - x.mean(dim=0, keepdim=True)
    n = x.shape[0]
    d = x.shape[1]
    sigma = centered.T @ centered / max(n - 1, 1)
    lam = shrinkage * sigma.diagonal().mean()
    return sigma + lam * torch.eye(d, dtype=sigma.dtype, device=sigma.device)


def projected_scores(activations: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    """Project ``(N, d)`` activations onto a ``(d,)`` direction."""
    return activations.to(torch.float32) @ direction.to(torch.float32)


def auc(harmful_scores: torch.Tensor, harmless_scores: torch.Tensor) -> float:
    """ROC-AUC via Mann-Whitney U with harmful treated as positive class."""
    nh = harmful_scores.shape[0]
    nhl = harmless_scores.shape[0]
    if nh == 0 or nhl == 0:
        raise ValueError(f"empty class: nh={nh}, nhl={nhl}")
    combined = torch.cat([harmful_scores, harmless_scores]).to(torch.float64)
    order = torch.argsort(combined)
    ranks = torch.empty_like(combined)
    ranks[order] = torch.arange(
        1,
        combined.shape[0] + 1,
        dtype=combined.dtype,
        device=combined.device,
    )
    u = ranks[:nh].sum() - nh * (nh + 1) / 2
    return float(u / (nh * nhl))


def orthogonal_mass_fraction(directions: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
    """For a ``(n, d)`` stack and a ``(d,)`` anchor direction, return ``(n,)`` fraction of each direction's
    L2 mass orthogonal to the anchor.

    A value close to 0 means a direction is nearly colinear with the anchor (strongly aligned). A value
    close to 1 means the direction is essentially orthogonal to the anchor (no shared component).
    """
    directions = directions.to(torch.float32)
    anchor = anchor.to(torch.float32)
    anchor_norm = anchor.norm().clamp(min=1e-12)
    anchor_hat = anchor / anchor_norm
    coeffs = directions @ anchor_hat
    parallel = coeffs.unsqueeze(-1) * anchor_hat.unsqueeze(0)
    perp = directions - parallel
    dir_norms = directions.norm(dim=-1).clamp(min=1e-12)
    return perp.norm(dim=-1) / dir_norms


def auc_gap_per_language(auc_matrix: torch.Tensor, anchor_idx: int) -> torch.Tensor:
    """Given an ``(n, n)`` AUC matrix with rows=direction-language and cols=test-language, return a
    ``(n,)`` vector of ``AUC_self - AUC_anchor`` for each test language.

    Positive values indicate that the language's own direction separates harmful/harmless better
    than the anchor (typically English) direction — evidence for language-specific encoding.
    """
    self_aucs = auc_matrix.diagonal()
    anchor_aucs = auc_matrix[anchor_idx]
    return self_aucs - anchor_aucs


def bootstrap_auc_gap(
    harmful_self: torch.Tensor,
    harmless_self: torch.Tensor,
    harmful_anchor: torch.Tensor,
    harmless_anchor: torch.Tensor,
    n_resamples: int = 1000,
    seed: int = 42,
) -> torch.Tensor:
    """Paired bootstrap of the AUC-gap metric ``AUC_self − AUC_anchor``.

    Samples harmful and harmless indices jointly with replacement so the same test items are used
    for both projections in each resample. Returns a ``(n_resamples,)`` tensor of gap values.
    """
    nh = harmful_self.shape[0]
    nhl = harmless_self.shape[0]
    if harmful_anchor.shape[0] != nh or harmless_anchor.shape[0] != nhl:
        raise ValueError("paired bootstrap requires equal-sized harmful/harmless for self and anchor")
    g = torch.Generator().manual_seed(int(seed))
    gaps = torch.empty(n_resamples, dtype=torch.float64)
    for b in range(n_resamples):
        h_idx = torch.randint(0, nh, (nh,), generator=g)
        l_idx = torch.randint(0, nhl, (nhl,), generator=g)
        gaps[b] = auc(harmful_self[h_idx], harmless_self[l_idx]) - auc(harmful_anchor[h_idx], harmless_anchor[l_idx])
    return gaps
