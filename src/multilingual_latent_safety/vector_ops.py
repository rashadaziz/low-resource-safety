"""Vector normalization and projection helpers."""

import torch


def unit_normalize(tensor: torch.Tensor, *, dim: int | None = -1, eps: float = 1e-12) -> torch.Tensor:
    """Normalize a tensor by L2 norm, optionally along one axis."""
    if dim is None:
        denom = tensor.norm()
    else:
        denom = tensor.norm(dim=dim, keepdim=True)
    return tensor / denom.clamp(min=eps)
