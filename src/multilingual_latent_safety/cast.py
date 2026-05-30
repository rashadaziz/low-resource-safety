"""CAST vector training and inference-time hooks.

This follows IBM's Conditional Activation Steering implementation:

* train behavior and condition vectors as the signed first principal component
  of contrastive activations;
* build the condition projector ``cc^T / c^T c``;
* compare ``cos(h, tanh(proj_c h))`` to a tuned threshold; and
* add the behavior vector only when the prompt-level condition fires.
"""


import contextlib
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Callable

import torch
from torch import nn

from multilingual_latent_safety.interventions import get_decoder_blocks
from multilingual_latent_safety.vector_ops import unit_normalize


def project_onto_direction(hidden: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    direction = direction.to(device=hidden.device, dtype=hidden.dtype)
    return hidden @ direction / direction.norm().clamp(min=1e-12)


def top_pc_power(
    rows: torch.Tensor,
    anchor: torch.Tensor,
    *,
    iterations: int = 80,
) -> torch.Tensor:
    rows = rows.to(dtype=torch.float32)
    vector = anchor.to(dtype=torch.float32)
    if vector.norm() < 1e-12:
        vector = rows[0].clone()
    vector = vector / vector.norm().clamp(min=1e-12)
    for iteration in range(int(iterations)):
        next_vector = rows.T @ (rows @ vector)
        norm = next_vector.norm()
        if norm < 1e-12:
            break
        vector = next_vector / norm
    return vector


def cast_pca_direction(
    positive: torch.Tensor,
    negative: torch.Tensor,
    *,
    method: str = "pca_pairwise",
    power_iterations: int = 80,
) -> torch.Tensor:
    """Return CAST's signed first-PC direction for one layer.

    ``positive`` is the behavior or condition class that should have the
    larger projection after sign alignment. This matches IBM's
    ``SteeringVector.train(..., method=...)`` direction orientation.
    """

    if positive.ndim != 2 or negative.ndim != 2:
        raise ValueError("positive and negative activations must be [n, d]")
    if positive.shape[1] != negative.shape[1]:
        raise ValueError(
            f"hidden-size mismatch: positive={positive.shape}, negative={negative.shape}"
        )
    if positive.shape[0] < 1 or negative.shape[0] < 1:
        raise ValueError("CAST direction training needs both classes")

    n = min(int(positive.shape[0]), int(negative.shape[0]))
    pos = positive[:n].to(dtype=torch.float32)
    neg = negative[:n].to(dtype=torch.float32)
    anchor = pos.mean(dim=0) - neg.mean(dim=0)

    if method == "pca_diff":
        train = pos - neg
    elif method == "pca_center":
        hidden = torch.cat([pos, neg], dim=0)
        train = hidden - hidden.mean(dim=0, keepdim=True)
    elif method == "pca_pairwise":
        center = (pos + neg) / 2.0
        train = torch.empty((2 * n, pos.shape[1]), dtype=torch.float32)
        train[0::2] = pos - center
        train[1::2] = neg - center
    else:
        raise ValueError(
            f"unknown CAST PCA method {method!r}; expected pca_diff, pca_center, or pca_pairwise"
        )

    direction = top_pc_power(train, anchor, iterations=power_iterations)
    pos_scores = project_onto_direction(pos, direction)
    neg_scores = project_onto_direction(neg, direction)
    if (pos_scores < neg_scores).float().mean() > (pos_scores > neg_scores).float().mean():
        direction = -direction
    return direction.to(dtype=torch.float32)


def cast_condition_similarity(
    hidden: torch.Tensor,
    condition_direction: torch.Tensor,
) -> torch.Tensor:
    """Compute CAST ``cos(h, tanh(proj_c h))`` for a batch of prompt states."""

    if hidden.ndim != 2:
        raise ValueError(f"hidden must be [batch, d], got {hidden.shape}")
    direction = condition_direction.to(device=hidden.device, dtype=hidden.dtype)
    denom = torch.dot(direction, direction).clamp(min=1e-12)
    projected = (hidden @ direction / denom).unsqueeze(-1) * direction
    projected = torch.tanh(projected)
    numerator = (hidden * projected).sum(dim=-1)
    hidden_norm = hidden.norm(dim=-1)
    projected_norm = projected.norm(dim=-1)
    return numerator / (hidden_norm * projected_norm).clamp(min=1e-12)


def f1_score_binary(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    y_true = y_true.to(dtype=torch.bool)
    y_pred = y_pred.to(dtype=torch.bool)
    tp = int((y_true & y_pred).sum().item())
    fp = int((~y_true & y_pred).sum().item())
    fn = int((y_true & ~y_pred).sum().item())
    denom = 2 * tp + fp + fn
    if denom == 0:
        return 0.0
    return float(2 * tp / denom)


@dataclass(frozen=True)
class CastConditionPoint:
    layers: tuple[int, ...]
    threshold: float
    comparator_threshold_is: str
    f1: float


def condition_predictions(
    similarities: Mapping[int, torch.Tensor],
    layers: Sequence[int],
    threshold: float,
    comparator_threshold_is: str,
) -> torch.Tensor:
    layer_preds = []
    for layer in layers:
        sim = similarities[int(layer)]
        if comparator_threshold_is == "smaller":
            layer_preds.append(sim > float(threshold))
        elif comparator_threshold_is == "larger":
            layer_preds.append(sim < float(threshold))
        else:
            raise ValueError(
                "comparator_threshold_is must be 'larger' or 'smaller', "
                f"got {comparator_threshold_is!r}"
            )
    return torch.stack(layer_preds, dim=0).any(dim=0)


def select_condition_point(
    positive_similarities: Mapping[int, torch.Tensor],
    negative_similarities: Mapping[int, torch.Tensor],
    *,
    candidate_layers: Sequence[int],
    threshold_min: float = 0.0,
    threshold_max: float = 1.0,
    threshold_step: float = 0.01,
    max_layers_to_combine: int = 1,
) -> CastConditionPoint:
    """Tune CAST condition layer/threshold/direction on validation labels."""

    layers = [int(layer) for layer in candidate_layers]
    if not layers:
        raise ValueError("candidate_layers cannot be empty")
    if max_layers_to_combine != 1:
        raise ValueError("max_layers_to_combine must be 1")

    similarities: dict[int, torch.Tensor] = {}
    for layer in layers:
        similarities[layer] = torch.cat(
            [
                positive_similarities[layer].to(dtype=torch.float32),
                negative_similarities[layer].to(dtype=torch.float32),
            ],
            dim=0,
        )
    y_true = torch.cat(
        [
            torch.ones_like(next(iter(positive_similarities.values())), dtype=torch.bool),
            torch.zeros_like(next(iter(negative_similarities.values())), dtype=torch.bool),
        ],
        dim=0,
    )

    thresholds = torch.arange(
        float(threshold_min),
        float(threshold_max) + float(threshold_step) / 2.0,
        float(threshold_step),
    )
    best = CastConditionPoint((layers[0],), float(thresholds[0].item()), "larger", 0.0)
    for layer in layers:
        for threshold in thresholds.tolist():
            for comparator in ("larger", "smaller"):
                pred = condition_predictions(similarities, [layer], threshold, comparator)
                f1 = f1_score_binary(y_true, pred)
                if f1 > best.f1:
                    best = CastConditionPoint((layer,), round(float(threshold), 6), comparator, f1)
    return best


@dataclass(frozen=True)
class CastBundle:
    behavior_directions: dict[int, torch.Tensor]
    condition_directions: dict[int, torch.Tensor]
    condition_layers: tuple[int, ...]
    behavior_layers: tuple[int, ...]
    threshold: float
    comparator_threshold_is: str
    behavior_strength: float
    condition_mode: str = "mean"
    apply_behavior_on_first_call: bool = True


class CastState:
    def __init__(self, bundle: CastBundle):
        self.bundle = bundle
        self.condition_met: torch.Tensor | None = None

    def reset(self) -> None:
        self.condition_met = None

    def observe_condition(self, layer_idx: int, residual: torch.Tensor) -> None:
        if residual.shape[1] <= 1:
            return
        direction = self.bundle.condition_directions[int(layer_idx)].to(
            device=residual.device,
            dtype=torch.float32,
        )
        if self.bundle.condition_mode == "mean":
            hidden = residual.mean(dim=1)
        elif self.bundle.condition_mode == "last":
            hidden = residual[:, -1, :]
        else:
            raise ValueError(
                f"unknown CAST condition_mode {self.bundle.condition_mode!r}; expected mean or last"
            )
        similarity = cast_condition_similarity(hidden.to(dtype=torch.float32), direction)
        if self.bundle.comparator_threshold_is == "smaller":
            met = similarity > float(self.bundle.threshold)
        elif self.bundle.comparator_threshold_is == "larger":
            met = similarity < float(self.bundle.threshold)
        else:
            raise ValueError(
                "comparator_threshold_is must be 'larger' or 'smaller', "
                f"got {self.bundle.comparator_threshold_is!r}"
            )
        met = met.to(device=residual.device, dtype=torch.bool)
        self.condition_met = met if self.condition_met is None else (self.condition_met | met)

    def apply_behavior(self, layer_idx: int, residual: torch.Tensor) -> torch.Tensor:
        if self.condition_met is None or not bool(self.condition_met.any()):
            return residual
        if residual.shape[1] > 1 and not self.bundle.apply_behavior_on_first_call:
            return residual
        direction = self.bundle.behavior_directions[int(layer_idx)].to(
            device=residual.device,
            dtype=residual.dtype,
        )
        control = float(self.bundle.behavior_strength) * direction.view(1, 1, -1)
        mask = self.condition_met[: residual.shape[0]].to(device=residual.device)
        out = residual.clone()
        out[mask, :, :] = out[mask, :, :] + control
        return out


def make_cast_hook(
    state: CastState,
    layer_idx: int,
    first_layer: int,
) -> Callable[[nn.Module, tuple, Any], Any]:
    def hook(module: nn.Module, inputs: tuple, output: Any) -> Any:
        residual = output[0] if isinstance(output, tuple) else output
        if residual.shape[1] > 1 and layer_idx == first_layer:
            state.reset()
        if layer_idx in state.bundle.condition_layers:
            state.observe_condition(layer_idx, residual)
        if layer_idx in state.bundle.behavior_layers:
            residual = state.apply_behavior(layer_idx, residual)
        if isinstance(output, tuple):
            return (residual, *output[1:])
        return residual

    return hook


@contextlib.contextmanager
def install_cast_hooks(model: nn.Module, bundle: CastBundle) -> Iterator[None]:
    blocks = get_decoder_blocks(model)
    layer_range = sorted(set(bundle.condition_layers) | set(bundle.behavior_layers))
    if not layer_range:
        raise ValueError("CAST requires at least one condition or behavior layer")
    max_layer = max(layer_range)
    if max_layer >= len(blocks):
        raise ValueError(f"CAST layer {max_layer} out of range for {len(blocks)} blocks")
    state = CastState(bundle)
    handles = []
    try:
        first_layer = min(layer_range)
        for layer in layer_range:
            handles.append(
                blocks[int(layer)].register_forward_hook(
                    make_cast_hook(state, int(layer), first_layer)
                )
            )
        yield
    finally:
        for handle in handles:
            handle.remove()
