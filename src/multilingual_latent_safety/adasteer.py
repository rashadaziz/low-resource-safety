"""AdaSteer vector loading and generation-time hooks.

The released AdaSteer implementation computes prompt-level coefficients from
the final prompt-token residual, then applies two layer-wise steering vectors
during generated-token decoding. This module keeps that behavior but implements
it with regular Hugging Face forward hooks instead of patched model classes.
"""


import contextlib
import pickle
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn

from multilingual_latent_safety.interventions import get_decoder_blocks
from multilingual_latent_safety.vector_ops import unit_normalize


@dataclass(frozen=True)
class AdaSteerSpec:
    key: str
    alpha_layer: int
    beta_layer: int
    alpha_multiplier: float
    alpha_offset: float
    alpha_bias: float
    alpha_min: float | None
    alpha_max: float | None
    beta_multiplier: float
    beta_offset: float
    beta_bias: float
    beta_min: float | None
    beta_max: float | None
    standard: float = 100.0


ADASTEER_SPECS: dict[str, AdaSteerSpec] = {
    "meta-llama/llama-3.1-8b-instruct": AdaSteerSpec(
        key="llama31-8b-instruct",
        alpha_layer=8,
        beta_layer=13,
        alpha_multiplier=0.02,
        alpha_offset=60.0,
        alpha_bias=0.0,
        alpha_min=-0.22,
        alpha_max=-0.08,
        beta_multiplier=-0.05,
        beta_offset=5.0,
        beta_bias=-0.5,
        beta_min=-0.75,
        beta_max=None,
    ),
    "qwen/qwen2.5-7b-instruct": AdaSteerSpec(
        key="qwen25-7b-instruct",
        alpha_layer=5,
        beta_layer=13,
        alpha_multiplier=0.1,
        alpha_offset=-140.0,
        alpha_bias=0.0,
        alpha_min=-0.2,
        alpha_max=0.0,
        beta_multiplier=-0.06,
        beta_offset=-50.0,
        beta_bias=0.0,
        beta_min=-0.6,
        beta_max=0.4,
    ),
    "google/gemma-2-9b-it": AdaSteerSpec(
        key="gemma2-9b-it",
        alpha_layer=12,
        beta_layer=19,
        alpha_multiplier=0.004,
        alpha_offset=-35.0,
        alpha_bias=0.0,
        alpha_min=-0.02,
        alpha_max=0.06,
        beta_multiplier=0.01,
        beta_offset=-50.0,
        beta_bias=0.0,
        beta_min=-0.06,
        beta_max=0.02,
    ),
}


@dataclass(frozen=True)
class AdaSteerBundle:
    spec: AdaSteerSpec
    rd_direction: torch.Tensor
    hd_direction: torch.Tensor
    rd_harmful_anchors: torch.Tensor
    rd_harmless_anchors: torch.Tensor
    hd_harmful_anchors: torch.Tensor
    hd_harmless_anchors: torch.Tensor


class AdaSteerState:
    def __init__(self, bundle: AdaSteerBundle):
        self.bundle = bundle
        self.alpha: torch.Tensor | None = None
        self.beta: torch.Tensor | None = None

    def reset(self) -> None:
        self.alpha = None
        self.beta = None

    def observe_prefill(self, layer_idx: int, residual: torch.Tensor) -> None:
        if residual.shape[1] <= 1:
            return
        spec = self.bundle.spec
        hidden = residual[:, -1, :].detach().to(dtype=torch.float32)
        if layer_idx == spec.alpha_layer:
            scaled = scaled_harmful_distance(
                hidden,
                self.bundle.rd_harmful_anchors[layer_idx],
                self.bundle.rd_harmless_anchors[layer_idx],
                standard=spec.standard,
            )
            self.alpha = linear_clamped(
                scaled,
                multiplier=spec.alpha_multiplier,
                offset=spec.alpha_offset,
                bias=spec.alpha_bias,
                min_value=spec.alpha_min,
                max_value=spec.alpha_max,
            )
        if layer_idx == spec.beta_layer:
            scaled = scaled_harmful_distance(
                hidden,
                self.bundle.hd_harmful_anchors[layer_idx],
                self.bundle.hd_harmless_anchors[layer_idx],
                standard=spec.standard,
            )
            self.beta = linear_clamped(
                scaled,
                multiplier=spec.beta_multiplier,
                offset=spec.beta_offset,
                bias=spec.beta_bias,
                min_value=spec.beta_min,
                max_value=spec.beta_max,
            )

    def apply_generation(self, layer_idx: int, residual: torch.Tensor) -> torch.Tensor:
        if residual.shape[1] != 1 or self.alpha is None or self.beta is None:
            return residual
        if layer_idx >= self.bundle.rd_direction.shape[0]:
            raise IndexError(
                f"AdaSteer vector missing layer {layer_idx}; "
                f"only {self.bundle.rd_direction.shape[0]} layers available"
            )
        batch_size = residual.shape[0]
        device = residual.device
        dtype = residual.dtype
        alpha = self.alpha[:batch_size].to(device=device, dtype=dtype).view(batch_size, 1, 1)
        beta = self.beta[:batch_size].to(device=device, dtype=dtype).view(batch_size, 1, 1)
        rd = self.bundle.rd_direction[layer_idx].to(device=device, dtype=dtype).view(1, 1, -1)
        hd = self.bundle.hd_direction[layer_idx].to(device=device, dtype=dtype).view(1, 1, -1)
        if rd.shape[-1] != residual.shape[-1] or hd.shape[-1] != residual.shape[-1]:
            raise ValueError(
                "AdaSteer vector hidden size mismatch: "
                f"residual={residual.shape[-1]}, rd={rd.shape[-1]}, hd={hd.shape[-1]}"
            )
        return residual + alpha * rd + beta * hd


def adasteer_spec_for_model(model_name: str) -> AdaSteerSpec:
    key = model_name.rstrip("/").lower()
    if key in ADASTEER_SPECS:
        return ADASTEER_SPECS[key]
    supported = ", ".join(sorted(spec.key for spec in ADASTEER_SPECS.values()))
    raise ValueError(
        f"AdaSteer has no released vectors for model {model_name!r}. "
        f"Supported vector sets: {supported}"
    )


def load_adasteer_bundle(root: str | Path, model_name: str) -> AdaSteerBundle:
    spec = adasteer_spec_for_model(model_name)
    base = Path(root) / spec.key
    rd_direction = load_tensor(base / "RD" / "mean_diff.pkl")
    hd_direction = load_tensor(base / "HD" / "proj.pkl")
    rd_harmful = mean_class_anchors(load_tensor(base / "RD" / "class_a.pkl"))
    rd_harmless = mean_class_anchors(load_tensor(base / "RD" / "class_b.pkl"))
    hd_harmful = mean_class_anchors(load_tensor(base / "HD" / "class_a.pkl"))
    hd_harmless = mean_class_anchors(load_tensor(base / "HD" / "class_b.pkl"))
    return AdaSteerBundle(
        spec=spec,
        rd_direction=rd_direction,
        hd_direction=hd_direction,
        rd_harmful_anchors=rd_harmful,
        rd_harmless_anchors=rd_harmless,
        hd_harmful_anchors=hd_harmful,
        hd_harmless_anchors=hd_harmless,
    )


def make_adasteer_hook(
    state: AdaSteerState,
    layer_idx: int,
    first_layer: int,
) -> Callable[[nn.Module, tuple, Any], Any]:
    def hook(module: nn.Module, inputs: tuple, output: Any) -> Any:
        residual = output[0] if isinstance(output, tuple) else output
        if residual.shape[1] > 1 and layer_idx == first_layer:
            state.reset()
        state.observe_prefill(layer_idx, residual)
        steered = state.apply_generation(layer_idx, residual)
        if isinstance(output, tuple):
            return (steered, *output[1:])
        return steered

    return hook


@contextlib.contextmanager
def install_adasteer_hooks(
    model: nn.Module,
    bundle: AdaSteerBundle,
    layer_range: Sequence[int],
) -> Iterator[None]:
    blocks = get_decoder_blocks(model)
    if not layer_range:
        raise ValueError("AdaSteer requires at least one layer hook")
    max_layer = max(layer_range)
    n_vector_layers = int(bundle.rd_direction.shape[0])
    if max_layer >= n_vector_layers:
        raise ValueError(
            f"AdaSteer vector set has {n_vector_layers} layers, "
            f"but layer_range requests layer {max_layer}"
        )
    state = AdaSteerState(bundle)
    first_layer = min(layer_range)
    handles = []
    try:
        for layer_idx in layer_range:
            hook = make_adasteer_hook(state, int(layer_idx), first_layer)
            handles.append(blocks[int(layer_idx)].register_forward_hook(hook))
        yield
    finally:
        for handle in handles:
            handle.remove()


def load_tensor(path: Path) -> torch.Tensor:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing AdaSteer vector file: {path}. "
            "Run scripts/routing_refusal/download_adasteer_vectors.py first."
        )
    with path.open("rb") as f:
        value = pickle.load(f)
    return torch.as_tensor(value, dtype=torch.float32)


def mean_class_anchors(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim != 3:
        raise ValueError(f"expected AdaSteer class anchor tensor [layers, examples, d], got {tensor.shape}")
    return tensor.mean(dim=1)


def scaled_harmful_distance(
    hidden: torch.Tensor,
    harmful_anchor: torch.Tensor,
    harmless_anchor: torch.Tensor,
    standard: float,
) -> torch.Tensor:
    harmful_anchor = harmful_anchor.to(device=hidden.device, dtype=hidden.dtype)
    harmless_anchor = harmless_anchor.to(device=hidden.device, dtype=hidden.dtype)
    acceptance_direction = unit_normalize(harmless_anchor - harmful_anchor)
    harmful_distance = (hidden - harmful_anchor) @ acceptance_direction
    harmless_distance = (hidden - harmless_anchor) @ acceptance_direction
    scale = (harmful_distance - harmless_distance).mean().clamp(min=1e-12)
    return harmful_distance / scale * float(standard)


def linear_clamped(
    scaled: torch.Tensor,
    multiplier: float,
    offset: float,
    bias: float,
    min_value: float | None,
    max_value: float | None,
) -> torch.Tensor:
    coeff = float(multiplier) * (scaled + float(offset)) + float(bias)
    if min_value is not None:
        coeff = torch.clamp(coeff, min=float(min_value))
    if max_value is not None:
        coeff = torch.clamp(coeff, max=float(max_value))
    return coeff.detach()
