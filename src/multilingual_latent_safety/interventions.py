"""Inference-time interventions on the residual stream.

Hooks installed on each transformer block's ``.output`` (post-block residual for HF causal LMs):

- ``directional_ablation``  -- ``h <- h - (h dot v_hat) v_hat`` (Arditi et al. 2024).
- ``directional_addition``  -- ``h <- h + alpha v_hat`` for refusal-activation sweeps.
- ``conditional_refusal``    -- harmful rows get final-prompt+generation addition;
  harmless rows get all-token ablation.
- ``patch``                  -- replace the residual at the last prompt token with a cached
  activation from a different forward pass (classical activation patching).

Each modifying hook takes a ``token_scope`` argument controlling where on the sequence axis
it fires. Detection is stateless: prefill steps have ``residual.shape[1] > 1``, generation
steps (with KV cache) have ``shape[1] == 1``.

- ``"all_tokens"`` -- modify every position in every step.
- ``"final_prompt_then_gen"`` -- prefill: only the last position; generation: every step.
- ``"final_prompt_only"`` -- prefill: only the last position; generation: no-op.
"""


import contextlib
from collections.abc import Iterator, Sequence
from typing import Any, Callable

import torch
from torch import nn

from multilingual_latent_safety.vector_ops import unit_normalize


TokenScope = str
VALID_TOKEN_SCOPES = ("all_tokens", "final_prompt_then_gen", "final_prompt_only")


def apply_with_scope(
    residual: torch.Tensor,
    modify: Callable[[torch.Tensor], torch.Tensor],
    scope: TokenScope,
) -> torch.Tensor:
    """Dispatch ``modify`` over the sequence axis according to ``scope``.

    ``modify`` receives a ``[B, T', d]`` slice and must return the same shape.
    """
    is_prefill = residual.shape[1] > 1
    if scope == "all_tokens":
        return modify(residual)
    if scope == "final_prompt_then_gen":
        if is_prefill:
            modified_last = modify(residual[:, -1:, :])
            return torch.cat([residual[:, :-1, :], modified_last], dim=1)
        return modify(residual)
    if scope == "final_prompt_only":
        if is_prefill:
            modified_last = modify(residual[:, -1:, :])
            return torch.cat([residual[:, :-1, :], modified_last], dim=1)
        return residual
    raise ValueError(f"Unknown token_scope {scope!r}; expected one of {VALID_TOKEN_SCOPES}")


def make_directional_ablation_hook(
    direction: torch.Tensor, device: torch.device, dtype: torch.dtype,
    token_scope: TokenScope = "all_tokens",
) -> Callable[[nn.Module, tuple, Any], Any]:
    v = unit_normalize(direction.to(device=device, dtype=dtype))

    def modify(residual: torch.Tensor) -> torch.Tensor:
        coeffs = residual @ v
        return residual - coeffs.unsqueeze(-1) * v

    def hook(module: nn.Module, inputs: tuple, output: Any) -> Any:
        residual = output[0] if isinstance(output, tuple) else output
        ablated = apply_with_scope(residual, modify, token_scope)
        if isinstance(output, tuple):
            return (ablated, *output[1:])
        return ablated

    return hook


def make_directional_addition_hook(
    direction: torch.Tensor,
    alpha: float,
    device: torch.device,
    dtype: torch.dtype,
    token_scope: TokenScope = "final_prompt_only",
) -> Callable[[nn.Module, tuple, Any], Any]:
    """Add ``alpha`` units of a normalized direction to selected residual positions."""
    v = unit_normalize(direction.to(device=device, dtype=dtype))
    alpha_t = torch.tensor(float(alpha), device=device, dtype=dtype)

    def modify(residual: torch.Tensor) -> torch.Tensor:
        return residual + alpha_t * v

    def hook(module: nn.Module, inputs: tuple, output: Any) -> Any:
        residual = output[0] if isinstance(output, tuple) else output
        shifted = apply_with_scope(residual, modify, token_scope)
        if isinstance(output, tuple):
            return (shifted, *output[1:])
        return shifted

    return hook


def make_conditional_refusal_hook(
    direction: torch.Tensor,
    alpha: float,
    harmful_mask: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> Callable[[nn.Module, tuple, Any], Any]:
    """Condition on a batch-level harmfulness gate.

    Rows marked harmful receive ``h <- h + alpha v_hat`` on the final prompt
    token and all generation steps. Rows marked harmless receive directional
    ablation ``h <- h - (h dot v_hat) v_hat`` at all tokens and generation steps.
    """
    v = unit_normalize(direction.to(device=device, dtype=dtype))
    alpha_t = torch.tensor(float(alpha), device=device, dtype=dtype)
    mask = harmful_mask.to(device=device, dtype=torch.bool).flatten()

    def ablate(residual: torch.Tensor) -> torch.Tensor:
        coeffs = residual @ v
        return residual - coeffs.unsqueeze(-1) * v

    def hook(module: nn.Module, inputs: tuple, output: Any) -> Any:
        residual = output[0] if isinstance(output, tuple) else output
        if residual.shape[0] != mask.numel():
            raise ValueError(
                f"batch mismatch: residual B={residual.shape[0]} vs harmful_mask B={mask.numel()}"
            )
        modified = residual.clone()
        harmful = mask
        harmless = ~mask
        if bool(harmless.any()):
            modified[harmless, :, :] = ablate(residual[harmless, :, :])
        if bool(harmful.any()):
            if residual.shape[1] > 1:
                modified[harmful, -1:, :] = residual[harmful, -1:, :] + alpha_t * v
            else:
                modified[harmful, :, :] = residual[harmful, :, :] + alpha_t * v
        if isinstance(output, tuple):
            return (modified, *output[1:])
        return modified

    return hook


def make_patch_hook(
    replacement: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    token_scope: TokenScope = "final_prompt_only",
) -> Callable[[nn.Module, tuple, Any], Any]:
    """Replace the last-prompt-token residual with ``replacement`` (shape ``[B, d_model]``).

    Classical activation patching: the entire residual at position T-1 is overwritten with a
    pre-computed activation from another forward pass. Only ``final_prompt_only`` is supported
    -- the patch fires once at prefill, and generation proceeds from the patched residual via
    KV cache.
    """
    if token_scope != "final_prompt_only":
        raise ValueError(
            f"make_patch_hook only supports token_scope='final_prompt_only', got {token_scope!r}"
        )
    rep = replacement.to(device=device, dtype=dtype)

    def modify(residual: torch.Tensor) -> torch.Tensor:
        if residual.shape[0] != rep.shape[0]:
            raise ValueError(
                f"batch mismatch: residual B={residual.shape[0]} vs replacement B={rep.shape[0]}"
            )
        return rep.unsqueeze(1).expand_as(residual).contiguous()

    def hook(module: nn.Module, inputs: tuple, output: Any) -> Any:
        residual = output[0] if isinstance(output, tuple) else output
        patched = apply_with_scope(residual, modify, token_scope)
        if isinstance(output, tuple):
            return (patched, *output[1:])
        return patched

    return hook


def get_decoder_blocks(model: nn.Module) -> nn.ModuleList:
    """Return the list of transformer blocks for HF causal LMs (Qwen, Llama, Gemma share this layout)."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise AttributeError(
        f"Cannot locate transformer blocks on {type(model).__name__}; add a dispatch case here"
    )


@contextlib.contextmanager
def install_hooks(
    model: nn.Module,
    layer_range: Sequence[int],
    hook_factory: Callable[[torch.device, torch.dtype], Callable[..., Any]],
) -> Iterator[None]:
    """Install ``hook_factory`` on every block in ``layer_range`` and remove on exit."""
    blocks = get_decoder_blocks(model)
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    handles = []
    try:
        for layer_idx in layer_range:
            hook = hook_factory(device, dtype)
            handles.append(blocks[layer_idx].register_forward_hook(hook))
        yield
    finally:
        for h in handles:
            h.remove()
