"""Activation extraction helpers for causal and probe experiments."""

from collections.abc import Sequence
from typing import Any

import torch
from nnsight import LanguageModel
from tqdm import tqdm

from multilingual_latent_safety.model import DTYPE_MAP, format_prompt, model_input_device, transformer_layers
from multilingual_latent_safety.runtime import batched

SYMBOLIC_POSITIONS = ("t_inst", "t_post_inst")
POSITION_PROBE = "XPROBEX"


def resolve_layers(layers_cfg: Any, num_layers: int) -> list[int]:
    """Resolve a layer selector (``"all"``, a list of indices, or a ``{start, end, step}`` mapping) to concrete indices."""
    if layers_cfg == "all":
        return list(range(num_layers))
    try:
        items = list(layers_cfg)
        if items and not hasattr(layers_cfg, "keys"):
            return [int(i) % num_layers for i in items]
    except TypeError:
        pass
    if hasattr(layers_cfg, "keys"):
        start = layers_cfg.get("start", 0)
        end = layers_cfg.get("end", num_layers)
        step = layers_cfg.get("step", 1)
        return list(range(start, end, step))
    raise ValueError(f"Unsupported layers config: {layers_cfg!r}")


def resolve_token_positions(tokenizer, chat_cfg: Any, specs: Sequence[Any]) -> list[int]:
    """Resolve a list of token-position specs (symbolic names or ints) to concrete negative indices.

    Supported symbolic names:
        - ``t_inst``: last token of the user instruction (before chat-template suffix)
        - ``t_post_inst``: last token of the full prompt (equivalent to ``-1``)
    Integer specs (or numeric strings) pass through unchanged.

    Symbolic resolution probes the tokenizer by formatting a single-token placeholder through
    the chat template and locating it via ``offset_mapping``. Returns a list of negative indices
    so the same offsets work regardless of per-prompt length under left padding.
    """
    specs = list(specs)
    needs_probe = any(str(s) in SYMBOLIC_POSITIONS for s in specs)
    t_inst_from_end: int | None = None
    if needs_probe:
        probed_prompt = format_prompt(tokenizer, POSITION_PROBE, chat_cfg)
        encoding = tokenizer(probed_prompt, return_offsets_mapping=True, add_special_tokens=False)
        probe_char_idx = probed_prompt.rfind(POSITION_PROBE)
        probe_end_char = probe_char_idx + len(POSITION_PROBE)
        last_probe_token: int | None = None
        for i, (start, end) in enumerate(encoding.offset_mapping):
            if start < probe_end_char and end > probe_char_idx:
                last_probe_token = i
        if last_probe_token is None:
            raise ValueError("resolve_token_positions: probe character not found in chat template")
        total_len = len(encoding.input_ids)
        t_inst_from_end = last_probe_token - total_len

    indices: list[int] = []
    for spec in specs:
        s = str(spec)
        if s == "t_inst":
            assert t_inst_from_end is not None
            indices.append(t_inst_from_end)
        elif s == "t_post_inst":
            indices.append(-1)
        elif s.lstrip("-").isdigit():
            indices.append(int(s))
        else:
            raise ValueError(f"resolve_token_positions: unknown spec {spec!r}")
    return indices


def hook_module(lm: LanguageModel, hook_point: str, layer_idx: int) -> Any:
    """Return the nnsight proxy for the requested hook point on a given transformer block."""
    block = lm.model.layers[layer_idx]
    if hook_point == "residual_stream":
        return block.output
    if hook_point == "attn_out":
        return block.self_attn.output[0]
    if hook_point == "mlp_out":
        return block.mlp.output
    raise ValueError(f"Unknown hook_point: {hook_point}")


def extract(
    lm: LanguageModel,
    prompts: Sequence[str],
    layers: Sequence[int],
    token_positions: Sequence[int],
    hook_point: str = "residual_stream",
    batch_size: int = 8,
    storage_dtype: str = "float32",
) -> dict[int, torch.Tensor]:
    """Run ``lm`` on ``prompts`` in batches and return per-layer activations of shape ``(N, |token_positions|, d_model)``."""
    tok_positions = list(token_positions)
    store_dtype = DTYPE_MAP[storage_dtype]
    per_layer: dict[int, list[torch.Tensor]] = {l: [] for l in layers}
    for batch in tqdm(list(batched(prompts, batch_size)), desc=f"extract[{hook_point}]"):
        with lm.trace(batch):
            layer_pieces = []
            for l in layers:
                h = hook_module(lm, hook_point, l)
                layer_pieces.append(torch.stack([h[:, p, :] for p in tok_positions], dim=1))
            all_acts = torch.stack(layer_pieces, dim=0).save()
        tensor = all_acts.detach().to(store_dtype).cpu()
        for i, l in enumerate(layers):
            per_layer[l].append(tensor[i].clone())
        del all_acts, tensor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return {l: torch.cat(per_layer[l], dim=0) for l in layers}


def prompt_token_lengths(tokenizer, prompts: Sequence[str]) -> list[int]:
    """Token counts used for prompt-mean pooling."""
    lengths: list[int] = []
    for prompt in prompts:
        encoding = tokenizer(prompt, add_special_tokens=False)
        lengths.append(len(encoding.input_ids))
    return lengths


def extract_mean_pooled(
    lm: LanguageModel,
    prompts: Sequence[str],
    layers: Sequence[int],
    hook_point: str = "residual_stream",
    batch_size: int = 8,
    storage_dtype: str = "float32",
) -> dict[int, torch.Tensor]:
    """Run ``lm`` and return per-layer mean-pooled prompt representations of shape ``(N, d_model)``."""
    store_dtype = DTYPE_MAP[storage_dtype]
    per_layer: dict[int, list[torch.Tensor]] = {l: [] for l in layers}
    lengths = prompt_token_lengths(lm.tokenizer, prompts)
    batches = list(zip(batched(prompts, batch_size), batched(lengths, batch_size), strict=True))
    for batch, batch_lengths in tqdm(batches, desc=f"extract_mean[{hook_point}]"):
        batch = list(batch)
        batch_lengths = [int(length) for length in batch_lengths]
        with lm.trace(batch):
            layer_pieces = []
            for l in layers:
                h = hook_module(lm, hook_point, l)
                pooled = torch.stack(
                    [h[i, -batch_lengths[i] :, :].mean(dim=0) for i in range(len(batch))],
                    dim=0,
                )
                layer_pieces.append(pooled)
            all_acts = torch.stack(layer_pieces, dim=0).save()
        tensor = all_acts.detach().to(store_dtype).cpu()
        for i, l in enumerate(layers):
            per_layer[l].append(tensor[i].clone())
        del all_acts, tensor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return {l: torch.cat(per_layer[l], dim=0) for l in layers}


@torch.inference_mode()
def extract_mean_pooled_transformers(
    *,
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    layers: Sequence[int],
    batch_size: int,
    storage_dtype: str,
) -> dict[int, torch.Tensor]:
    """Mean-pool prompt representations from a plain transformers model."""
    out_dtype = DTYPE_MAP[storage_dtype]
    layer_ids = list(layers)
    chunks: dict[int, list[torch.Tensor]] = {layer: [] for layer in layer_ids}
    input_device = model_input_device(model)
    layer_stack = transformer_layers(model)
    selected = sorted(set(layer_ids))
    if selected[-1] >= len(layer_stack):
        raise ValueError(
            f"requested layer {selected[-1]} but model only has {len(layer_stack)} layers"
        )
    captured: dict[int, torch.Tensor] = {}

    def make_hook(layer: int):
        def hook(module: torch.nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            captured[layer] = hidden

        return hook

    handles = [layer_stack[layer].register_forward_hook(make_hook(layer)) for layer in selected]
    try:
        for start in tqdm(range(0, len(prompts), batch_size), desc="extract_mean[transformers]"):
            captured.clear()
            batch_prompts = list(prompts[start : start + batch_size])
            encoded = tokenizer(
                batch_prompts,
                padding=True,
                return_tensors="pt",
                add_special_tokens=False,
            )
            encoded = {key: value.to(input_device) for key, value in encoded.items()}
            model(
                **encoded,
                output_hidden_states=False,
                use_cache=False,
                return_dict=True,
            )
            missing = [layer for layer in selected if layer not in captured]
            if missing:
                raise RuntimeError(f"failed to capture transformer layers: {missing}")
            mask = encoded["attention_mask"].bool()
            for layer in layer_ids:
                hidden = captured[layer]
                for row in range(hidden.shape[0]):
                    chunks[layer].append(
                        hidden[row, mask[row]].mean(dim=0).detach().to("cpu", dtype=out_dtype)
                    )
            del encoded, mask
    finally:
        for handle in handles:
            handle.remove()
    return {layer: torch.stack(values, dim=0) for layer, values in chunks.items()}
