"""Batched text generation helpers."""

from collections.abc import Sequence
from typing import Any

import torch
from omegaconf import DictConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from multilingual_latent_safety.model import load_transformers_model
from multilingual_latent_safety.runtime import batched, set_seed


def load_generation_model(cfg: DictConfig) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load a plain transformers model + tokenizer for decoding.

    Separate from ``model.load_model`` because nnsight's LanguageModel wrapper is not required
    for decoding and plain transformers is cheaper and avoids the nnsight dispatch overhead.
    """
    return load_transformers_model(cfg)


def generation_kwargs(gen_cfg: DictConfig, tokenizer: AutoTokenizer) -> dict[str, Any]:
    kwargs: dict[str, Any] = dict(
        max_new_tokens=int(gen_cfg.max_new_tokens),
        do_sample=bool(gen_cfg.do_sample),
        temperature=float(gen_cfg.temperature),
        top_p=float(gen_cfg.top_p),
        repetition_penalty=float(gen_cfg.repetition_penalty),
        pad_token_id=tokenizer.pad_token_id,
    )
    if not gen_cfg.do_sample:
        kwargs.pop("temperature")
        kwargs.pop("top_p")
    if gen_cfg.stop_sequences:
        kwargs["stop_strings"] = list(gen_cfg.stop_sequences)
        kwargs["tokenizer"] = tokenizer
    return kwargs


@torch.inference_mode()
def generate_completions(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: Sequence[str],
    gen_cfg: DictConfig,
) -> list[str]:
    """Greedy or sampled decode over ``prompts``; returns only the assistant continuation (no prompt prefix)."""
    completions: list[str] = []
    gen_kwargs = generation_kwargs(gen_cfg, tokenizer)
    set_seed(int(gen_cfg.seed))

    for batch in tqdm(list(batched(list(prompts), int(gen_cfg.batch_size))), desc="generate"):
        inputs = tokenizer(list(batch), return_tensors="pt", padding=True, add_special_tokens=False)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        out = model.generate(**inputs, **gen_kwargs)
        prompt_len = inputs["input_ids"].shape[1]
        gen_tokens = out[:, prompt_len:]
        texts = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
        completions.extend(texts)
        del out, gen_tokens, inputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return completions
