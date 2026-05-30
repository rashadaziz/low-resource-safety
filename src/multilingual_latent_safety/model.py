
"""Model construction, device, token-position, and chat prompt helpers."""

import torch
from nnsight import LanguageModel
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import Any

DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def build_quantization_config(quant_cfg: DictConfig) -> BitsAndBytesConfig:
    """Translate a Hydra quantization group into a transformers ``BitsAndBytesConfig``."""
    return BitsAndBytesConfig(
        load_in_4bit=bool(quant_cfg.get("load_in_4bit", False)),
        load_in_8bit=bool(quant_cfg.get("load_in_8bit", False)),
        bnb_4bit_quant_type=quant_cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=DTYPE_MAP[quant_cfg.get("bnb_4bit_compute_dtype", "bfloat16")],
        bnb_4bit_use_double_quant=bool(quant_cfg.get("bnb_4bit_use_double_quant", True)),
    )


def torch_dtype(dtype_name: str) -> Any:
    """Resolve a config dtype name accepted by transformers model loaders."""
    if dtype_name == "auto":
        return "auto"
    if dtype_name not in DTYPE_MAP:
        raise ValueError(f"unsupported dtype: {dtype_name}")
    return DTYPE_MAP[dtype_name]


def model_load_kwargs(
    *,
    device_map: Any,
    trust_remote_code: bool,
    revision: str | None,
    dtype: str,
    quantization: DictConfig | None = None,
    dispatch: bool = False,
) -> dict[str, Any]:
    """Common model-loading kwargs for nnsight and plain HuggingFace loaders."""
    kwargs: dict[str, Any] = {
        "device_map": device_map,
        "trust_remote_code": trust_remote_code,
    }
    if dispatch:
        kwargs["dispatch"] = True
    if revision is not None:
        kwargs["revision"] = revision
    if quantization is None:
        kwargs["torch_dtype"] = torch_dtype(dtype)
    else:
        kwargs["quantization_config"] = build_quantization_config(quantization)
    return kwargs


def configure_tokenizer(tokenizer: Any, *, padding_side: str) -> Any:
    """Apply repo-standard tokenizer padding settings."""
    tokenizer.padding_side = padding_side
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def load_causal_lm_and_tokenizer(
    *,
    model_name: str,
    device_map: Any,
    trust_remote_code: bool,
    revision: str | None,
    dtype: str,
    tokenizer_use_fast: bool,
    tokenizer_padding_side: str,
    quantization: DictConfig | None = None,
) -> tuple[Any, Any]:
    """Load a plain HuggingFace causal LM and tokenizer from normalized config fields."""
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_load_kwargs(
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            revision=revision,
            dtype=dtype,
            quantization=quantization,
        ),
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        revision=revision,
        trust_remote_code=trust_remote_code,
        use_fast=tokenizer_use_fast,
    )
    configure_tokenizer(tokenizer, padding_side=tokenizer_padding_side)
    model.eval()
    return model, tokenizer


def load_model(cfg: DictConfig) -> Any:
    """Load an nnsight LanguageModel from a Hydra model config, ensuring a pad token is set.

    Honours an optional ``cfg.quantization`` group (``load_in_4bit``/``load_in_8bit`` + bnb knobs).
    When quantization is active we drop ``torch_dtype`` and let ``BitsAndBytesConfig`` govern dtype.
    """
    kwargs = model_load_kwargs(
        device_map=cfg.device_map,
        trust_remote_code=cfg.trust_remote_code,
        revision=cfg.revision,
        dtype=cfg.dtype,
        quantization=cfg.get("quantization"),
        dispatch=True,
    )
    lm = LanguageModel(cfg.name, **kwargs)
    configure_tokenizer(lm.tokenizer, padding_side=cfg.tokenizer.padding_side)
    return lm


def load_transformers_model(cfg: DictConfig) -> tuple[Any, Any]:
    """Load a plain ``transformers`` causal LM and tokenizer from the shared model config."""
    return load_causal_lm_and_tokenizer(
        model_name=cfg.name,
        device_map=cfg.device_map,
        trust_remote_code=cfg.trust_remote_code,
        revision=cfg.revision,
        dtype=cfg.dtype,
        tokenizer_use_fast=bool(cfg.tokenizer.use_fast),
        tokenizer_padding_side=cfg.tokenizer.padding_side,
        quantization=cfg.get("quantization"),
    )


def model_input_device(model: Any) -> torch.device:
    """Return the first non-meta parameter device, falling back to the available accelerator."""
    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def transformer_layers(model: Any) -> Any:
    """Return the decoder block stack for common HuggingFace causal LM layouts."""
    candidate_paths = (
        ("model", "layers"),
        ("language_model", "model", "layers"),
        ("transformer", "h"),
    )
    for path in candidate_paths:
        current = model
        for attr in path:
            current = getattr(current, attr, None)
            if current is None:
                break
        if current is not None:
            return current
    raise AttributeError("could not find transformer layer stack on model")


def last_nonpad_positions(attention_mask: torch.Tensor) -> torch.Tensor:
    """Index of the last non-padding token for each row of an attention mask."""
    flipped = torch.flip(attention_mask.to(torch.long), dims=[1])
    distance_from_end = torch.argmax(flipped, dim=1)
    return attention_mask.shape[1] - 1 - distance_from_end


def format_prompt(tokenizer, instruction: str, chat_cfg: DictConfig) -> str:
    """Wrap an instruction in the tokenizer's chat template, optionally prepending a system prompt."""
    messages = []
    if chat_cfg.get("system_prompt") is not None:
        messages.append({"role": "system", "content": chat_cfg.system_prompt})
    messages.append({"role": "user", "content": instruction})
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=chat_cfg.add_generation_prompt,
    )
