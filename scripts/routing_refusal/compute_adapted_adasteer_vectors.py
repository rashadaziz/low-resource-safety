"""Train AdaSteer-format vectors on the PolyRefuse LRL budget.

The released AdaSteer baseline remains zero-shot. This script creates a
separate adapted vector root using only ``budget_per_class`` train prompts per
configured low-resource language.
"""


import json
import pickle
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from multilingual_latent_safety.adasteer import adasteer_spec_for_model
from multilingual_latent_safety.data import load_polyrefuse
from multilingual_latent_safety.generation import batched, load_generation_model
from multilingual_latent_safety.model import format_prompt
from multilingual_latent_safety.probes import sample_balanced_indices


def write_pickle(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f)


def sample_budget_prompts(cfg: DictConfig) -> tuple[list[str], list[str], dict[str, Any]]:
    harmful_all: list[str] = []
    harmless_all: list[str] = []
    sample_meta: dict[str, Any] = {}
    for language in cfg.source_languages:
        harmful = load_polyrefuse(
            root=cfg.dataset.root,
            subset="harmful",
            split=cfg.train_split,
            language=language,
        )
        harmless = load_polyrefuse(
            root=cfg.dataset.root,
            subset="harmless",
            split=cfg.train_split,
            language=language,
        )
        harmful_idx, harmless_idx = sample_balanced_indices(
            len(harmful),
            len(harmless),
            int(cfg.budget_per_class),
            int(cfg.seed) + sum(ord(ch) for ch in str(language)),
        )
        harmful_items = [harmful[int(i)] for i in harmful_idx.tolist()]
        harmless_items = [harmless[int(i)] for i in harmless_idx.tolist()]
        harmful_all.extend(harmful_items)
        harmless_all.extend(harmless_items)
        sample_meta[str(language)] = {
            "harmful_indices": [int(i) for i in harmful_idx.tolist()],
            "harmless_indices": [int(i) for i in harmless_idx.tolist()],
            "n_harmful": len(harmful_items),
            "n_harmless": len(harmless_items),
        }
    return harmful_all, harmless_all, sample_meta


@torch.inference_mode()
def extract_last_prompt_hiddens(
    model: torch.nn.Module,
    tokenizer,
    prompts: list[str],
    *,
    batch_size: int,
    max_length: int,
) -> torch.Tensor:
    n_layers = int(model.config.num_hidden_layers)
    chunks: list[torch.Tensor] = []
    device = getattr(model, "device", next(model.parameters()).device)
    for batch in tqdm(list(batched(prompts, batch_size)), desc="extract"):
        encoded = tokenizer(
            list(batch),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
        )
        attention_mask = encoded["attention_mask"].to(device)
        inputs = {key: value.to(device) for key, value in encoded.items()}
        output = model(**inputs, output_hidden_states=True, use_cache=False)
        batch_layers = []
        reversed_mask = torch.flip(attention_mask, dims=[1])
        last_indices = attention_mask.shape[1] - 1 - reversed_mask.argmax(dim=1)
        batch_index = torch.arange(attention_mask.shape[0], device=device)
        for layer in range(n_layers):
            hidden = output.hidden_states[layer + 1]
            batch_layers.append(hidden[batch_index, last_indices, :].detach().cpu())
        chunks.append(torch.stack(batch_layers, dim=0))
        del output, inputs, encoded, attention_mask
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return torch.cat(chunks, dim=1).to(dtype=torch.float32)


def mean_diff(class_a: torch.Tensor, class_b: torch.Tensor) -> np.ndarray:
    return (class_a.mean(dim=1) - class_b.mean(dim=1)).cpu().numpy().astype(np.float32)


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="routing_refusal/compute_adapted_adasteer_vectors",
)
def main(cfg: DictConfig) -> None:
    spec = adasteer_spec_for_model(str(cfg.model.name))
    out_root = Path(cfg.output_root) / spec.key
    meta_path = out_root / "metadata.json"
    if meta_path.exists() and not bool(cfg.overwrite):
        print(f"[skip] {meta_path}")
        return

    model, tokenizer = load_generation_model(cfg.model)
    harmful, harmless, sample_meta = sample_budget_prompts(cfg)
    harmful_prompts = [format_prompt(tokenizer, text, cfg.model.chat) for text in harmful]
    harmless_prompts = [format_prompt(tokenizer, text, cfg.model.chat) for text in harmless]
    class_a = extract_last_prompt_hiddens(
        model,
        tokenizer,
        harmful_prompts,
        batch_size=int(cfg.batch_size),
        max_length=int(cfg.max_length),
    )
    class_b = extract_last_prompt_hiddens(
        model,
        tokenizer,
        harmless_prompts,
        batch_size=int(cfg.batch_size),
        max_length=int(cfg.max_length),
    )

    class_a_np = class_a.cpu().numpy().astype(np.float32)
    class_b_np = class_b.cpu().numpy().astype(np.float32)
    direction = mean_diff(class_a, class_b)

    # AdaSteer expects RD plus an HD projection. PolyRefuse supplies one
    # harmful/harmless contrast, so adapted HD uses the same budgeted contrast.
    for subdir in ("RD", "HD"):
        write_pickle(class_a_np, out_root / subdir / "class_a.pkl")
        write_pickle(class_b_np, out_root / subdir / "class_b.pkl")
        write_pickle(direction, out_root / subdir / "mean_diff.pkl")
    write_pickle(direction, out_root / "HD" / "proj.pkl")

    metadata = {
        "method": "adasteer_adapted",
        "model": str(cfg.model.name),
        "model_key": spec.key,
        "source_dataset": "polyrefuse",
        "source_languages": list(cfg.source_languages),
        "train_split": str(cfg.train_split),
        "budget_per_class_per_language": int(cfg.budget_per_class),
        "seed": int(cfg.seed),
        "sample_meta": sample_meta,
        "config": OmegaConf.to_container(cfg, resolve=True),
        "note": (
            "AdaSteer released code trains RD/HD from external anchors. "
            "This adapted root replaces those anchors with the same PolyRefuse "
            "harmful/harmless budget used by competing adapted baselines."
        ),
    }
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"[done] {meta_path}")


if __name__ == "__main__":
    main()
