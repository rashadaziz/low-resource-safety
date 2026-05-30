"""Extract OMNIGuard mean-pooled prompt representations across layers."""


import itertools
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from safetensors.torch import save_file

from multilingual_latent_safety.activations import (
    extract_mean_pooled,
    extract_mean_pooled_transformers,
    resolve_layers,
)
from multilingual_latent_safety.data import load_polyrefuse
from multilingual_latent_safety.model import (
    format_prompt,
    load_model,
    load_transformers_model,
)
from multilingual_latent_safety.omniguard import representation_dir, representation_file


@hydra.main(version_base=None, config_path="../../configs", config_name="recoverability/extract_omniguard_representations")
def main(cfg: DictConfig) -> None:
    backend = str(cfg.extraction.get("backend", "nnsight"))
    if backend == "nnsight":
        lm = load_model(cfg.model)
        tokenizer = lm.tokenizer
        model = None
        num_layers = lm.config.num_hidden_layers
    elif backend == "transformers":
        model, tokenizer = load_transformers_model(cfg.model)
        lm = None
        num_layers = model.config.num_hidden_layers
    else:
        raise ValueError(f"unknown extraction backend {backend!r}")
    layers = resolve_layers(cfg.extraction.layers, num_layers)
    output_root = Path(cfg.output_root)
    combos = list(itertools.product(cfg.dataset.languages, cfg.dataset.splits, cfg.dataset.subsets))

    for language, split, subset in combos:
        out_dir = representation_dir(output_root, language, split, subset)
        out_dir.mkdir(parents=True, exist_ok=True)
        first_layer_path = representation_file(output_root, language, split, subset, layers[0])
        if first_layer_path.exists() and not bool(cfg.overwrite):
            print(f"[skip] {language}/{split}/{subset} already exists")
            continue

        instructions = load_polyrefuse(
            root=cfg.dataset.root,
            subset=subset,
            split=split,
            language=language,
            max_samples=cfg.dataset.max_samples,
            shuffle=cfg.dataset.shuffle,
            seed=cfg.dataset.seed,
        )
        prompts = [format_prompt(tokenizer, instruction, cfg.model.chat) for instruction in instructions]
        print(f"[run ] {language}/{split}/{subset}: {len(prompts)} prompts, {len(layers)} layers")
        if backend == "nnsight":
            reps = extract_mean_pooled(
                lm=lm,
                prompts=prompts,
                layers=layers,
                hook_point=cfg.extraction.hook_point,
                batch_size=cfg.extraction.batch_size,
                storage_dtype=cfg.extraction.storage_dtype,
            )
        else:
            reps = extract_mean_pooled_transformers(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                layers=layers,
                batch_size=cfg.extraction.batch_size,
                storage_dtype=cfg.extraction.storage_dtype,
            )
        for layer, tensor in reps.items():
            save_file(
                {"representations": tensor.contiguous()},
                representation_file(output_root, language, split, subset, layer),
                metadata={
                    "model": cfg.model.name,
                    "language": language,
                    "split": split,
                    "subset": subset,
                    "layer": str(layer),
                    "hook_point": cfg.extraction.hook_point,
                    "backend": backend,
                    "pooling": "mean_prompt_tokens",
                    "alignment_key": "row_index_within_split_subset",
                },
            )
        del reps
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
