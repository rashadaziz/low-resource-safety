import itertools
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from safetensors.torch import save_file

from multilingual_latent_safety.activations import DTYPE_MAP, extract, resolve_layers, resolve_token_positions
from multilingual_latent_safety.data import load_polyrefuse
from multilingual_latent_safety.model import format_prompt, load_model
from multilingual_latent_safety.paths import activations_dir, activations_file


@hydra.main(version_base=None, config_path="../../configs", config_name="hrl_direction/extract_activations")
def main(cfg: DictConfig) -> None:
    """Extract per-layer activations from the configured model over every (language, split, subset) cell.

    Token positions in ``cfg.extraction.token_positions`` can be symbolic names (``t_inst``,
    ``t_post_inst``) or integer indices. Symbolic names are resolved once per model via the chat
    template; the symbolic names are persisted in metadata so downstream analysis can address them
    by name regardless of the underlying integer offset.
    """
    lm = load_model(cfg.model)
    num_layers = lm.config.num_hidden_layers
    layers = resolve_layers(cfg.extraction.layers, num_layers)

    position_specs = list(cfg.extraction.token_positions)
    resolved_positions = resolve_token_positions(lm.tokenizer, cfg.model.chat, position_specs)
    print(f"[info] token positions: {position_specs} → indices {resolved_positions}")

    output_root = Path(cfg.output_root)
    combos = list(itertools.product(cfg.dataset.languages, cfg.dataset.splits, cfg.dataset.subsets))

    for language, split, subset in combos:
        out_dir = activations_dir(output_root, language, split, subset)
        out_dir.mkdir(parents=True, exist_ok=True)

        first_layer_path = activations_file(output_root, language, split, subset, layers[0])
        if first_layer_path.exists() and not cfg.overwrite:
            print(f"[skip] {language}/{split}/{subset} — already exists")
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
        prompts = [format_prompt(lm.tokenizer, instr, cfg.model.chat) for instr in instructions]
        print(f"[run ] {language}/{split}/{subset}: {len(prompts)} prompts, {len(layers)} layers")

        if prompts:
            acts = extract(
                lm=lm,
                prompts=prompts,
                layers=layers,
                token_positions=resolved_positions,
                hook_point=cfg.extraction.hook_point,
                batch_size=cfg.extraction.batch_size,
                storage_dtype=cfg.extraction.storage_dtype,
            )
        else:
            empty_shape = (0, len(resolved_positions), int(lm.config.hidden_size))
            acts = {
                layer: torch.empty(empty_shape, dtype=DTYPE_MAP[cfg.extraction.storage_dtype])
                for layer in layers
            }

        for layer, tensor in acts.items():
            save_file(
                {"activations": tensor.contiguous()},
                activations_file(output_root, language, split, subset, layer),
                metadata={
                    "model": cfg.model.name,
                    "language": language,
                    "split": split,
                    "subset": subset,
                    "layer": str(layer),
                    "hook_point": cfg.extraction.hook_point,
                    "token_positions": ",".join(str(p) for p in position_specs),
                    "token_indices": ",".join(str(i) for i in resolved_positions),
                },
            )
        del acts
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
