"""Generate completions with configured steering interventions.

The retained Hydra modes are:
- ``directional_ablation`` for the HRL-direction ablation sweep.
- ``conditional_vhrl`` for sample-efficient conditional HRL-subspace steering.
- ``adasteer`` for the AdaSteer comparison baseline.
"""

import itertools
import json
from pathlib import Path
import sys

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multilingual_latent_safety.adasteer import install_adasteer_hooks, load_adasteer_bundle
from multilingual_latent_safety.conditional_vhrl import run_conditional_vhrl
from multilingual_latent_safety.data import load_polyrefuse
from multilingual_latent_safety.generation import generate_completions, load_generation_model
from multilingual_latent_safety.interventions import (
    install_hooks,
    make_directional_ablation_hook,
)
from multilingual_latent_safety.model import format_prompt
from multilingual_latent_safety.paths import (
    direction_file,
    intervention_completions_file,
    pooled_direction_file,
)


def load_direction_vector(
    cfg: DictConfig,
    target_lang: str,
    probe_layer: int,
) -> torch.Tensor:
    src = str(cfg.intervention.direction_source_lang)
    token_pos = str(cfg.intervention.token_position)
    estimator = str(cfg.intervention.direction_estimator)
    split = str(cfg.intervention.direction_split)
    if src in {"hrl_pooled", "hrl_pooled_dim"}:
        path = pooled_direction_file(Path(cfg.pooled_root), "hrl", token_pos, probe_layer)
        return load_file(path)["direction"]
    actual_source = target_lang if src == "per_language" else src
    if estimator in ("diff_of_means", "pca_top1"):
        path = direction_file(Path(cfg.directions_root), actual_source, split, token_pos, probe_layer)
        return load_file(path)["direction"]
    raise ValueError(f"unknown direction_estimator: {estimator}")


def build_hook_factory(
    cfg: DictConfig,
    direction: torch.Tensor,
):
    mode = str(cfg.intervention.mode)
    token_scope = str(cfg.intervention.get("token_scope", "all_tokens"))
    if mode == "directional_ablation":
        return lambda device, dtype: make_directional_ablation_hook(
            direction, device, dtype, token_scope=token_scope
        )
    raise ValueError(f"unknown intervention mode: {mode}")


@hydra.main(version_base=None, config_path="../../configs", config_name="routing_refusal/intervene_and_generate")
def main(cfg: DictConfig) -> None:
    mode = str(cfg.intervention.mode)
    if mode == "conditional_vhrl":
        run_conditional_vhrl(cfg)
        return

    model, tokenizer = load_generation_model(cfg.model)
    n_layers = getattr(model.config, "num_hidden_layers", None)
    if n_layers is None:
        raise RuntimeError("model.config.num_hidden_layers missing; add a dispatch case")

    run_specs: list[tuple[int, list[int], int | None]] = []
    if mode == "adasteer":
        layer_start = int(cfg.intervention.layer_range.get("start", 0))
        layer_end_cfg = cfg.intervention.layer_range.get("end", None)
        layer_end = n_layers if layer_end_cfg is None else int(layer_end_cfg)
        layer_step = int(cfg.intervention.layer_range.get("step", 1))
        layer_range = list(range(layer_start, min(layer_end, n_layers), layer_step))
        if layer_range != list(range(n_layers)):
            raise ValueError(
                "AdaSteer upstream applies steering across all decoder layers; "
                f"got layer_range={layer_range}, expected 0..{n_layers - 1}"
            )
        run_specs.append((0, layer_range, None))
    else:
        # Probe layer = where the direction was estimated. Defaults to the last layer in the
        # ablation range; override via `intervention.direction_layer` to decouple the direction
        # layer from the ablation layer range (used by the anchor-layer sweep).
        layer_start = int(cfg.intervention.layer_range.start)
        layer_end = int(cfg.intervention.layer_range.end)
        layer_step = int(cfg.intervention.layer_range.step)
        override = cfg.intervention.get("direction_layer", None)
        sweep_layers = bool(cfg.intervention.get("sweep_layers", False))
        if sweep_layers:
            if override is not None:
                raise ValueError("intervention.sweep_layers=true cannot be combined with direction_layer")
            for layer in range(layer_start, min(layer_end, n_layers), layer_step):
                run_specs.append((layer, [layer], layer))
        else:
            if override is None:
                probe_layer = min(layer_end - 1, n_layers - 1)
                layer_tag = None
            else:
                probe_layer = int(override)
                if not 0 <= probe_layer < n_layers:
                    raise ValueError(
                        f"intervention.direction_layer={probe_layer} out of range [0, {n_layers})"
                    )
                layer_tag = probe_layer
            layer_range = list(
                range(
                    layer_start,
                    min(layer_end, n_layers),
                    layer_step,
                )
            )
            run_specs.append((probe_layer, layer_range, layer_tag))

    output_root = Path(cfg.output_root)
    combos = list(itertools.product(cfg.target_languages, cfg.splits, cfg.subsets))
    adasteer_bundle = None
    if mode == "adasteer":
        adasteer_bundle = load_adasteer_bundle(
            Path(cfg.intervention.vector_root),
            str(cfg.model.name),
        )
        print(
            "[info] AdaSteer vectors "
            f"model_key={adasteer_bundle.spec.key} "
            f"alpha_layer={adasteer_bundle.spec.alpha_layer} "
            f"beta_layer={adasteer_bundle.spec.beta_layer}"
        )

    for probe_layer, layer_range, layer_tag in run_specs:
        for target_lang, split, subset in combos:
            src_tag = str(cfg.intervention.get("direction_source_lang", "released"))
            out_path = intervention_completions_file(
                output_root,
                cfg.intervention.name,
                src_tag,
                target_lang,
                split,
                subset,
                direction_layer=layer_tag,
            )
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if out_path.exists() and not cfg.overwrite:
                print(f"[skip] {target_lang}/{split}/{subset} -> {out_path}")
                continue

            direction = None if mode == "adasteer" else load_direction_vector(
                cfg, target_lang, probe_layer
            )

            instructions = load_polyrefuse(
                root=cfg.dataset.root,
                subset=subset,
                split=split,
                language=target_lang,
                max_samples=cfg.dataset.max_samples,
                shuffle=cfg.dataset.shuffle,
                seed=cfg.dataset.seed,
            )
            prompts = [format_prompt(tokenizer, instr, cfg.model.chat) for instr in instructions]

            if mode == "adasteer":
                if adasteer_bundle is None:
                    raise RuntimeError("AdaSteer bundle was not loaded")
                with install_adasteer_hooks(model, adasteer_bundle, layer_range):
                    completions = generate_completions(model, tokenizer, prompts, cfg.generation)
            else:
                if direction is None:
                    raise RuntimeError("direction vector was not loaded")
                factory = build_hook_factory(
                    cfg,
                    direction,
                )
                with install_hooks(model, layer_range, factory):
                    completions = generate_completions(model, tokenizer, prompts, cfg.generation)

            meta = {
                "model": cfg.model.name,
                "generation": OmegaConf.to_container(cfg.generation, resolve=True),
                "intervention": OmegaConf.to_container(cfg.intervention, resolve=True),
                "layer_range": layer_range,
                "language": target_lang,
                "split": split,
                "subset": subset,
            }
            if mode == "adasteer":
                if adasteer_bundle is None:
                    raise RuntimeError("AdaSteer bundle was not loaded")
                meta["adasteer"] = {
                    "model_key": adasteer_bundle.spec.key,
                    "vector_root": str(cfg.intervention.vector_root),
                    "alpha_layer": adasteer_bundle.spec.alpha_layer,
                    "beta_layer": adasteer_bundle.spec.beta_layer,
                    "steering_scope": "generated_tokens_all_layers",
                }
            with open(out_path, "w") as f:
                f.write(json.dumps({"__meta__": meta}) + "\n")
                for idx, (instr, comp) in enumerate(zip(instructions, completions, strict=True)):
                    f.write(
                        json.dumps(
                            {"prompt_id": idx, "instruction": instr, "completion": comp},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            print(f"[done] {out_path}")


if __name__ == "__main__":
    main()
