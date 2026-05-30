"""Generate harmful PolyRefuse validation completions under v_HRL addition sweeps."""


import json
from pathlib import Path
import sys

import hydra
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import load_file

from multilingual_latent_safety.completions import completion_rows
from multilingual_latent_safety.data import load_polyrefuse
from multilingual_latent_safety.generation import generate_completions, load_generation_model
from multilingual_latent_safety.interventions import install_hooks, make_directional_addition_hook
from multilingual_latent_safety.json_io import write_jsonl_with_meta
from multilingual_latent_safety.model import format_prompt
from multilingual_latent_safety.paths import completions_file, pooled_direction_file

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from results import (
    DEFAULT_LAMBDAS,
    method_slug,
)


def as_float_list(values) -> list[float]:
    if values is None:
        return list(DEFAULT_LAMBDAS)
    return [float(value) for value in values]


def completions_exist(root: Path, languages: list[str], split: str, subset: str) -> bool:
    return all(completions_file(root, language, split, subset).exists() for language in languages)


def write_completions(
    path: Path,
    cfg: DictConfig,
    language: str,
    instructions: list[str],
    completions: list[str],
    *,
    lambda_value: float,
    alpha: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "model": cfg.model.name,
        "generation": OmegaConf.to_container(cfg.generation, resolve=True),
        "language": language,
        "split": str(cfg.evaluation_split),
        "subset": str(cfg.subset),
        "dataset": cfg.dataset.name,
        "intervention": "refusal_activation_sweep",
        "direction": "v_HRL",
        "direction_pool": "hrl",
        "layer": int(cfg.layer),
        "token_position": str(cfg.token_position),
        "token_scope": str(cfg.token_scope),
        "lambda": float(lambda_value),
        "alpha": float(alpha),
    }
    write_jsonl_with_meta(path, meta, completion_rows(instructions, completions))


@hydra.main(version_base=None, config_path="../../configs", config_name="activation_sweep/run_refusal_activation_sweep")
def main(cfg: DictConfig) -> None:
    model, tokenizer = load_generation_model(cfg.model)
    layer = int(cfg.layer)
    split = str(cfg.evaluation_split)
    subset = str(cfg.subset)
    languages = [str(language) for language in cfg.target_languages]
    lambdas = as_float_list(cfg.lambdas)

    direction_path = pooled_direction_file(Path(cfg.pooled_root), "hrl", cfg.token_position, layer)
    direction = load_file(direction_path)["direction"]

    base_root = Path(cfg.output_root) / method_slug(layer)
    base_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "model": cfg.model.name,
        "layer": layer,
        "token_position": str(cfg.token_position),
        "token_scope": str(cfg.token_scope),
        "target_languages": languages,
        "evaluation_split": split,
        "subset": subset,
        "lambdas": lambdas,
        "direction_path": str(direction_path),
    }
    (base_root / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    for lambda_value in lambdas:
        alpha = float(lambda_value)
        lambda_root = base_root / f"lambda={float(lambda_value):g}"
        if completions_exist(lambda_root, languages, split, subset) and not bool(cfg.overwrite):
            print(f"[skip] {lambda_root}")
            continue
        print(f"[run ] lambda={lambda_value:g} alpha={alpha:.4f} langs={len(languages)}")
        hook_factory = lambda hook_device, hook_dtype: make_directional_addition_hook(
            direction,
            alpha,
            hook_device,
            hook_dtype,
            token_scope=str(cfg.token_scope),
        )
        with install_hooks(model, [layer], hook_factory):
            for language in languages:
                out_file = completions_file(lambda_root, language, split, subset)
                if out_file.exists() and not bool(cfg.overwrite):
                    print(f"[skip] {language}/{split}/{subset} -> {out_file}")
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
                prompts = [format_prompt(tokenizer, instr, cfg.model.chat) for instr in instructions]
                print(f"[gen ] {language}/{split}/{subset}: {len(prompts)} prompts")
                completions = generate_completions(model, tokenizer, prompts, cfg.generation)
                write_completions(
                    out_file,
                    cfg,
                    language,
                    instructions,
                    completions,
                    lambda_value=lambda_value,
                    alpha=alpha,
                )
                print(f"[done] {out_file}")


if __name__ == "__main__":
    main()
