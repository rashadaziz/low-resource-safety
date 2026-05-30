import itertools
import json
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from multilingual_latent_safety.data import load_polyrefuse
from multilingual_latent_safety.generation import generate_completions, load_generation_model
from multilingual_latent_safety.model import format_prompt
from multilingual_latent_safety.paths import completions_dir, completions_file


@hydra.main(version_base=None, config_path="../../configs", config_name="refusal_gap/generate_completions")
def main(cfg: DictConfig) -> None:
    """Greedy-decode PolyRefuse prompts across every (language, split, subset) cell and write one JSONL per cell."""
    model, tokenizer = load_generation_model(cfg.model)
    output_root = Path(cfg.output_root)
    combos = list(itertools.product(cfg.dataset.languages, cfg.splits, cfg.subsets))

    for language, split, subset in combos:
        out_file = completions_file(output_root, language, split, subset)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        if out_file.exists() and not cfg.overwrite:
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
        print(f"[run ] {language}/{split}/{subset}: {len(prompts)} prompts")
        completions = generate_completions(model, tokenizer, prompts, cfg.generation)

        meta = {
            "model": cfg.model.name,
            "generation": OmegaConf.to_container(cfg.generation, resolve=True),
            "language": language,
            "split": split,
            "subset": subset,
            "dataset": cfg.dataset.name,
        }
        with open(out_file, "w") as f:
            f.write(json.dumps({"__meta__": meta}) + "\n")
            for idx, (instr, comp) in enumerate(zip(instructions, completions, strict=True)):
                f.write(
                    json.dumps(
                        {
                            "prompt_id": idx,
                            "instruction": instr,
                            "completion": comp,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        print(f"[done] {out_file}")


if __name__ == "__main__":
    main()
