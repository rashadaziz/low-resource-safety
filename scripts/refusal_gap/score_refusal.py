import asyncio
import itertools
import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from multilingual_latent_safety.completions import completion_instruction
from multilingual_latent_safety.data import load_polyrefuse
from multilingual_latent_safety.json_io import read_jsonl_with_meta, write_jsonl_with_meta
from multilingual_latent_safety.judges.refusal import score_many
from multilingual_latent_safety.paths import completions_file, refusal_score_file


@hydra.main(version_base=None, config_path="../../configs", config_name="refusal_gap/score_refusal")
def main(cfg: DictConfig) -> None:
    """Score PolyRefuse completions with the refusal judge via OpenRouter."""
    logging.getLogger("httpx").setLevel(logging.WARNING)
    if cfg.judge.backend != "openrouter":
        raise ValueError(
            f"score_refusal expects judge.backend=openrouter, got {cfg.judge.backend}"
        )
    completions_root = Path(cfg.completions_root)
    output_root = Path(cfg.output_root)
    combos = list(itertools.product(cfg.dataset.languages, cfg.splits, cfg.subsets))

    for language, split, subset in combos:
        comp_path = completions_file(completions_root, language, split, subset)
        if not comp_path.exists():
            print(f"[skip] missing completions: {comp_path}")
            continue
        out_path = refusal_score_file(output_root, language, split, subset)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists() and not cfg.overwrite:
            print(f"[skip] {language}/{split}/{subset} -> {out_path}")
            continue

        meta, items = read_jsonl_with_meta(comp_path)
        fallback_instructions = load_polyrefuse(
            root=cfg.dataset.root,
            subset=subset,
            split=split,
            language=language,
        )
        payload = [
            {
                "user_prompt": completion_instruction(it, fallback_instructions),
                "response": it["completion"],
                "language": language,
            }
            for it in items
        ]
        print(f"[run ] {language}/{split}/{subset}: {len(payload)} items")
        scores = asyncio.run(score_many(cfg.judge, payload))

        out_meta = {
            "judge": OmegaConf.to_container(cfg.judge, resolve=True),
            "completions_meta": meta,
            "language": language,
            "split": split,
            "subset": subset,
        }
        write_jsonl_with_meta(
            out_path,
            out_meta,
            [
                {"prompt_id": it["prompt_id"], "refusal": sc["refusal"]}
                for it, sc in zip(items, scores, strict=True)
            ],
        )
        print(f"[done] {out_path}")


if __name__ == "__main__":
    main()
