"""Score PolyRefuse completions with local guard-model refusal judges."""


import itertools
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from multilingual_latent_safety.completions import completion_instruction
from multilingual_latent_safety.data import load_polyrefuse
from multilingual_latent_safety.json_io import read_jsonl_with_meta, write_jsonl_with_meta
from multilingual_latent_safety.judges.guard_refusal import (
    load_guard_model,
    score_guard_items,
    supported_languages,
)


def completion_path(
    completions_root: Path,
    model: str,
    gen_tag: str,
    language: str,
    split: str,
    subset: str,
) -> Path:
    return completions_root / model / gen_tag / language / split / f"{subset}.jsonl"


def output_path(
    output_root: Path,
    model: str,
    gen_tag: str,
    language: str,
    split: str,
    subset: str,
) -> Path:
    return output_root / model / gen_tag / language / split / f"{subset}.jsonl"


def selected_languages(cfg: DictConfig) -> list[str]:
    supported = set(str(lang) for lang in cfg.guard_judge.supported_languages)
    expected = set(supported_languages(str(cfg.guard_judge.parser)))
    if supported != expected:
        raise ValueError(
            "guard_judge.supported_languages does not match the parser's PolyRefuse overlap"
        )

    if cfg.languages is None:
        requested = [str(lang) for lang in cfg.dataset.languages]
        return [lang for lang in requested if lang in supported]

    requested = [str(lang) for lang in cfg.languages]
    unsupported = [lang for lang in requested if lang not in supported]
    if unsupported:
        raise ValueError(
            f"{cfg.guard_judge.name} does not support requested PolyRefuse languages: "
            f"{unsupported}"
        )
    return requested


def planned_cells(cfg: DictConfig) -> list[tuple[str, str, str, str, Path, Path]]:
    completions_root = Path(cfg.completions_root)
    output_root = Path(cfg.output_root)
    cells = []
    for model, language, split, subset in itertools.product(
        cfg.generator_models,
        selected_languages(cfg),
        cfg.splits,
        cfg.subsets,
    ):
        comp_path = completion_path(
            completions_root, str(model), str(cfg.gen_tag), language, str(split), str(subset)
        )
        out_path = output_path(
            output_root, str(model), str(cfg.gen_tag), language, str(split), str(subset)
        )
        cells.append((str(model), language, str(split), str(subset), comp_path, out_path))
    return cells


@hydra.main(version_base=None, config_path="../../configs", config_name="refusal_gap/score_guard_refusal")
def main(cfg: DictConfig) -> None:
    """Run local guard scoring over existing completion artifacts."""

    cells = planned_cells(cfg)
    missing = [str(comp_path) for *_, comp_path, _ in cells if not comp_path.exists()]
    if missing and bool(cfg.fail_on_missing):
        preview = "\n  ".join(missing[:25])
        extra = "" if len(missing) <= 25 else f"\n  ... {len(missing) - 25} more"
        raise SystemExit(f"missing completion files:\n  {preview}{extra}")

    if bool(cfg.dry_run):
        print(f"[dry-run] judge={cfg.guard_judge.name} cells={len(cells)}")
        for model, language, split, subset, comp_path, out_path in cells:
            status = "ok" if comp_path.exists() else "missing"
            print(f"{status:7s} {model} {language}/{split}/{subset} -> {out_path}")
        return

    todo = []
    for cell in cells:
        *_, comp_path, out_path = cell
        if not comp_path.exists():
            print(f"[skip] missing completions: {comp_path}")
            continue
        if out_path.exists() and not bool(cfg.overwrite):
            print(f"[skip] exists: {out_path}")
            continue
        todo.append(cell)

    if not todo:
        print("[done] no cells to score")
        return

    model, tokenizer = load_guard_model(cfg.guard_judge)
    for generator_model, language, split, subset, comp_path, out_path in todo:
        meta, items = read_jsonl_with_meta(comp_path)
        fallback_instructions = load_polyrefuse(
            root=cfg.dataset.root,
            subset=subset,
            split=split,
            language=language,
        )
        payload = [
            {
                "user_prompt": completion_instruction(item, fallback_instructions),
                "response": item["completion"],
                "language": language,
            }
            for item in items
        ]
        print(
            f"[run ] {cfg.guard_judge.name} {generator_model} "
            f"{language}/{split}/{subset}: {len(payload)} items"
        )
        scores = score_guard_items(model, tokenizer, cfg.guard_judge, payload)

        out_meta = {
            "judge": OmegaConf.to_container(cfg.guard_judge, resolve=True),
            "completions_meta": meta,
            "generator_model": generator_model,
            "language": language,
            "split": split,
            "subset": subset,
        }
        write_jsonl_with_meta(
            out_path,
            out_meta,
            [
                {"prompt_id": item["prompt_id"], "refusal": score["refusal"]}
                for item, score in zip(items, scores, strict=True)
            ],
        )
        print(f"[done] {out_path}")


if __name__ == "__main__":
    main()
