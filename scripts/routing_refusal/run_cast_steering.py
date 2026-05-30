"""Train/tune CAST and generate PolyRefuse completions.

Two variants are supported:

- ``cast_zero_shot`` trains behavior and condition vectors on CAST's released
  demo data and tunes the condition point on CAST's own held-out split.
- ``cast_adapted`` trains vectors on the PolyRefuse LRL train budget
  (32 harmful + 32 harmless per LRL by default) and tunes the condition point on
  PolyRefuse validation data.
"""


import itertools
import json
from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from multilingual_latent_safety.cast import (
    CastBundle,
    CastConditionPoint,
    cast_condition_similarity,
    cast_pca_direction,
    install_cast_hooks,
    select_condition_point,
)
from multilingual_latent_safety.completions import completion_rows
from multilingual_latent_safety.data import load_polyrefuse
from multilingual_latent_safety.generation import batched, generate_completions, load_generation_model
from multilingual_latent_safety.json_io import read_json, write_jsonl_with_meta
from multilingual_latent_safety.model import format_prompt, last_nonpad_positions
from multilingual_latent_safety.paths import completions_file
from multilingual_latent_safety.probes import sample_balanced_indices
from multilingual_latent_safety.runtime import stable_seed
from multilingual_latent_safety.vector_ops import unit_normalize


def layer_list(layer_cfg: Any, n_layers: int) -> list[int]:
    if layer_cfg == "all":
        return list(range(n_layers))
    if hasattr(layer_cfg, "keys"):
        start = int(layer_cfg.get("start", 0))
        end_cfg = layer_cfg.get("end", n_layers)
        end = n_layers if end_cfg is None else int(end_cfg)
        step = int(layer_cfg.get("step", 1))
        return list(range(start, min(end, n_layers), step))
    return [int(layer) % n_layers for layer in list(layer_cfg)]


def format_train_prompt(
    tokenizer: Any,
    instruction: str,
    chat_cfg: DictConfig,
    *,
    add_generation_prompt: bool,
) -> str:
    messages = []
    if chat_cfg.get("system_prompt") is not None:
        messages.append({"role": "system", "content": chat_cfg.system_prompt})
    messages.append({"role": "user", "content": instruction})
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )


def official_condition_pairs(
    root: Path,
    split: str,
    max_pairs: int | None,
) -> tuple[list[str], list[str], dict[str, Any]]:
    rows = read_json(root / "condition_harmful.json")[split]
    data = list(rows if max_pairs is None else rows[: int(max_pairs)])
    positive = [str(row["harmful"]) for row in data]
    negative = [str(row["harmless"]) for row in data]
    return positive, negative, {"source": "cast_official_condition_harmful", "split": split, "pairs": len(data)}


def official_behavior_pairs(
    root: Path,
    tokenizer: Any,
    chat_cfg: DictConfig,
    *,
    questions: int,
    suffixes: int,
    add_generation_prompt: bool,
) -> tuple[list[str], list[str], list[int], list[int], dict[str, Any]]:
    alpaca = read_json(root / "alpaca.json")["train"]
    refusal = read_json(root / "behavior_refusal.json")
    question_rows = list(alpaca[: int(questions)])
    non_compliant = list(refusal["non_compliant_responses"][: int(suffixes)])
    compliant = list(refusal["compliant_responses"][: int(suffixes)])
    suffix_pairs = list(zip(non_compliant, compliant, strict=True))

    positive: list[str] = []
    negative: list[str] = []
    positive_suffix_lengths: list[int] = []
    negative_suffix_lengths: list[int] = []
    for pos_suffix, neg_suffix in suffix_pairs:
        pos_len = len(tokenizer.encode(pos_suffix, add_special_tokens=False))
        neg_len = len(tokenizer.encode(neg_suffix, add_special_tokens=False))
        for row in question_rows:
            prompt = format_train_prompt(
                tokenizer,
                str(row["question"]),
                chat_cfg,
                add_generation_prompt=add_generation_prompt,
            )
            positive.append(prompt + pos_suffix)
            negative.append(prompt + neg_suffix)
            positive_suffix_lengths.append(pos_len)
            negative_suffix_lengths.append(neg_len)

    meta = {
        "source": "cast_official_behavior_refusal",
        "alpaca_questions": len(question_rows),
        "suffix_pairs": len(suffix_pairs),
        "pairs": len(positive),
    }
    return positive, negative, positive_suffix_lengths, negative_suffix_lengths, meta


def polyrefuse_budget_pairs(
    cfg: DictConfig,
    tokenizer: Any,
    *,
    split: str,
    languages: list[str],
    add_generation_prompt: bool,
) -> tuple[list[str], list[str], dict[str, Any]]:
    budget = int(cfg.cast.budget_per_class)
    positive: list[str] = []
    negative: list[str] = []
    counts: dict[str, dict[str, int]] = {}
    for language in languages:
        harmful = load_polyrefuse(cfg.dataset.root, "harmful", split, language)
        harmless = load_polyrefuse(cfg.dataset.root, "harmless", split, language)
        n = min(budget, len(harmful), len(harmless))
        seed = stable_seed(int(cfg.cast.seed), cfg.model.name, language, split, budget, "cast")
        harmful_idx, harmless_idx = sample_balanced_indices(len(harmful), len(harmless), n, seed)
        for idx in harmful_idx.tolist():
            positive.append(
                format_train_prompt(
                    tokenizer,
                    harmful[int(idx)],
                    cfg.model.chat,
                    add_generation_prompt=add_generation_prompt,
                )
            )
        for idx in harmless_idx.tolist():
            negative.append(
                format_train_prompt(
                    tokenizer,
                    harmless[int(idx)],
                    cfg.model.chat,
                    add_generation_prompt=add_generation_prompt,
                )
            )
        counts[language] = {"harmful": n, "harmless": n}
    meta = {
        "source": "polyrefuse_budget",
        "split": split,
        "budget_per_class": budget,
        "languages": languages,
        "counts": counts,
        "pairs": len(positive),
    }
    return positive, negative, meta


@torch.inference_mode()
def extract_pooled_hiddens(
    model: torch.nn.Module,
    tokenizer: Any,
    prompts: list[str],
    layers: list[int],
    *,
    pool: str,
    batch_size: int,
    max_length: int | None,
    suffix_lengths: list[int] | None = None,
) -> dict[int, torch.Tensor]:
    if suffix_lengths is not None and len(suffix_lengths) != len(prompts):
        raise ValueError("suffix_lengths must match prompts")
    device = getattr(model, "device", next(model.parameters()).device)
    per_layer: dict[int, list[torch.Tensor]] = {int(layer): [] for layer in layers}
    prompt_batches = list(batched(prompts, batch_size))
    offset = 0
    for batch in tqdm(prompt_batches, desc=f"extract[{pool}]"):
        batch = list(batch)
        batch_suffix_lengths = None
        if suffix_lengths is not None:
            batch_suffix_lengths = suffix_lengths[offset : offset + len(batch)]
        offset += len(batch)
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=max_length is not None,
            max_length=max_length,
            add_special_tokens=False,
        )
        attention_mask = encoded["attention_mask"].to(device)
        inputs = {key: value.to(device) for key, value in encoded.items()}
        output = model(**inputs, output_hidden_states=True, use_cache=False)
        for layer in layers:
            hidden = output.hidden_states[int(layer) + 1].to(dtype=torch.float32)
            layer_mask = attention_mask.to(device=hidden.device, dtype=torch.float32)
            layer_last_idx = last_nonpad_positions(attention_mask.to(device=hidden.device))
            layer_batch_idx = torch.arange(layer_mask.shape[0], device=hidden.device)
            token_counts = layer_mask.sum(dim=1).clamp(min=1.0)
            if pool == "mean":
                pooled = (hidden * layer_mask.unsqueeze(-1)).sum(dim=1) / token_counts.unsqueeze(-1)
            elif pool == "last":
                pooled = hidden[layer_batch_idx, layer_last_idx, :]
            elif pool == "suffix":
                if batch_suffix_lengths is None:
                    raise ValueError("pool='suffix' requires suffix_lengths")
                pieces = []
                for row, suffix_len in enumerate(batch_suffix_lengths):
                    last = int(layer_last_idx[row].item())
                    width = max(1, min(int(suffix_len), last + 1))
                    start = last - width + 1
                    pieces.append(hidden[row, start : last + 1, :].mean(dim=0))
                pooled = torch.stack(pieces, dim=0)
            else:
                raise ValueError(f"unknown pool {pool!r}")
            per_layer[int(layer)].append(pooled.detach().cpu())

        del output, inputs, encoded
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {layer: torch.cat(chunks, dim=0) for layer, chunks in per_layer.items()}


def train_direction_set(
    model: torch.nn.Module,
    tokenizer: Any,
    positive: list[str],
    negative: list[str],
    layers: list[int],
    *,
    pool: str,
    method: str,
    batch_size: int,
    max_length: int | None,
    positive_suffix_lengths: list[int] | None = None,
    negative_suffix_lengths: list[int] | None = None,
) -> dict[int, torch.Tensor]:
    pos_hiddens = extract_pooled_hiddens(
        model,
        tokenizer,
        positive,
        layers,
        pool=pool,
        batch_size=batch_size,
        max_length=max_length,
        suffix_lengths=positive_suffix_lengths,
    )
    neg_hiddens = extract_pooled_hiddens(
        model,
        tokenizer,
        negative,
        layers,
        pool=pool,
        batch_size=batch_size,
        max_length=max_length,
        suffix_lengths=negative_suffix_lengths,
    )
    return {
        int(layer): unit_normalize(
            cast_pca_direction(pos_hiddens[int(layer)], neg_hiddens[int(layer)], method=method)
        ).cpu()
        for layer in layers
    }


def train_or_load_cast_artifact(
    cfg: DictConfig,
    model: torch.nn.Module,
    tokenizer: Any,
    condition_layers: list[int],
    behavior_layers: list[int],
) -> dict[str, Any]:
    artifact_root = Path(cfg.cast.artifact_root)
    artifact_path = artifact_root / "cast_vectors.pt"
    if artifact_path.exists() and not bool(cfg.cast.retrain):
        return torch.load(artifact_path, map_location="cpu", weights_only=False)

    batch_size = int(cfg.cast.extraction_batch_size)
    max_length_cfg = cfg.cast.get("max_length")
    max_length = None if max_length_cfg is None else int(max_length_cfg)
    add_generation_prompt = bool(cfg.cast.train_add_generation_prompt)

    if cfg.cast.training_source == "official":
        official_root = Path(cfg.cast.official_data_root)
        cond_pos_raw, cond_neg_raw, cond_meta = official_condition_pairs(
            official_root,
            str(cfg.cast.official_condition_split),
            None if cfg.cast.official_condition_max_pairs is None else int(cfg.cast.official_condition_max_pairs),
        )
        cond_pos = [
            format_train_prompt(
                tokenizer,
                prompt,
                cfg.model.chat,
                add_generation_prompt=add_generation_prompt,
            )
            for prompt in cond_pos_raw
        ]
        cond_neg = [
            format_train_prompt(
                tokenizer,
                prompt,
                cfg.model.chat,
                add_generation_prompt=add_generation_prompt,
            )
            for prompt in cond_neg_raw
        ]
        beh_pos, beh_neg, beh_pos_lens, beh_neg_lens, beh_meta = official_behavior_pairs(
            official_root,
            tokenizer,
            cfg.model.chat,
            questions=int(cfg.cast.official_behavior_questions),
            suffixes=int(cfg.cast.official_behavior_suffixes),
            add_generation_prompt=add_generation_prompt,
        )
        behavior_pool = str(cfg.cast.behavior_pool)
    elif cfg.cast.training_source == "polyrefuse":
        languages = [str(language) for language in cfg.cast.source_languages]
        cond_pos, cond_neg, cond_meta = polyrefuse_budget_pairs(
            cfg,
            tokenizer,
            split=str(cfg.cast.train_split),
            languages=languages,
            add_generation_prompt=add_generation_prompt,
        )
        beh_pos, beh_neg = cond_pos, cond_neg
        beh_pos_lens = beh_neg_lens = None
        beh_meta = {
            **cond_meta,
            "source": "polyrefuse_budget_prompt_contrast",
            "positive": "harmful",
            "negative": "harmless",
        }
        behavior_pool = str(cfg.cast.adapted_behavior_pool)
    else:
        raise ValueError("cast.training_source must be 'official' or 'polyrefuse'")

    print(
        f"[train] condition pairs={len(cond_pos)} layers={condition_layers} "
        f"pool={cfg.cast.condition_pool}"
    )
    condition_directions = train_direction_set(
        model,
        tokenizer,
        cond_pos,
        cond_neg,
        condition_layers,
        pool=str(cfg.cast.condition_pool),
        method=str(cfg.cast.method),
        batch_size=batch_size,
        max_length=max_length,
    )
    print(f"[train] behavior pairs={len(beh_pos)} layers={behavior_layers} pool={behavior_pool}")
    behavior_directions = train_direction_set(
        model,
        tokenizer,
        beh_pos,
        beh_neg,
        behavior_layers,
        pool=behavior_pool,
        method=str(cfg.cast.method),
        batch_size=batch_size,
        max_length=max_length,
        positive_suffix_lengths=beh_pos_lens,
        negative_suffix_lengths=beh_neg_lens,
    )

    artifact = {
        "condition_directions": condition_directions,
        "behavior_directions": behavior_directions,
        "metadata": {
            "model": str(cfg.model.name),
            "variant": str(cfg.cast.variant),
            "training_source": str(cfg.cast.training_source),
            "method": str(cfg.cast.method),
            "train_add_generation_prompt": add_generation_prompt,
            "condition": cond_meta,
            "behavior": beh_meta,
            "condition_layers": condition_layers,
            "behavior_layers": behavior_layers,
        },
    }
    artifact_root.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, artifact_path)
    (artifact_root / "cast_vectors_meta.json").write_text(
        json.dumps(artifact["metadata"], indent=2, ensure_ascii=False) + "\n"
    )
    print(f"[done] vectors -> {artifact_path}")
    return artifact


def validation_prompts(
    cfg: DictConfig,
    tokenizer: Any,
    *,
    language: str | None,
) -> tuple[list[str], list[str], dict[str, Any]]:
    add_generation_prompt = bool(cfg.cast.train_add_generation_prompt)
    if cfg.cast.training_source == "official":
        pos_raw, neg_raw, meta = official_condition_pairs(
            Path(cfg.cast.official_data_root),
            str(cfg.cast.official_tune_split),
            None if cfg.cast.official_tune_max_pairs is None else int(cfg.cast.official_tune_max_pairs),
        )
        pos = [
            format_train_prompt(
                tokenizer,
                prompt,
                cfg.model.chat,
                add_generation_prompt=add_generation_prompt,
            )
            for prompt in pos_raw
        ]
        neg = [
            format_train_prompt(
                tokenizer,
                prompt,
                cfg.model.chat,
                add_generation_prompt=add_generation_prompt,
            )
            for prompt in neg_raw
        ]
        return pos, neg, {**meta, "tune_scope": "global_official"}

    if language is None:
        raise ValueError("polyrefuse validation requires a target language")
    harmful = load_polyrefuse(cfg.dataset.root, "harmful", str(cfg.cast.tune_split), language)
    harmless = load_polyrefuse(cfg.dataset.root, "harmless", str(cfg.cast.tune_split), language)
    pos = [
        format_train_prompt(
            tokenizer,
            prompt,
            cfg.model.chat,
            add_generation_prompt=add_generation_prompt,
        )
        for prompt in harmful
    ]
    neg = [
        format_train_prompt(
            tokenizer,
            prompt,
            cfg.model.chat,
            add_generation_prompt=add_generation_prompt,
        )
        for prompt in harmless
    ]
    return pos, neg, {
        "source": "polyrefuse_validation",
        "language": language,
        "split": str(cfg.cast.tune_split),
        "harmful": len(pos),
        "harmless": len(neg),
        "tune_scope": "per_language",
    }


def tune_condition_point(
    cfg: DictConfig,
    model: torch.nn.Module,
    tokenizer: Any,
    artifact: dict[str, Any],
    candidate_layers: list[int],
    *,
    language: str | None,
) -> tuple[CastConditionPoint, dict[str, Any]]:
    pos_prompts, neg_prompts, tune_meta = validation_prompts(cfg, tokenizer, language=language)
    batch_size = int(cfg.cast.extraction_batch_size)
    max_length_cfg = cfg.cast.get("max_length")
    max_length = None if max_length_cfg is None else int(max_length_cfg)
    pos_hiddens = extract_pooled_hiddens(
        model,
        tokenizer,
        pos_prompts,
        candidate_layers,
        pool=str(cfg.cast.condition_pool),
        batch_size=batch_size,
        max_length=max_length,
    )
    neg_hiddens = extract_pooled_hiddens(
        model,
        tokenizer,
        neg_prompts,
        candidate_layers,
        pool=str(cfg.cast.condition_pool),
        batch_size=batch_size,
        max_length=max_length,
    )
    condition_directions = artifact["condition_directions"]
    pos_sims = {
        int(layer): cast_condition_similarity(
            pos_hiddens[int(layer)],
            condition_directions[int(layer)].to(dtype=torch.float32),
        )
        for layer in candidate_layers
    }
    neg_sims = {
        int(layer): cast_condition_similarity(
            neg_hiddens[int(layer)],
            condition_directions[int(layer)].to(dtype=torch.float32),
        )
        for layer in candidate_layers
    }
    point = select_condition_point(
        pos_sims,
        neg_sims,
        candidate_layers=candidate_layers,
        threshold_min=float(cfg.cast.threshold.min),
        threshold_max=float(cfg.cast.threshold.max),
        threshold_step=float(cfg.cast.threshold.step),
        max_layers_to_combine=int(cfg.cast.threshold.max_layers_to_combine),
    )
    meta = {
        **tune_meta,
        "condition_layers": list(point.layers),
        "threshold": point.threshold,
        "comparator_threshold_is": point.comparator_threshold_is,
        "f1": point.f1,
    }
    print(
        f"[tune] {language or 'global'} layers={list(point.layers)} "
        f"threshold={point.threshold:.4f} comparator={point.comparator_threshold_is} "
        f"f1={point.f1:.3f}"
    )
    return point, meta


def condition_points_for_targets(
    cfg: DictConfig,
    model: torch.nn.Module,
    tokenizer: Any,
    artifact: dict[str, Any],
    candidate_layers: list[int],
) -> tuple[dict[str, CastConditionPoint], dict[str, Any]]:
    artifact_root = Path(cfg.cast.artifact_root)
    points_path = artifact_root / "condition_points.json"
    target_languages = [str(language) for language in cfg.target_languages]
    if points_path.exists() and not bool(cfg.cast.retune):
        raw = json.loads(points_path.read_text())
        points = {
            language: CastConditionPoint(
                layers=tuple(int(layer) for layer in item["condition_layers"]),
                threshold=float(item["threshold"]),
                comparator_threshold_is=str(item["comparator_threshold_is"]),
                f1=float(item["f1"]),
            )
            for language, item in raw["points"].items()
        }
        return points, raw

    if cfg.cast.training_source == "official":
        point, meta = tune_condition_point(
            cfg,
            model,
            tokenizer,
            artifact,
            candidate_layers,
            language=None,
        )
        points = {language: point for language in target_languages}
        raw_points = {"global": meta, **{language: meta for language in target_languages}}
    else:
        points = {}
        raw_points = {}
        for language in target_languages:
            point, meta = tune_condition_point(
                cfg,
                model,
                tokenizer,
                artifact,
                candidate_layers,
                language=language,
            )
            points[language] = point
            raw_points[language] = meta

    payload = {
        "model": str(cfg.model.name),
        "variant": str(cfg.cast.variant),
        "training_source": str(cfg.cast.training_source),
        "points": raw_points,
    }
    artifact_root.mkdir(parents=True, exist_ok=True)
    points_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(f"[done] condition points -> {points_path}")
    return points, payload


def make_bundle(
    cfg: DictConfig,
    artifact: dict[str, Any],
    point: CastConditionPoint,
    behavior_layers: list[int],
) -> CastBundle:
    return CastBundle(
        condition_directions={
            int(layer): artifact["condition_directions"][int(layer)].to(dtype=torch.float32)
            for layer in point.layers
        },
        behavior_directions={
            int(layer): artifact["behavior_directions"][int(layer)].to(dtype=torch.float32)
            for layer in behavior_layers
        },
        condition_layers=tuple(int(layer) for layer in point.layers),
        behavior_layers=tuple(int(layer) for layer in behavior_layers),
        threshold=float(point.threshold),
        comparator_threshold_is=str(point.comparator_threshold_is),
        behavior_strength=float(cfg.cast.behavior_strength),
        condition_mode=str(cfg.cast.inference_condition_mode),
        apply_behavior_on_first_call=bool(cfg.cast.apply_behavior_on_first_call),
    )


def run_cast_steering(cfg: DictConfig) -> None:
    model, tokenizer = load_generation_model(cfg.model)
    n_layers = int(model.config.num_hidden_layers)
    condition_layers = layer_list(cfg.cast.condition_layers, n_layers)
    behavior_layers = layer_list(cfg.cast.behavior_layers, n_layers)
    if not condition_layers:
        raise ValueError("CAST condition_layers resolved to empty list")
    if not behavior_layers:
        raise ValueError("CAST behavior_layers resolved to empty list")

    artifact = train_or_load_cast_artifact(cfg, model, tokenizer, condition_layers, behavior_layers)
    condition_points, condition_points_meta = condition_points_for_targets(
        cfg,
        model,
        tokenizer,
        artifact,
        condition_layers,
    )

    output_root = Path(cfg.output_root)
    combos = list(itertools.product(cfg.target_languages, cfg.splits, cfg.subsets))
    for language, split, subset in combos:
        language = str(language)
        split = str(split)
        subset = str(subset)
        out_path = completions_file(output_root, language, split, subset)
        if out_path.exists() and not bool(cfg.overwrite):
            print(f"[skip] {language}/{split}/{subset} -> {out_path}")
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
        bundle = make_bundle(cfg, artifact, condition_points[language], behavior_layers)
        with install_cast_hooks(model, bundle):
            completions = generate_completions(model, tokenizer, prompts, cfg.generation)

        point = condition_points[language]
        meta = {
            "model": str(cfg.model.name),
            "generation": OmegaConf.to_container(cfg.generation, resolve=True),
            "language": language,
            "split": split,
            "subset": subset,
            "cast": {
                "variant": str(cfg.cast.variant),
                "training_source": str(cfg.cast.training_source),
                "artifact_root": str(cfg.cast.artifact_root),
                "method": str(cfg.cast.method),
                "condition_layers": list(point.layers),
                "threshold": float(point.threshold),
                "comparator_threshold_is": str(point.comparator_threshold_is),
                "condition_f1": float(point.f1),
                "behavior_layers": behavior_layers,
                "behavior_strength": float(cfg.cast.behavior_strength),
                "condition_mode": str(cfg.cast.inference_condition_mode),
                "training_meta": artifact["metadata"],
                "condition_points": condition_points_meta,
            },
        }
        write_jsonl_with_meta(out_path, meta, completion_rows(instructions, completions))
        print(f"[done] {out_path}")


@hydra.main(version_base=None, config_path="../../configs", config_name="routing_refusal/run_cast_steering")
def main(cfg: DictConfig) -> None:
    run_cast_steering(cfg)


if __name__ == "__main__":
    main()
