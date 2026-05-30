"""Conditional v_HRL training, gating, and generation utilities.

The gate is the sample-efficient HRL-only subspace logistic classifier:
source HRL train activations define the SVD subspace. For budgeted LRL runs,
each target LRL gets a tiny balanced train budget and its own threshold fit on
those examples. For b=0, the gate and threshold are fit from HRL source train
activations only.
"""


import json
import random
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import load_file
from tqdm import tqdm

from multilingual_latent_safety.activation_store import ActivationCache
from multilingual_latent_safety.analysis import load_resource_tiers
from multilingual_latent_safety.data import load_polyrefuse
from multilingual_latent_safety.generation import generation_kwargs, load_generation_model
from multilingual_latent_safety.interventions import (
    get_decoder_blocks,
    install_hooks,
    make_conditional_refusal_hook,
)
from multilingual_latent_safety.json_io import write_jsonl_with_meta
from multilingual_latent_safety.probe_evaluation import (
    LogisticProbe,
    ScoreBundle,
    contrast_direction,
    fit_logistic_probe,
    select_global_threshold_for_objective,
    source_pair,
    stack_pairs,
    subspace_basis,
)
from multilingual_latent_safety.model import format_prompt
from multilingual_latent_safety.paths import (
    completions_file,
    pooled_direction_file,
)
from multilingual_latent_safety.probes import sample_balanced_indices
from multilingual_latent_safety.runtime import (
    batched,
    set_seed,
    stable_seed,
)
from multilingual_latent_safety.vector_ops import unit_normalize


@dataclass(frozen=True)
class LanguageGate:
    basis: torch.Tensor
    probe: LogisticProbe
    budget: int


@dataclass(frozen=True)
class GateTrainingResult:
    gates: dict[str, LanguageGate]
    thresholds: dict[str, float]
    threshold_fit_scores: dict[str, float]
    threshold_source: str
    source_languages: list[str]


@dataclass(frozen=True)
class AlphaCalibration:
    source: str
    base: float
    metadata: dict[str, Any]


@dataclass
class EvalItem:
    language: str
    split: str
    subset: str
    prompt_id: int
    instruction: str
    prompt: str = ""
    activation_norm: float = 0.0
    gate_logit: float = 0.0
    gate_pred_harmful: bool = False


def project_features(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x.to(device=basis.device, dtype=torch.float32) @ basis.to(dtype=torch.float32)


def score_bundle_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> ScoreBundle:
    labels = labels.to(device=logits.device, dtype=torch.float32)
    return ScoreBundle(harmful=logits[labels > 0.5], harmless=logits[labels <= 0.5])


def select_gate_thresholds(
    threshold_scores: Mapping[str, ScoreBundle],
    target_languages: Sequence[str],
    objective: str,
    threshold_source: str,
) -> tuple[dict[str, float], dict[str, float], str]:
    if not threshold_scores:
        raise ValueError("threshold score map must be non-empty")

    if threshold_source == "source_train":
        choice = select_global_threshold_for_objective(
            dict(threshold_scores),
            objective=objective,
        )
        thresholds = {language: choice.threshold for language in target_languages}
        scores = {language: choice.validation_macro_f1 for language in target_languages}
        return thresholds, scores, threshold_source

    if threshold_source == "target_train_budget":
        missing = sorted(set(target_languages) - set(threshold_scores))
        if missing:
            raise ValueError(f"missing target threshold scores for {missing}")
        thresholds: dict[str, float] = {}
        scores: dict[str, float] = {}
        for language in target_languages:
            choice = select_global_threshold_for_objective(
                {language: threshold_scores[language]},
                objective=objective,
            )
            thresholds[language] = choice.threshold
            scores[language] = choice.validation_macro_f1
        return thresholds, scores, threshold_source

    raise ValueError(f"unknown threshold source {threshold_source!r}")


def train_source_only_gates(
    cfg: DictConfig,
    target_languages: Sequence[str],
    source_languages: list[str],
    basis: torch.Tensor,
    source_features: torch.Tensor,
    source_y: torch.Tensor,
) -> GateTrainingResult:
    probe = fit_logistic_probe(
        source_features,
        source_y,
        seed=stable_seed(int(cfg.seed), cfg.model.name, cfg.rank, "source-only-probe-fit"),
        l2=float(cfg.probe.l2),
        lr=float(cfg.probe.lr),
        epochs=int(cfg.probe.epochs),
        standardize=bool(cfg.probe.standardize),
        balanced_loss=bool(cfg.probe.balanced_loss),
    )
    with torch.no_grad():
        source_logits = probe.logits(source_features)
    thresholds, fit_scores, threshold_source = select_gate_thresholds(
        {"source_train": score_bundle_from_logits(source_logits, source_y)},
        [str(language) for language in target_languages],
        str(cfg.threshold_objective),
        "source_train",
    )
    gates = {
        str(language): LanguageGate(basis=basis, probe=probe, budget=0)
        for language in target_languages
    }
    return GateTrainingResult(
        gates=gates,
        thresholds=thresholds,
        threshold_fit_scores=fit_scores,
        threshold_source=threshold_source,
        source_languages=source_languages,
    )


def train_gates(cfg: DictConfig) -> GateTrainingResult:
    probe_device = str(cfg.probe_device)
    device = torch.device("cuda" if probe_device == "auto" and torch.cuda.is_available() else probe_device)
    tiers = load_resource_tiers(cfg.dataset_config)
    all_languages = list(cfg.dataset.languages)
    source_languages = list(cfg.source_languages) or [
        language for language in all_languages if tiers[language] == "high"
    ]
    target_languages = list(cfg.target_languages)
    cache = ActivationCache(
        Path(cfg.activations_root),
        int(cfg.layer),
        str(cfg.token_position),
        device,
    )

    source_directions = torch.stack(
        [
            contrast_direction(*cache.pair(language, str(cfg.train_split)))
            for language in source_languages
        ],
        dim=0,
    )
    basis = subspace_basis(source_directions, int(cfg.rank)).to(device)
    source_x, source_y = source_pair(cache, source_languages, str(cfg.train_split))
    source_features = project_features(source_x, basis)

    requested_budget = int(cfg.budget)
    if requested_budget < 0:
        raise ValueError(f"budget must be non-negative, got {requested_budget}")
    if requested_budget == 0:
        return train_source_only_gates(
            cfg,
            target_languages,
            source_languages,
            basis,
            source_features,
            source_y,
        )

    gates: dict[str, LanguageGate] = {}
    calibration_scores: dict[str, ScoreBundle] = {}
    for language in target_languages:
        harmful, harmless = cache.pair(language, str(cfg.train_split))
        budget = min(requested_budget, harmful.shape[0], harmless.shape[0])
        if budget < 1:
            raise ValueError(
                f"budget={requested_budget} leaves no target examples for {language}: "
                f"harmful={harmful.shape[0]}, harmless={harmless.shape[0]}"
            )
        draw_seed = stable_seed(
            int(cfg.seed), cfg.model.name, language, cfg.budget, cfg.rank, "conditional-vhrl-gate"
        )
        h_idx, n_idx = sample_balanced_indices(
            harmful.shape[0],
            harmless.shape[0],
            budget,
            draw_seed,
        )
        few_h = harmful[h_idx.to(harmful.device)]
        few_n = harmless[n_idx.to(harmless.device)]
        target_x, target_y = stack_pairs([(few_h, few_n)])
        target_features = project_features(target_x, basis)
        train_x = torch.cat([source_features, target_features], dim=0)
        train_y = torch.cat([source_y, target_y], dim=0)
        probe = fit_logistic_probe(
            train_x,
            train_y,
            seed=stable_seed(
                int(cfg.seed),
                cfg.model.name,
                language,
                cfg.budget,
                cfg.rank,
                "probe-fit",
            ),
            l2=float(cfg.probe.l2),
            lr=float(cfg.probe.lr),
            epochs=int(cfg.probe.epochs),
            standardize=bool(cfg.probe.standardize),
            balanced_loss=bool(cfg.probe.balanced_loss),
        )
        with torch.no_grad():
            logits = probe.logits(target_features)
        calibration_scores[language] = ScoreBundle(
            harmful=logits[target_y > 0.5],
            harmless=logits[target_y <= 0.5],
        )
        gates[language] = LanguageGate(basis=basis, probe=probe, budget=budget)

    thresholds, fit_scores, threshold_source = select_gate_thresholds(
        calibration_scores,
        target_languages,
        str(cfg.threshold_objective),
        "target_train_budget",
    )
    return GateTrainingResult(
        gates=gates,
        thresholds=thresholds,
        threshold_fit_scores=fit_scores,
        threshold_source=threshold_source,
        source_languages=source_languages,
    )


def per_language_quota(languages: Sequence[str], total: int, seed: int) -> dict[str, int]:
    if total <= 0:
        return {language: 0 for language in languages}
    ordered = list(languages)
    random.Random(seed).shuffle(ordered)
    base = total // len(ordered)
    extra = total % len(ordered)
    return {language: base + int(idx < extra) for idx, language in enumerate(ordered)}


def samples_per_subset_value(cfg: DictConfig) -> int | None:
    value = cfg.get("samples_per_subset")
    if value is None:
        return None
    return int(value)


def sample_eval_items(cfg: DictConfig) -> list[EvalItem]:
    items: list[EvalItem] = []
    samples_per_subset = samples_per_subset_value(cfg)
    for subset in cfg.subsets:
        quota = None
        if samples_per_subset is not None:
            quota = per_language_quota(
                list(cfg.target_languages),
                samples_per_subset,
                stable_seed(int(cfg.seed), cfg.evaluation_split, subset, "eval-quota"),
            )
        for language in cfg.target_languages:
            instructions = load_polyrefuse(
                root=cfg.dataset.root,
                subset=str(subset),
                split=str(cfg.evaluation_split),
                language=str(language),
            )
            rows = [
                EvalItem(
                    language=str(language),
                    split=str(cfg.evaluation_split),
                    subset=str(subset),
                    prompt_id=idx,
                    instruction=instruction,
                )
                for idx, instruction in enumerate(instructions)
                ]
            rng = random.Random(stable_seed(int(cfg.seed), language, subset, "eval-sample"))
            if quota is None:
                selected = rows
            else:
                selected = rng.sample(rows, min(quota[str(language)], len(rows)))
            items.extend(selected)
    return items


@torch.inference_mode()
def capture_final_prompt_activations(
    model: torch.nn.Module,
    tokenizer,
    prompts: Sequence[str],
    layer: int,
    batch_size: int,
) -> torch.Tensor:
    blocks = get_decoder_blocks(model)
    device = getattr(model, "device", next(model.parameters()).device)
    captured: list[torch.Tensor] = []

    def hook(module: torch.nn.Module, inputs: tuple, output: Any) -> None:
        residual = output[0] if isinstance(output, tuple) else output
        captured.append(residual[:, -1, :].detach().to(torch.float32).cpu())

    handle = blocks[int(layer)].register_forward_hook(hook)
    try:
        for batch in tqdm(list(batched(list(prompts), int(batch_size))), desc="capture"):
            inputs = tokenizer(
                list(batch),
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            model(**inputs, use_cache=False)
            del inputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        handle.remove()
    return torch.cat(captured, dim=0)


def annotate_gate_outputs(
    items: list[EvalItem],
    activations: torch.Tensor,
    gates: dict[str, LanguageGate],
    thresholds: Mapping[str, float],
) -> None:
    for item, activation in zip(items, activations, strict=True):
        gate = gates[item.language]
        with torch.no_grad():
            feature = project_features(activation.unsqueeze(0), gate.basis)
            logit = float(gate.probe.logits(feature).item())
        item.activation_norm = float(activation.norm().item())
        item.gate_logit = logit
        item.gate_pred_harmful = logit >= float(thresholds[item.language])


@torch.inference_mode()
def generate_with_conditional_intervention(
    model: torch.nn.Module,
    tokenizer,
    items: Sequence[EvalItem],
    direction: torch.Tensor,
    layer: int,
    alpha: float,
    gen_cfg: DictConfig,
) -> list[str]:
    set_seed(int(gen_cfg.seed))
    kwargs = generation_kwargs(gen_cfg, tokenizer)
    device = getattr(model, "device", next(model.parameters()).device)
    completions: list[str] = []
    for batch_items in tqdm(list(batched(list(items), int(gen_cfg.batch_size))), desc="generate"):
        prompts = [item.prompt for item in batch_items]
        harmful_mask = torch.tensor(
            [item.gate_pred_harmful for item in batch_items],
            dtype=torch.bool,
        )
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        hook_factory = lambda hook_device, hook_dtype: make_conditional_refusal_hook(
            direction,
            alpha,
            harmful_mask,
            hook_device,
            hook_dtype,
        )
        with install_hooks(model, [int(layer)], hook_factory):
            out = model.generate(**inputs, **kwargs)
        prompt_len = inputs["input_ids"].shape[1]
        texts = tokenizer.batch_decode(out[:, prompt_len:], skip_special_tokens=True)
        completions.extend(texts)
        del out, inputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return completions


def as_nonzero_lambdas(values: Sequence[float]) -> list[float]:
    lambdas = [float(value) for value in values]
    zero_lambdas = [value for value in lambdas if value == 0.0]
    if zero_lambdas:
        raise ValueError("conditional v_HRL steering does not generate lambda=0 baselines")
    return lambdas


def experiment_root(cfg: DictConfig) -> Path:
    return Path(cfg.output_root) / (
        f"conditional_vhrl_hrl_subspace_rank={int(cfg.rank)}"
        f"_budget={int(cfg.budget)}_layer={int(cfg.layer)}"
    )


def conditional_hrl_projection_gap(
    cfg: DictConfig,
    source_languages: Sequence[str],
    direction: torch.Tensor,
) -> AlphaCalibration:
    languages = [str(language) for language in source_languages]
    if not languages:
        raise ValueError("hrl_projection_gap requires at least one source language")

    unit_direction = unit_normalize(direction.flatten().to(dtype=torch.float32), dim=0)
    cache = ActivationCache(
        cfg.activations_root,
        int(cfg.layer),
        str(cfg.token_position),
    )
    harmful_scores: list[torch.Tensor] = []
    harmless_scores: list[torch.Tensor] = []
    split = str(cfg.alpha_calibration_split)
    harmful_quantile = float(cfg.get("alpha_projection_harmful_quantile", 0.5))
    harmless_quantile = float(cfg.get("alpha_projection_harmless_quantile", 0.5))
    if not 0.0 <= harmful_quantile <= 1.0:
        raise ValueError(
            "alpha_projection_harmful_quantile must be in [0, 1], "
            f"got {harmful_quantile}"
        )
    if not 0.0 <= harmless_quantile <= 1.0:
        raise ValueError(
            "alpha_projection_harmless_quantile must be in [0, 1], "
            f"got {harmless_quantile}"
        )
    for language in languages:
        harmful, harmless = cache.pair(language, split)
        if (
            harmful.shape[1] != unit_direction.numel()
            or harmless.shape[1] != unit_direction.numel()
        ):
            raise ValueError(
                f"activation width mismatch for {language}: "
                f"harmful={harmful.shape}, harmless={harmless.shape}, "
                f"direction={tuple(unit_direction.shape)}"
            )
        harmful_scores.append(harmful.to(dtype=torch.float32) @ unit_direction)
        harmless_scores.append(harmless.to(dtype=torch.float32) @ unit_direction)

    harmful_projection = torch.cat(harmful_scores)
    harmless_projection = torch.cat(harmless_scores)
    harmful_value = float(torch.quantile(harmful_projection, harmful_quantile).item())
    harmless_value = float(torch.quantile(harmless_projection, harmless_quantile).item())
    harmful_median = float(torch.quantile(harmful_projection, 0.5).item())
    harmless_median = float(torch.quantile(harmless_projection, 0.5).item())
    gap = harmful_value - harmless_value
    if not bool(torch.isfinite(torch.tensor(gap)).item()) or gap <= 0.0:
        raise ValueError(
            "hrl_projection_gap must be positive and finite, got "
            f"{gap:.6g} (harmful_q{harmful_quantile:g}={harmful_value:.6g}, "
            f"harmless_q{harmless_quantile:g}={harmless_value:.6g})"
        )

    metadata: dict[str, Any] = {
        "alpha_base": gap,
        "alpha_calibration_languages": languages,
        "alpha_calibration_split": split,
        "alpha_calibration_token_position": str(cfg.token_position),
        "hrl_projection_harmful_quantile": harmful_quantile,
        "hrl_projection_harmless_quantile": harmless_quantile,
        "hrl_projection_harmful_value": harmful_value,
        "hrl_projection_harmless_value": harmless_value,
        "hrl_projection_harmful_median": harmful_median,
        "hrl_projection_harmless_median": harmless_median,
        "hrl_projection_gap": gap,
        "hrl_projection_harmful_n": int(harmful_projection.numel()),
        "hrl_projection_harmless_n": int(harmless_projection.numel()),
    }
    return AlphaCalibration(source="hrl_projection_gap", base=gap, metadata=metadata)


def conditional_alpha_calibration(
    cfg: DictConfig,
    source_languages: Sequence[str],
    direction: torch.Tensor,
) -> AlphaCalibration:
    return conditional_hrl_projection_gap(cfg, source_languages, direction)


def write_completions(
    root: Path,
    cfg: DictConfig,
    items: Sequence[EvalItem],
    completions: Sequence[str],
    *,
    lambda_value: float,
    alpha: float,
    alpha_calibration: AlphaCalibration,
    thresholds: Mapping[str, float],
    threshold_fit_scores: Mapping[str, float],
    threshold_source: str,
) -> None:
    rows_by_cell: dict[tuple[str, str, str], list[tuple[EvalItem, str]]] = defaultdict(list)
    for item, completion in zip(items, completions, strict=True):
        rows_by_cell[(item.language, item.split, item.subset)].append((item, completion))

    for (language, split, subset), rows in rows_by_cell.items():
        out_file = completions_file(root, language, split, subset)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "model": cfg.model.name,
            "generation": OmegaConf.to_container(cfg.generation, resolve=True),
            "language": language,
            "split": split,
            "subset": subset,
            "dataset": cfg.dataset.name,
            "intervention": "conditional_vhrl",
            "direction": "v_HRL",
            "harmful_scope": "final_prompt_then_gen",
            "harmless_scope": "all_tokens_directional_ablation",
            "gate": "hrl_only_subspace_logistic",
            "rank": int(cfg.rank),
            "budget": int(cfg.budget),
            "layer": int(cfg.layer),
            "lambda": float(lambda_value),
            "alpha": float(alpha),
            "gate_threshold": float(thresholds[language]),
            "gate_threshold_source": str(threshold_source),
            "gate_threshold_objective": str(cfg.threshold_objective),
            "gate_thresholds": {
                str(threshold_language): float(threshold)
                for threshold_language, threshold in thresholds.items()
            },
            "gate_threshold_fit_score": float(threshold_fit_scores[language]),
            "gate_threshold_fit_scores": {
                str(score_language): float(score)
                for score_language, score in threshold_fit_scores.items()
            },
        }
        meta.update(alpha_calibration.metadata)
        write_jsonl_with_meta(
            out_file,
            meta,
            [
                {
                    "prompt_id": item.prompt_id,
                    "instruction": item.instruction,
                    "completion": completion,
                    "gate_logit": item.gate_logit,
                    "gate_pred_harmful": item.gate_pred_harmful,
                    "activation_norm": item.activation_norm,
                }
                for item, completion in rows
            ],
        )


def completion_files_exist(root: Path, items: Sequence[EvalItem]) -> bool:
    cells = {(item.language, item.split, item.subset) for item in items}
    return all(
        completions_file(root, language, split, subset).exists()
        for language, split, subset in cells
    )


def group_items_by_cell(items: Sequence[EvalItem]) -> dict[tuple[str, str, str], list[EvalItem]]:
    grouped: dict[tuple[str, str, str], list[EvalItem]] = defaultdict(list)
    for item in items:
        grouped[(item.language, item.split, item.subset)].append(item)
    return grouped


def run_conditional_vhrl(cfg: DictConfig) -> None:
    gate_result = train_gates(cfg)
    model, tokenizer = load_generation_model(cfg.model)
    direction_path = pooled_direction_file(
        Path(cfg.pooled_root),
        "hrl",
        cfg.token_position,
        int(cfg.layer),
    )
    direction = load_file(direction_path)["direction"]

    items = sample_eval_items(cfg)
    for item in items:
        item.prompt = format_prompt(tokenizer, item.instruction, cfg.model.chat)

    activations = capture_final_prompt_activations(
        model,
        tokenizer,
        [item.prompt for item in items],
        int(cfg.layer),
        int(cfg.capture_batch_size),
    )
    annotate_gate_outputs(items, activations, gate_result.gates, gate_result.thresholds)
    alpha_calibration = conditional_alpha_calibration(cfg, gate_result.source_languages, direction)
    base_root = experiment_root(cfg)
    base_root.mkdir(parents=True, exist_ok=True)

    summary = {
        "model": cfg.model.name,
        "layer": int(cfg.layer),
        "rank": int(cfg.rank),
        "budget": int(cfg.budget),
        "source_languages": gate_result.source_languages,
        "target_languages": list(cfg.target_languages),
        "thresholds": {
            str(language): float(threshold)
            for language, threshold in gate_result.thresholds.items()
        },
        "threshold_source": gate_result.threshold_source,
        "threshold_objective": str(cfg.threshold_objective),
        "threshold_fit_scores": {
            str(language): float(score)
            for language, score in gate_result.threshold_fit_scores.items()
        },
        "n_items": len(items),
        "n_harmful": sum(item.subset == "harmful" for item in items),
        "n_harmless": sum(item.subset == "harmless" for item in items),
        "gate_pred_harmful": sum(item.gate_pred_harmful for item in items),
    }
    summary.update(alpha_calibration.metadata)
    (base_root / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    items_by_cell = group_items_by_cell(items)
    for lambda_value in as_nonzero_lambdas(cfg.lambdas):
        out_root = base_root / f"lambda={float(lambda_value):g}"
        alpha = float(lambda_value) * alpha_calibration.base
        for (language, split, subset), cell_items in items_by_cell.items():
            out_file = completions_file(out_root, language, split, subset)
            if out_file.exists() and not bool(cfg.overwrite):
                print(f"[skip] {language}/{split}/{subset} -> {out_file}")
                continue
            print(
                f"[run ] {language}/{split}/{subset} lambda={lambda_value:g} "
                f"alpha={alpha:.4f} threshold_source={gate_result.threshold_source} "
                f"items={len(cell_items)}"
            )
            completions = generate_with_conditional_intervention(
                model,
                tokenizer,
                cell_items,
                direction,
                int(cfg.layer),
                alpha,
                cfg.generation,
            )
            write_completions(
                out_root,
                cfg,
                cell_items,
                completions,
                lambda_value=lambda_value,
                alpha=alpha,
                alpha_calibration=alpha_calibration,
                thresholds=gate_result.thresholds,
                threshold_fit_scores=gate_result.threshold_fit_scores,
                threshold_source=gate_result.threshold_source,
            )
            print(f"[done] {out_file}")
        print(f"[done] {out_root}")
