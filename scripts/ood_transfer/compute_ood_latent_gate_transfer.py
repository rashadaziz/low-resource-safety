"""OOD transfer evaluation for latent harmfulness gates."""

import json
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

from multilingual_latent_safety.activation_store import ActivationCache
from multilingual_latent_safety.csv_io import write_rows
from multilingual_latent_safety.probe_evaluation import (
    ScoreBundle,
    append_macro_row,
    binary_metrics,
    direction_scores,
    language_threshold_fit_scores,
    logistic_logits,
    language_direction_matrix,
    pooled_contrast_direction,
    select_global_threshold_for_objective,
    select_language_thresholds,
    source_languages,
    source_pair,
    source_train_direction_scores,
    source_train_logit_scores,
    split_logits_by_split,
    stack_pairs,
    subspace_basis,
)
from multilingual_latent_safety.probes import sample_balanced_indices
from multilingual_latent_safety.runtime import stable_seed


def target_tiers(target_languages: list[str]) -> dict[str, str]:
    return {language: "ood" for language in target_languages}


def model_meta(model_cfg) -> dict[str, object]:
    return {
        "model": str(model_cfg.name),
        "model_short": str(model_cfg.short),
        "layer": int(model_cfg.layer),
    }


def target_pair_scores(cache: ActivationCache, language: str, split: str, direction: torch.Tensor) -> ScoreBundle:
    return direction_scores(direction, *cache.pair(language, split))


CategoryIndices = dict[tuple[str, str, str, str], list[int]]


def polyrefuse_json_path(root: Path, subset: str, split: str, language: str) -> Path:
    return root / f"{subset}_{split}_translated_{language}.json"


def normalize_category_value(value: object) -> str:
    text = "" if value is None else str(value).strip()
    return text if text else "__missing__"


def load_indosafety_category_indices(
    dataset_root: Path,
    *,
    languages: list[str],
    split: str,
    columns: list[str],
) -> CategoryIndices:
    indices: CategoryIndices = {}
    for language in languages:
        if not language.startswith("indosafety_"):
            continue
        path = polyrefuse_json_path(dataset_root, "harmful", split, language)
        if not path.exists():
            continue
        with path.open() as f:
            rows = json.load(f)
        for idx, row in enumerate(rows):
            for column in columns:
                category = normalize_category_value(row.get(column))
                indices.setdefault((language, split, column, category), []).append(idx)
    return indices


def threshold_choices(
    validation_scores: dict[str, ScoreBundle],
    threshold_scores: dict[str, ScoreBundle] | None,
    *,
    threshold_scope: str,
) -> tuple[dict[str, object], float | str, float]:
    fit_scores = validation_scores if threshold_scores is None else threshold_scores
    if not fit_scores:
        raise ValueError("threshold score map must be non-empty")

    if threshold_scope == "global":
        choice = select_global_threshold_for_objective(fit_scores, objective="macro_f1")
        return {language: choice for language in validation_scores}, choice.threshold, choice.validation_macro_f1
    if threshold_scope == "global_balanced":
        choice = select_global_threshold_for_objective(fit_scores, objective="balanced_macro_f1")
        return {language: choice for language in validation_scores}, choice.threshold, choice.validation_macro_f1
    if threshold_scope == "language_oracle":
        aligned = language_threshold_fit_scores(fit_scores, validation_scores)
        choices = select_language_thresholds(aligned)
        selection_score = float(
            torch.tensor(
                [choice.validation_macro_f1 for choice in choices.values()],
                dtype=torch.float32,
            )
            .mean()
            .item()
        )
        return choices, "", selection_score
    raise ValueError(f"unknown threshold scope {threshold_scope!r}")


def append_ood_test_rows(
    rows: list[dict[str, object]],
    *,
    metadata: dict[str, object],
    target_tiers: dict[str, str],
    test_scores: dict[str, ScoreBundle],
    threshold_scores: dict[str, ScoreBundle] | None,
    threshold_source: str,
    threshold_scope: str,
) -> None:
    thresholds, threshold_value, selection_score = threshold_choices(
        test_scores,
        threshold_scores,
        threshold_scope=threshold_scope,
    )
    metric_rows: list[dict[str, float | int]] = []
    for language, scores in test_scores.items():
        threshold = float(thresholds[language].threshold)  # type: ignore[attr-defined]
        metrics = binary_metrics(scores, threshold)
        metric_rows.append(metrics)
        rows.append(
            {
                **metadata,
                "split": "test",
                "target_language": language,
                "target_tier": target_tiers[language],
                "threshold_scope": threshold_scope,
                "threshold_source": threshold_source,
                "threshold": threshold,
                "threshold_selection_macro_f1_at_threshold": selection_score,
                "validation_macro_f1_at_threshold": selection_score,
                **metrics,
            }
        )
    append_macro_row(
        rows,
        metadata=metadata,
        split="test",
        target_language="__macro__",
        target_tier="all",
        threshold_scope=threshold_scope,
        threshold_source=threshold_source,
        threshold_value=threshold_value,
        selection_score=selection_score,
        metric_rows=metric_rows,
    )


def _sample_subset(
    cache: ActivationCache,
    language: str,
    split: str,
    subset: str,
    budget: int,
) -> torch.Tensor | None:
    try:
        tensor = cache.subset(language, split, subset)
    except FileNotFoundError:
        return None
    if tensor.shape[0] < budget:
        return None
    return tensor


def sample_ood_pair(
    cache: ActivationCache,
    language: str,
    *,
    train_split: str,
    test_split: str,
    budget: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, str, str]:
    harmful_split = train_split
    harmful = _sample_subset(cache, language, train_split, "harmful", budget)
    if harmful is None:
        harmful_split = test_split
        harmful = _sample_subset(cache, language, test_split, "harmful", budget)
    harmless_split = train_split
    harmless = _sample_subset(cache, language, train_split, "harmless", budget)
    if harmless is None:
        harmless_split = test_split
        harmless = _sample_subset(cache, language, test_split, "harmless", budget)
    if harmful is None or harmless is None:
        raise ValueError(f"cannot sample {budget}-shot OOD pair for {language}")
    h_idx, n_idx = sample_balanced_indices(harmful.shape[0], harmless.shape[0], budget, seed)
    return (
        harmful[h_idx.to(harmful.device)],
        harmless[n_idx.to(harmless.device)],
        harmful_split,
        harmless_split,
    )


def append_indosafety_category_rows(
    rows: list[dict[str, object]],
    *,
    metadata: dict[str, object],
    target_tiers: dict[str, str],
    validation_scores: dict[str, ScoreBundle],
    test_scores: dict[str, ScoreBundle],
    threshold_scores: dict[str, ScoreBundle] | None,
    threshold_source: str,
    threshold_scope: str,
    category_indices: CategoryIndices,
    category_columns: list[str],
    test_split: str,
    min_harmful: int,
) -> None:
    if not category_indices:
        return
    thresholds, threshold_value, selection_score = threshold_choices(
        validation_scores,
        threshold_scores,
        threshold_scope=threshold_scope,
    )
    for language, scores in test_scores.items():
        if language not in thresholds:
            continue
        threshold = float(thresholds[language].threshold)  # type: ignore[attr-defined]
        for column in category_columns:
            keys = sorted(
                key
                for key in category_indices
                if key[0] == language and key[1] == test_split and key[2] == column
            )
            for _, _, _, category in keys:
                indices = category_indices[(language, test_split, column, category)]
                if len(indices) < min_harmful:
                    continue
                index = torch.tensor(indices, dtype=torch.long, device=scores.harmful.device)
                if int(index.max().item()) >= scores.harmful.numel():
                    raise ValueError(
                        f"category index out of range for {language}/{test_split}/{column}/{category}"
                    )
                metrics = binary_metrics(
                    ScoreBundle(
                        harmful=scores.harmful.index_select(0, index),
                        harmless=scores.harmless,
                    ),
                    threshold,
                )
                rows.append(
                    {
                        **metadata,
                        "split": "test",
                        "target_language": language,
                        "target_tier": target_tiers[language],
                        "evaluation_group": "indosafety_category",
                        "indosafety_category_column": column,
                        "indosafety_category": category,
                        "harmful_scope": f"{test_split}:{column}",
                        "harmless_scope": "all_harmless_for_language_split",
                        "threshold_scope": threshold_scope,
                        "threshold_source": threshold_source,
                        "threshold": threshold if threshold_value != "" else threshold,
                        "threshold_selection_macro_f1_at_threshold": selection_score,
                        "validation_macro_f1_at_threshold": selection_score,
                        **metrics,
                    }
                )


def append_hrl_direction_rows(
    rows: list[dict[str, object]],
    *,
    category_rows: list[dict[str, object]] | None = None,
    category_indices: CategoryIndices | None = None,
    category_columns: list[str] | None = None,
    source_cache: ActivationCache,
    target_cache: ActivationCache,
    meta: dict[str, object],
    source_langs: list[str],
    target_langs: list[str],
    tiers: dict[str, str],
    cfg: DictConfig,
) -> None:
    direction = pooled_contrast_direction(source_cache, source_langs, str(cfg.source_train_split))
    test_scores = {
        target: target_pair_scores(target_cache, target, str(cfg.target_test_split), direction)
        for target in target_langs
    }
    threshold_scores = {
        "source_train": source_train_direction_scores(
            source_cache,
            source_langs,
            str(cfg.source_train_split),
            direction,
        )
    }
    method_meta = {
        **meta,
        "method": "ood_hrl_direction",
        "classifier": "mean_difference",
        "gate": "v_hrl",
        "adaptation": "threshold_source_train",
        "shot": 0,
        "budget": 0,
        "seed": "",
        "source_group": "hrl",
        "source_languages": ",".join(source_langs),
        "source_n": len(source_langs),
        "rank": "",
    }
    append_ood_test_rows(
        rows,
        metadata=method_meta,
        target_tiers=tiers,
        test_scores=test_scores,
        threshold_scores=threshold_scores,
        threshold_source="source_train",
        threshold_scope=str(cfg.threshold_selection),
    )
    if category_rows is not None and category_indices is not None and category_columns is not None:
        append_indosafety_category_rows(
            category_rows,
            metadata=method_meta,
            target_tiers=tiers,
            validation_scores=test_scores,
            test_scores=test_scores,
            threshold_scores=threshold_scores,
            threshold_source="source_train",
            threshold_scope=str(cfg.threshold_selection),
            category_indices=category_indices,
            category_columns=category_columns,
            test_split=str(cfg.target_test_split),
            min_harmful=int(cfg.category_breakdown.min_harmful),
        )

    for seed in range(int(cfg.seeds)):
        validation_scores = {}
        test_scores = {}
        threshold_scores = {}
        for target in target_langs:
            draw_seed = stable_seed(int(cfg.seed_offset), seed, meta["model"], target, "v_hrl")
            few_h, few_n, _, _ = sample_ood_pair(
                target_cache,
                target,
                train_split=str(cfg.target_train_split),
                test_split=str(cfg.target_test_split),
                budget=int(cfg.budget),
                seed=draw_seed,
            )
            threshold_scores[target] = direction_scores(direction, few_h, few_n)
            validation_scores[target] = threshold_scores[target]
            test_scores[target] = target_pair_scores(target_cache, target, str(cfg.target_test_split), direction)
        method_meta = {
            **meta,
            "method": "ood_hrl_direction",
            "classifier": "mean_difference",
            "gate": "v_hrl",
            "adaptation": "threshold_target_budget",
            "shot": int(cfg.budget),
            "budget": int(cfg.budget),
            "seed": seed,
            "source_group": "hrl",
            "source_languages": ",".join(source_langs),
            "source_n": len(source_langs),
            "rank": "",
        }
        append_ood_test_rows(
            rows,
            metadata=method_meta,
            target_tiers=tiers,
            test_scores=test_scores,
            threshold_scores=threshold_scores,
            threshold_source="target_budget",
            threshold_scope=str(cfg.threshold_selection),
        )
        if category_rows is not None and category_indices is not None and category_columns is not None:
            append_indosafety_category_rows(
                category_rows,
                metadata=method_meta,
                target_tiers=tiers,
                validation_scores=validation_scores,
                test_scores=test_scores,
                threshold_scores=threshold_scores,
                threshold_source="target_budget",
                threshold_scope=str(cfg.threshold_selection),
                category_indices=category_indices,
                category_columns=category_columns,
                test_split=str(cfg.target_test_split),
                min_harmful=int(cfg.category_breakdown.min_harmful),
            )


def logistic_score_bundle(
    *,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    source_score_x: torch.Tensor,
    validation_x: torch.Tensor,
    test_x: torch.Tensor,
    validation_n_harmful: int,
    test_n_harmful: int,
    seed: int,
    cfg: DictConfig,
) -> tuple[ScoreBundle, ScoreBundle, ScoreBundle]:
    scored_x = torch.cat([source_score_x, validation_x, test_x], dim=0)
    logits = logistic_logits(
        train_x,
        train_y,
        scored_x,
        seed=seed,
        l2=float(cfg.subspace_logistic.l2),
        lr=float(cfg.subspace_logistic.lr),
        epochs=int(cfg.subspace_logistic.epochs),
        standardize=bool(cfg.subspace_logistic.standardize),
        balanced_loss=bool(cfg.subspace_logistic.balanced_loss),
    )
    source_logits = logits[: source_score_x.shape[0]]
    validation_logits = logits[source_score_x.shape[0] : source_score_x.shape[0] + validation_x.shape[0]]
    test_logits = logits[source_score_x.shape[0] + validation_x.shape[0] :]
    source_scores = source_train_logit_scores(source_logits, train_y[: source_score_x.shape[0]])
    validation_scores = ScoreBundle(
        validation_logits[:validation_n_harmful],
        validation_logits[validation_n_harmful:],
    )
    test_scores = ScoreBundle(test_logits[:test_n_harmful], test_logits[test_n_harmful:])
    return source_scores, validation_scores, test_scores


def append_hrl_subspace_rows(
    rows: list[dict[str, object]],
    *,
    category_rows: list[dict[str, object]] | None = None,
    category_indices: CategoryIndices | None = None,
    category_columns: list[str] | None = None,
    source_cache: ActivationCache,
    target_cache: ActivationCache,
    meta: dict[str, object],
    source_langs: list[str],
    target_langs: list[str],
    tiers: dict[str, str],
    cfg: DictConfig,
) -> None:
    rank = int(cfg.rank)
    basis = subspace_basis(language_direction_matrix(source_cache, source_langs, str(cfg.source_train_split)), rank)
    source_x, source_y = source_pair(source_cache, source_langs, str(cfg.source_train_split))
    source_features = source_x @ basis

    eval_chunks: list[torch.Tensor] = []
    spec: list[tuple[str, str, int, int]] = []
    for target in target_langs:
        harmful, harmless = target_cache.pair(target, str(cfg.target_test_split))
        eval_chunks.extend([harmful, harmless])
        spec.append((str(cfg.target_test_split), target, harmful.shape[0], harmless.shape[0]))
    eval_features = torch.cat(eval_chunks, dim=0) @ basis
    seed = stable_seed(int(cfg.seed_offset), meta["model"], "subspace_zero_shot")
    logits = logistic_logits(
        source_features,
        source_y,
        torch.cat([source_features, eval_features], dim=0),
        seed=seed,
        l2=float(cfg.subspace_logistic.l2),
        lr=float(cfg.subspace_logistic.lr),
        epochs=int(cfg.subspace_logistic.epochs),
        standardize=bool(cfg.subspace_logistic.standardize),
        balanced_loss=bool(cfg.subspace_logistic.balanced_loss),
    )
    source_scores = source_train_logit_scores(logits[: source_features.shape[0]], source_y)
    split_scores = split_logits_by_split(logits[source_features.shape[0] :], spec)
    test_scores = split_scores[str(cfg.target_test_split)]
    threshold_scores = {"source_train": source_scores}
    method_meta = {
        **meta,
        "method": "ood_hrl_subspace_logistic",
        "classifier": "logistic_subspace",
        "gate": "hrl_subspace",
        "adaptation": "source_readout",
        "shot": 0,
        "budget": 0,
        "seed": "",
        "source_group": "hrl",
        "source_languages": ",".join(source_langs),
        "source_n": len(source_langs),
        "rank": rank,
    }
    append_ood_test_rows(
        rows,
        metadata=method_meta,
        target_tiers=tiers,
        test_scores=test_scores,
        threshold_scores=threshold_scores,
        threshold_source="source_train",
        threshold_scope=str(cfg.threshold_selection),
    )
    if category_rows is not None and category_indices is not None and category_columns is not None:
        append_indosafety_category_rows(
            category_rows,
            metadata=method_meta,
            target_tiers=tiers,
            validation_scores=test_scores,
            test_scores=test_scores,
            threshold_scores=threshold_scores,
            threshold_source="source_train",
            threshold_scope=str(cfg.threshold_selection),
            category_indices=category_indices,
            category_columns=category_columns,
            test_split=str(cfg.target_test_split),
            min_harmful=int(cfg.category_breakdown.min_harmful),
        )

    for seed_idx in range(int(cfg.seeds)):
        validation_scores = {}
        test_scores = {}
        threshold_scores = {}
        for target in target_langs:
            draw_seed = stable_seed(int(cfg.seed_offset), seed_idx, meta["model"], target, "subspace_one_shot")
            few_h, few_n, _, _ = sample_ood_pair(
                target_cache,
                target,
                train_split=str(cfg.target_train_split),
                test_split=str(cfg.target_test_split),
                budget=int(cfg.budget),
                seed=draw_seed,
            )
            target_x, target_y = stack_pairs([(few_h, few_n)])
            test_h, test_n = target_cache.pair(target, str(cfg.target_test_split))
            train_x = torch.cat([source_x, target_x], dim=0) @ basis
            train_y = torch.cat([source_y, target_y], dim=0)
            _, validation, test = logistic_score_bundle(
                train_x=train_x,
                train_y=train_y,
                source_score_x=source_features,
                validation_x=target_x @ basis,
                test_x=torch.cat([test_h, test_n], dim=0) @ basis,
                validation_n_harmful=few_h.shape[0],
                test_n_harmful=test_h.shape[0],
                seed=draw_seed,
                cfg=cfg,
            )
            validation_scores[target] = validation
            threshold_scores[target] = validation
            test_scores[target] = test
        method_meta = {
            **meta,
            "method": "ood_hrl_subspace_logistic",
            "classifier": "logistic_subspace",
            "gate": "hrl_subspace",
            "adaptation": "readout_refit",
            "shot": int(cfg.budget),
            "budget": int(cfg.budget),
            "seed": seed_idx,
            "source_group": "hrl",
            "source_languages": ",".join(source_langs),
            "source_n": len(source_langs),
            "rank": rank,
        }
        append_ood_test_rows(
            rows,
            metadata=method_meta,
            target_tiers=tiers,
            test_scores=test_scores,
            threshold_scores=threshold_scores,
            threshold_source="target_budget",
            threshold_scope=str(cfg.threshold_selection),
        )
        if category_rows is not None and category_indices is not None and category_columns is not None:
            append_indosafety_category_rows(
                category_rows,
                metadata=method_meta,
                target_tiers=tiers,
                validation_scores=validation_scores,
                test_scores=test_scores,
                threshold_scores=threshold_scores,
                threshold_source="target_budget",
                threshold_scope=str(cfg.threshold_selection),
                category_indices=category_indices,
                category_columns=category_columns,
                test_split=str(cfg.target_test_split),
                min_harmful=int(cfg.category_breakdown.min_harmful),
            )


def run_ood_latent_gate_transfer(cfg: DictConfig) -> None:
    source_langs_all = [str(language) for language in cfg.dataset.languages]
    source_tiers = {str(language): str(tier) for language, tier in cfg.dataset.resource_tier.items()}
    hrl_langs = source_languages("hrl", source_langs_all, source_tiers)
    target_langs = [str(language) for language in cfg.target_languages]
    tiers = target_tiers(target_langs)
    device_name = str(cfg.device)
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else device_name)
    all_rows: list[dict[str, object]] = []
    all_category_rows: list[dict[str, object]] = []
    category_enabled = bool(cfg.category_breakdown.enabled)
    category_columns = [str(column) for column in cfg.category_breakdown.columns]
    category_indices = (
        load_indosafety_category_indices(
            Path(cfg.category_breakdown.dataset_root),
            languages=target_langs,
            split=str(cfg.target_test_split),
            columns=category_columns,
        )
        if category_enabled
        else {}
    )

    for model_cfg in cfg.models:
        meta = model_meta(model_cfg)
        source_root = Path(
            str(cfg.source_activations_root_template).format(model=str(model_cfg.name))
        )
        target_root = Path(
            str(cfg.ood_activations_root_template).format(model=str(model_cfg.name))
        )
        source_cache = ActivationCache(source_root, int(model_cfg.layer), str(cfg.token_position), device)
        target_cache = ActivationCache(target_root, int(model_cfg.layer), str(cfg.token_position), device)
        rows: list[dict[str, object]] = []
        category_rows: list[dict[str, object]] = []
        if bool(cfg.gates.include_hrl_direction):
            append_hrl_direction_rows(
                rows,
                category_rows=category_rows if category_enabled else None,
                category_indices=category_indices if category_enabled else None,
                category_columns=category_columns if category_enabled else None,
                source_cache=source_cache,
                target_cache=target_cache,
                meta=meta,
                source_langs=hrl_langs,
                target_langs=target_langs,
                tiers=tiers,
                cfg=cfg,
            )
        if bool(cfg.gates.include_hrl_subspace):
            append_hrl_subspace_rows(
                rows,
                category_rows=category_rows if category_enabled else None,
                category_indices=category_indices if category_enabled else None,
                category_columns=category_columns if category_enabled else None,
                source_cache=source_cache,
                target_cache=target_cache,
                meta=meta,
                source_langs=hrl_langs,
                target_langs=target_langs,
                tiers=tiers,
                cfg=cfg,
            )
        out = Path(cfg.output_root) / str(model_cfg.name) / "ood_latent_gate_transfer.csv"
        write_rows(out, rows)
        if category_enabled:
            category_out = (
                Path(cfg.output_root)
                / str(model_cfg.name)
                / "ood_latent_gate_transfer_indosafety_categories.csv"
            )
            write_rows(category_out, category_rows)
            all_category_rows.extend(category_rows)
            print(f"[done] {model_cfg.short} categories: {len(category_rows)} rows")
        all_rows.extend(rows)
        print(f"[done] {model_cfg.short}: {len(rows)} rows")

    combined = Path(cfg.output_root) / "_combined" / "ood_latent_gate_transfer.csv"
    write_rows(combined, all_rows)
    if category_enabled:
        combined_categories = (
            Path(cfg.output_root) / "_combined" / "ood_latent_gate_transfer_indosafety_categories.csv"
        )
        write_rows(combined_categories, all_category_rows)
        print(f"[done] combined categories: {len(all_category_rows)} rows")
    print(f"[done] combined: {len(all_rows)} rows")


@hydra.main(version_base=None, config_path="../../configs", config_name="ood_transfer/compute_ood_latent_gate_transfer")
def main(cfg: DictConfig) -> None:
    run_ood_latent_gate_transfer(cfg)


if __name__ == "__main__":
    main()
