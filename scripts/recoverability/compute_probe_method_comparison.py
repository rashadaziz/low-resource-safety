"""Paper-facing comparison of harmfulness probe methods over stored activations."""


from pathlib import Path
from typing import Iterable

import hydra
import torch
from omegaconf import DictConfig

from multilingual_latent_safety.activation_store import ActivationCache
from multilingual_latent_safety.csv_io import write_rows
from multilingual_latent_safety.probe_evaluation import (
    ScoreBundle,
    append_method,
    contrast_direction,
    direction_scores,
    fit_mlp_probe,
    language_direction_matrix,
    logistic_logits,
    mlp_logits,
    pca_subspace_basis,
    pooled_contrast_direction,
    random_subspace_basis,
    residual_memory_subspace_basis,
    source_languages,
    source_pair,
    source_train_direction_scores,
    source_train_logit_scores,
    stack_pairs,
    subspace_basis,
    target_refit_subspace_basis,
    split_logits_by_split,
)
from multilingual_latent_safety.omniguard import (
    load_representations,
    read_selected_layer,
    representation_file,
    selected_layer_file,
)
from multilingual_latent_safety.probes import sample_balanced_indices
from multilingual_latent_safety.runtime import stable_seed


PREFERRED_FIELDS = [
    "model",
    "model_short",
    "layer",
    "token_position",
    "representation",
    "layer_selection",
    "method",
    "classifier",
    "source_group",
    "source_languages",
    "source_n",
    "rank",
    "base_rank",
    "memory_rank",
    "memory_languages",
    "budget",
    "seed",
    "threshold_scope",
    "threshold_source",
    "threshold",
    "threshold_selection_macro_f1_at_threshold",
    "validation_macro_f1_at_threshold",
    "split",
    "target_language",
    "target_tier",
    "evaluation_group",
    "indosafety_category_column",
    "indosafety_category",
    "harmful_scope",
    "harmless_scope",
    "auc",
    "average_precision",
    "precision",
    "recall",
    "f1",
    "accuracy",
    "balanced_accuracy",
    "tp",
    "fp",
    "tn",
    "fn",
    "n_harmful",
    "n_harmless",
]


class OmniguardRepresentationCache:
    def __init__(self, root: Path, layer: int, device: torch.device) -> None:
        self.root = root
        self.layer = int(layer)
        self.device = device
        self.cache: dict[tuple[str, str], tuple[torch.Tensor, torch.Tensor]] = {}

    def pair(self, language: str, split: str) -> tuple[torch.Tensor, torch.Tensor]:
        key = (language, split)
        if key not in self.cache:
            harmful = load_representations(self.root, language, split, "harmful", self.layer).to(self.device)
            harmless = load_representations(self.root, language, split, "harmless", self.layer).to(self.device)
            self.cache[key] = harmful, harmless
        return self.cache[key]


def sampled_pair(
    cache: ActivationCache,
    language: str,
    split: str,
    budget: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    harmful, harmless = cache.pair(language, split)
    h_idx, n_idx = sample_balanced_indices(harmful.shape[0], harmless.shape[0], int(budget), int(seed))
    return harmful[h_idx.to(harmful.device)], harmless[n_idx.to(harmless.device)]


def source_group_metadata(
    group_name: str,
    languages: list[str],
    tiers: dict[str, str],
    source_ns: list[int],
) -> dict[str, object]:
    return {
        "source_languages": "target_specific" if "loo" in group_name else ",".join(source_languages(group_name, languages, tiers)),
        "source_n": min(source_ns) if min(source_ns) == max(source_ns) else "target_specific",
    }


def direction_bank_features(x: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
    return x.to(torch.float32) @ directions.to(x.device, dtype=torch.float32).T


def projected_features(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x.to(torch.float32) @ basis.to(x.device, dtype=torch.float32)


def target_score_maps_from_directions(
    cache: ActivationCache,
    target_languages: list[str],
    directions: dict[str, torch.Tensor],
    *,
    val_split: str,
    test_split: str,
) -> tuple[dict[str, ScoreBundle], dict[str, ScoreBundle]]:
    val_scores: dict[str, ScoreBundle] = {}
    test_scores: dict[str, ScoreBundle] = {}
    for language in target_languages:
        val_scores[language] = direction_scores(directions[language], *cache.pair(language, val_split))
        test_scores[language] = direction_scores(directions[language], *cache.pair(language, test_split))
    return val_scores, test_scores


def eval_x_for_targets(
    cache: ActivationCache,
    target_languages: list[str],
    splits: Iterable[str],
) -> tuple[torch.Tensor, list[tuple[str, str, int, int]]]:
    chunks: list[torch.Tensor] = []
    spec: list[tuple[str, str, int, int]] = []
    for split in splits:
        for language in target_languages:
            harmful, harmless = cache.pair(language, split)
            chunks.extend([harmful, harmless])
            spec.append((split, language, harmful.shape[0], harmless.shape[0]))
    return torch.cat(chunks, dim=0), spec


def split_logits(
    logits: torch.Tensor,
    spec: list[tuple[str, str, int, int]],
) -> tuple[dict[str, ScoreBundle], dict[str, ScoreBundle]]:
    out = split_logits_by_split(logits, spec)
    return out["val"], out["test"]


def threshold_fit_scores(
    cfg: DictConfig,
    calibration_scores: dict[str, ScoreBundle] | None,
) -> tuple[dict[str, ScoreBundle] | None, str]:
    source = str(cfg.get("threshold_source", "validation"))
    if source == "validation":
        return None, "validation"
    if source == "target_train_budget":
        if calibration_scores is None:
            raise ValueError("threshold_source=target_train_budget requires budget calibration scores")
        return calibration_scores, source
    raise ValueError(f"unknown threshold_source {source!r}")


def budgeted_logistic_scores(
    *,
    cache: ActivationCache,
    target: str,
    source_y: torch.Tensor,
    source_features: torch.Tensor,
    transform,
    budget: int,
    seed: int,
    cfg: DictConfig,
) -> tuple[ScoreBundle, ScoreBundle, ScoreBundle]:
    few_h, few_n = sampled_pair(cache, target, str(cfg.train_split), budget, seed)
    target_x, target_y = stack_pairs([(few_h, few_n)])
    eval_x, spec = eval_x_for_targets(cache, [target], [str(cfg.val_split), str(cfg.test_split)])
    calibration_features = transform(target_x)
    eval_features = transform(eval_x)
    train_x = torch.cat([source_features, calibration_features], dim=0)
    train_y = torch.cat([source_y, target_y], dim=0)
    scored_x = torch.cat([calibration_features, eval_features], dim=0)
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
    calibration_logits = logits[: target_x.shape[0]]
    eval_logits = logits[target_x.shape[0] :]
    calibration_scores = ScoreBundle(
        calibration_logits[: few_h.shape[0]],
        calibration_logits[few_h.shape[0] :],
    )
    val_one, test_one = split_logits(eval_logits, spec)
    return calibration_scores, val_one[target], test_one[target]


def zero_shot_logistic_scores(
    *,
    cache: ActivationCache,
    target: str,
    source_y: torch.Tensor,
    source_features: torch.Tensor,
    transform,
    seed: int,
    cfg: DictConfig,
) -> tuple[ScoreBundle, ScoreBundle, ScoreBundle]:
    eval_x, spec = eval_x_for_targets(cache, [target], [str(cfg.val_split), str(cfg.test_split)])
    eval_features = transform(eval_x)
    scored_x = torch.cat([source_features, eval_features], dim=0)
    logits = logistic_logits(
        source_features,
        source_y,
        scored_x,
        seed=seed,
        l2=float(cfg.subspace_logistic.l2),
        lr=float(cfg.subspace_logistic.lr),
        epochs=int(cfg.subspace_logistic.epochs),
        standardize=bool(cfg.subspace_logistic.standardize),
        balanced_loss=bool(cfg.subspace_logistic.balanced_loss),
    )
    source_logits = logits[: source_features.shape[0]]
    eval_logits = logits[source_features.shape[0] :]
    val_one, test_one = split_logits(eval_logits, spec)
    return source_train_logit_scores(source_logits, source_y), val_one[target], test_one[target]


def append_direction_baselines(
    rows: list[dict[str, object]],
    *,
    cache: ActivationCache,
    model_meta: dict[str, object],
    languages: list[str],
    tiers: dict[str, str],
    target_languages: list[str],
    cfg: DictConfig,
) -> None:
    for group in list(cfg.direction_source_groups):
        directions: dict[str, torch.Tensor] = {}
        source_ns: list[int] = []
        for target in target_languages:
            src = source_languages(str(group), languages, tiers, target_language=target)
            source_ns.append(len(src))
            directions[target] = pooled_contrast_direction(cache, src, str(cfg.train_split))
        val_scores, test_scores = target_score_maps_from_directions(
            cache,
            target_languages,
            directions,
            val_split=str(cfg.val_split),
            test_split=str(cfg.test_split),
        )
        append_method(
            rows,
            metadata={
                **model_meta,
                "method": "source_direction",
                "classifier": "mean_difference",
                "source_group": str(group),
                "source_languages": "target_specific" if "loo" in str(group) else ",".join(source_languages(str(group), languages, tiers)),
                "source_n": min(source_ns) if min(source_ns) == max(source_ns) else "target_specific",
                "rank": "",
                "budget": 0,
                "seed": "",
            },
            target_tiers=tiers,
            validation_scores=val_scores,
            test_scores=test_scores,
            include_oracle_threshold=bool(cfg.include_oracle_thresholds),
            threshold_scope=str(cfg.threshold_selection),
        )


def append_target_direction_methods(
    rows: list[dict[str, object]],
    *,
    cache: ActivationCache,
    model_meta: dict[str, object],
    target_languages: list[str],
    tiers: dict[str, str],
    cfg: DictConfig,
) -> None:
    full_directions = {
        language: contrast_direction(*cache.pair(language, str(cfg.train_split))) for language in target_languages
    }
    val_scores, test_scores = target_score_maps_from_directions(
        cache,
        target_languages,
        full_directions,
        val_split=str(cfg.val_split),
        test_split=str(cfg.test_split),
    )
    append_method(
        rows,
        metadata={
            **model_meta,
            "method": "target_full_oracle_direction",
            "classifier": "mean_difference",
            "source_group": "target",
            "source_languages": "target_train",
            "source_n": 1,
            "rank": "",
            "budget": "full",
            "seed": "",
        },
        target_tiers=tiers,
        validation_scores=val_scores,
        test_scores=test_scores,
        include_oracle_threshold=bool(cfg.include_oracle_thresholds),
        threshold_scope=str(cfg.threshold_selection),
    )

    for budget in list(cfg.budgets):
        budget = int(budget)
        if budget <= 0:
            continue
        for seed in range(int(cfg.seeds)):
            directions: dict[str, torch.Tensor] = {}
            calibration_scores: dict[str, ScoreBundle] = {}
            for language in target_languages:
                draw_seed = stable_seed(int(cfg.seed_offset), seed, model_meta["model"], language, budget, "target")
                few_h, few_n = sampled_pair(cache, language, str(cfg.train_split), budget, draw_seed)
                directions[language] = contrast_direction(
                    few_h,
                    few_n,
                )
                calibration_scores[language] = direction_scores(directions[language], few_h, few_n)
            val_scores, test_scores = target_score_maps_from_directions(
                cache,
                target_languages,
                directions,
                val_split=str(cfg.val_split),
                test_split=str(cfg.test_split),
            )
            threshold_scores, threshold_source = threshold_fit_scores(cfg, calibration_scores)
            append_method(
                rows,
                metadata={
                    **model_meta,
                    "method": "target_only_direction",
                    "classifier": "mean_difference",
                    "source_group": "target",
                    "source_languages": "target_fewshot",
                    "source_n": 1,
                    "rank": "",
                    "budget": budget,
                    "seed": seed,
                },
                target_tiers=tiers,
                validation_scores=val_scores,
                test_scores=test_scores,
                threshold_scores=threshold_scores,
                threshold_source=threshold_source,
                include_oracle_threshold=bool(cfg.include_oracle_thresholds),
                threshold_scope=str(cfg.threshold_selection),
            )


def append_budgeted_source_direction_baselines(
    rows: list[dict[str, object]],
    *,
    cache: ActivationCache,
    model_meta: dict[str, object],
    languages: list[str],
    tiers: dict[str, str],
    target_languages: list[str],
    cfg: DictConfig,
) -> None:
    baseline_cfg = cfg.data_matched_baselines
    include_source_direction = bool(baseline_cfg.get("include_source_direction_threshold", False))
    include_random_direction = bool(baseline_cfg.get("include_random_direction_threshold", False))
    if not include_source_direction and not include_random_direction:
        return
    for group in list(baseline_cfg.source_groups):
        group_name = str(group)
        source_ns: list[int] = []
        for target in target_languages:
            src = source_languages(group_name, languages, tiers, target_language=target)
            source_ns.append(len(src))
        if include_source_direction:
            directions: dict[str, torch.Tensor] = {}
            for target in target_languages:
                src = source_languages(group_name, languages, tiers, target_language=target)
                directions[target] = pooled_contrast_direction(cache, src, str(cfg.train_split))
            val_scores, test_scores = target_score_maps_from_directions(
                cache,
                target_languages,
                directions,
                val_split=str(cfg.val_split),
                test_split=str(cfg.test_split),
            )
            if bool(baseline_cfg.include_zero_shot):
                calibration_scores = {}
                for target in target_languages:
                    src = source_languages(group_name, languages, tiers, target_language=target)
                    calibration_scores[f"source_train::{target}"] = source_train_direction_scores(
                        cache,
                        src,
                        str(cfg.train_split),
                        directions[target],
                    )
                append_method(
                    rows,
                    metadata={
                        **model_meta,
                        "method": "source_direction_budget_threshold",
                        "classifier": "mean_difference",
                        "source_group": group_name,
                        **source_group_metadata(group_name, languages, tiers, source_ns),
                        "rank": "",
                        "budget": 0,
                        "seed": "",
                    },
                    target_tiers=tiers,
                    validation_scores=val_scores,
                    test_scores=test_scores,
                    threshold_scores=threshold_scores,
                    threshold_source="source_train",
                    include_oracle_threshold=False,
                    threshold_scope=str(cfg.threshold_selection),
                )
            for budget in list(cfg.budgets):
                budget = int(budget)
                if budget <= 0:
                    continue
                for seed in range(int(cfg.seeds)):
                    calibration_scores: dict[str, ScoreBundle] = {}
                    for target in target_languages:
                        draw_seed = stable_seed(
                            int(cfg.seed_offset),
                            seed,
                            model_meta["model"],
                            target,
                            budget,
                            group_name,
                            "source_direction_budget_threshold",
                        )
                        few_h, few_n = sampled_pair(cache, target, str(cfg.train_split), budget, draw_seed)
                        calibration_scores[target] = direction_scores(directions[target], few_h, few_n)
                    threshold_scores, threshold_source = threshold_fit_scores(cfg, calibration_scores)
                    append_method(
                        rows,
                        metadata={
                            **model_meta,
                            "method": "source_direction_budget_threshold",
                            "classifier": "mean_difference",
                            "source_group": group_name,
                            **source_group_metadata(group_name, languages, tiers, source_ns),
                            "rank": "",
                            "budget": budget,
                            "seed": seed,
                        },
                        target_tiers=tiers,
                        validation_scores=val_scores,
                        test_scores=test_scores,
                        threshold_scores=threshold_scores,
                        threshold_source=threshold_source,
                        include_oracle_threshold=bool(cfg.include_oracle_thresholds),
                        threshold_scope=str(cfg.threshold_selection),
                    )
        if include_random_direction:
            source_x, _ = source_pair(
                cache,
                source_languages(group_name, languages, tiers, target_language=target_languages[0]),
                str(cfg.train_split),
            )

            def random_direction(seed: int) -> torch.Tensor:
                return random_subspace_basis(source_x.shape[1], 1, seed=seed, device=cache.device)[:, 0]

            if bool(baseline_cfg.include_zero_shot):
                direction = random_direction(
                    stable_seed(
                        int(cfg.seed_offset),
                        model_meta["model"],
                        group_name,
                        "random_direction_zero_shot",
                    )
                )
                directions = {target: direction for target in target_languages}
                val_scores, test_scores = target_score_maps_from_directions(
                    cache,
                    target_languages,
                    directions,
                    val_split=str(cfg.val_split),
                    test_split=str(cfg.test_split),
                )
                calibration_scores = {}
                for target in target_languages:
                    src = source_languages(group_name, languages, tiers, target_language=target)
                    calibration_scores[f"source_train::{target}"] = source_train_direction_scores(
                        cache,
                        src,
                        str(cfg.train_split),
                        direction,
                    )
                append_method(
                    rows,
                    metadata={
                        **model_meta,
                        "method": "random_direction",
                        "classifier": "random_direction",
                        "source_group": group_name,
                        **source_group_metadata(group_name, languages, tiers, source_ns),
                        "rank": "",
                        "budget": 0,
                        "seed": "",
                    },
                    target_tiers=tiers,
                    validation_scores=val_scores,
                    test_scores=test_scores,
                    threshold_scores=calibration_scores,
                    threshold_source="source_train",
                    include_oracle_threshold=False,
                    threshold_scope=str(cfg.threshold_selection),
                )
            for budget in list(cfg.budgets):
                budget = int(budget)
                if budget <= 0:
                    continue
                for seed in range(int(cfg.seeds)):
                    direction = random_direction(
                        stable_seed(
                            int(cfg.seed_offset),
                            seed,
                            model_meta["model"],
                            group_name,
                            "random_direction",
                        )
                    )
                    directions = {target: direction for target in target_languages}
                    val_scores, test_scores = target_score_maps_from_directions(
                        cache,
                        target_languages,
                        directions,
                        val_split=str(cfg.val_split),
                        test_split=str(cfg.test_split),
                    )
                    calibration_scores = {}
                    for target in target_languages:
                        draw_seed = stable_seed(
                            int(cfg.seed_offset),
                            seed,
                            model_meta["model"],
                            target,
                            budget,
                            group_name,
                            "random_direction",
                        )
                        few_h, few_n = sampled_pair(cache, target, str(cfg.train_split), budget, draw_seed)
                        calibration_scores[target] = direction_scores(direction, few_h, few_n)
                    threshold_scores, threshold_source = threshold_fit_scores(cfg, calibration_scores)
                    append_method(
                        rows,
                        metadata={
                            **model_meta,
                            "method": "random_direction",
                            "classifier": "random_direction",
                            "source_group": group_name,
                            **source_group_metadata(group_name, languages, tiers, source_ns),
                            "rank": "",
                            "budget": budget,
                            "seed": seed,
                        },
                        target_tiers=tiers,
                        validation_scores=val_scores,
                        test_scores=test_scores,
                        threshold_scores=threshold_scores,
                        threshold_source=threshold_source,
                        include_oracle_threshold=bool(cfg.include_oracle_thresholds),
                        threshold_scope=str(cfg.threshold_selection),
                    )


def append_data_matched_logistic_baselines(
    rows: list[dict[str, object]],
    *,
    cache: ActivationCache,
    model_meta: dict[str, object],
    languages: list[str],
    tiers: dict[str, str],
    target_languages: list[str],
    cfg: DictConfig,
) -> None:
    baseline_cfg = cfg.data_matched_baselines
    source_cache: dict[tuple[str, ...], tuple[torch.Tensor, torch.Tensor]] = {}
    direction_cache: dict[tuple[str, ...], torch.Tensor] = {}
    pca_cache: dict[tuple[tuple[str, ...], int], torch.Tensor] = {}
    random_cache: dict[tuple[int, int], torch.Tensor] = {}

    def cached_source(src: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        key = tuple(src)
        if key not in source_cache:
            source_cache[key] = source_pair(cache, src, str(cfg.train_split))
        return source_cache[key]

    def cached_directions(src: list[str]) -> torch.Tensor:
        key = tuple(src)
        if key not in direction_cache:
            direction_cache[key] = language_direction_matrix(cache, src, str(cfg.train_split))
        return direction_cache[key]

    def cached_pca(src: list[str], rank: int) -> torch.Tensor:
        key = (tuple(src), rank)
        if key not in pca_cache:
            source_x, _ = cached_source(src)
            pca_cache[key] = pca_subspace_basis(source_x, rank)
        return pca_cache[key]

    def cached_random(dim: int, rank: int, seed: int) -> torch.Tensor:
        key = (rank, seed)
        if key not in random_cache:
            random_cache[key] = random_subspace_basis(dim, rank, seed=seed, device=cache.device)
        return random_cache[key]

    def append_zero_shot_logistic_method(method: str, classifier: str, group_name: str, rank: int | str) -> None:
        val_scores: dict[str, ScoreBundle] = {}
        test_scores: dict[str, ScoreBundle] = {}
        calibration_scores: dict[str, ScoreBundle] = {}
        source_ns: list[int] = []
        for target in target_languages:
            src = source_languages(group_name, languages, tiers, target_language=target)
            source_ns.append(len(src))
            source_x, source_y = cached_source(src)
            if method == "source_direction_bank_logistic":
                directions = cached_directions(src)
                source_features = direction_bank_features(source_x, directions)
                transform = lambda x, directions=directions: direction_bank_features(x, directions)
            elif method == "full_space_logistic":
                source_features = source_x.to(torch.float32)
                transform = lambda x: x.to(torch.float32)
            elif method == "random_subspace_logistic":
                rank_int = int(rank)
                basis_seed = stable_seed(
                    int(cfg.seed_offset),
                    model_meta["model"],
                    group_name,
                    rank_int,
                    "random_subspace_basis_zero_shot",
                )
                basis = cached_random(source_x.shape[1], rank_int, basis_seed)
                source_features = projected_features(source_x, basis)
                transform = lambda x, basis=basis: projected_features(x, basis)
            elif method == "pca_subspace_logistic":
                rank_int = int(rank)
                if rank_int > min(source_x.shape):
                    continue
                basis = cached_pca(src, rank_int)
                source_features = projected_features(source_x, basis)
                transform = lambda x, basis=basis: projected_features(x, basis)
            else:
                raise ValueError(f"unknown zero-shot logistic method {method!r}")
            train_seed = stable_seed(
                int(cfg.seed_offset),
                model_meta["model"],
                target,
                group_name,
                rank,
                method,
                "zero_shot",
            )
            calibration, val_one, test_one = zero_shot_logistic_scores(
                cache=cache,
                target=target,
                source_y=source_y,
                source_features=source_features,
                transform=transform,
                seed=train_seed,
                cfg=cfg,
            )
            calibration_scores[f"source_train::{target}"] = calibration
            val_scores[target] = val_one
            test_scores[target] = test_one
        if set(val_scores) != set(target_languages):
            return
        append_method(
            rows,
            metadata={
                **model_meta,
                "method": method,
                "classifier": classifier,
                "source_group": group_name,
                **source_group_metadata(group_name, languages, tiers, source_ns),
                "rank": rank,
                "budget": 0,
                "seed": "",
            },
            target_tiers=tiers,
            validation_scores=val_scores,
            test_scores=test_scores,
            threshold_scores=calibration_scores,
            threshold_source="source_train",
            include_oracle_threshold=False,
            threshold_scope=str(cfg.threshold_selection),
        )

    for group in list(baseline_cfg.source_groups):
        group_name = str(group)
        if bool(baseline_cfg.include_zero_shot):
            if bool(baseline_cfg.include_direction_bank_logistic):
                append_zero_shot_logistic_method(
                    "source_direction_bank_logistic",
                    "logistic_direction_bank",
                    group_name,
                    "",
                )
            if bool(baseline_cfg.include_full_space_logistic):
                append_zero_shot_logistic_method("full_space_logistic", "logistic", group_name, "")
            for rank in list(cfg.ranks):
                rank = int(rank)
                if bool(baseline_cfg.include_random_subspace_logistic):
                    append_zero_shot_logistic_method(
                        "random_subspace_logistic",
                        "logistic_random_subspace",
                        group_name,
                        rank,
                    )
                if bool(baseline_cfg.include_unsupervised_pca_logistic):
                    append_zero_shot_logistic_method(
                        "pca_subspace_logistic",
                        "logistic_unsupervised_pca",
                        group_name,
                        rank,
                    )
        for budget in list(cfg.budgets):
            budget = int(budget)
            if budget <= 0:
                continue
            for seed in range(int(cfg.seeds)):
                if bool(baseline_cfg.include_direction_bank_logistic):
                    val_scores: dict[str, ScoreBundle] = {}
                    test_scores: dict[str, ScoreBundle] = {}
                    calibration_scores: dict[str, ScoreBundle] = {}
                    source_ns: list[int] = []
                    for target in target_languages:
                        src = source_languages(group_name, languages, tiers, target_language=target)
                        source_ns.append(len(src))
                        source_x, source_y = cached_source(src)
                        directions = cached_directions(src)
                        source_features = direction_bank_features(source_x, directions)
                        draw_seed = stable_seed(
                            int(cfg.seed_offset),
                            seed,
                            model_meta["model"],
                            target,
                            budget,
                            group_name,
                            "direction_bank_logistic",
                        )
                        calibration, val_one, test_one = budgeted_logistic_scores(
                            cache=cache,
                            target=target,
                            source_y=source_y,
                            source_features=source_features,
                            transform=lambda x, directions=directions: direction_bank_features(x, directions),
                            budget=budget,
                            seed=draw_seed,
                            cfg=cfg,
                        )
                        calibration_scores[target] = calibration
                        val_scores[target] = val_one
                        test_scores[target] = test_one
                    threshold_scores, threshold_source = threshold_fit_scores(cfg, calibration_scores)
                    append_method(
                        rows,
                        metadata={
                            **model_meta,
                            "method": "source_direction_bank_logistic",
                            "classifier": "logistic_direction_bank",
                            "source_group": group_name,
                            **source_group_metadata(group_name, languages, tiers, source_ns),
                            "rank": "",
                            "budget": budget,
                            "seed": seed,
                        },
                        target_tiers=tiers,
                        validation_scores=val_scores,
                        test_scores=test_scores,
                        threshold_scores=threshold_scores,
                        threshold_source=threshold_source,
                        include_oracle_threshold=bool(cfg.include_oracle_thresholds),
                        threshold_scope=str(cfg.threshold_selection),
                    )

                if bool(baseline_cfg.include_full_space_logistic):
                    val_scores = {}
                    test_scores = {}
                    calibration_scores = {}
                    source_ns = []
                    for target in target_languages:
                        src = source_languages(group_name, languages, tiers, target_language=target)
                        source_ns.append(len(src))
                        source_x, source_y = cached_source(src)
                        draw_seed = stable_seed(
                            int(cfg.seed_offset),
                            seed,
                            model_meta["model"],
                            target,
                            budget,
                            group_name,
                            "full_space_logistic",
                        )
                        calibration, val_one, test_one = budgeted_logistic_scores(
                            cache=cache,
                            target=target,
                            source_y=source_y,
                            source_features=source_x.to(torch.float32),
                            transform=lambda x: x.to(torch.float32),
                            budget=budget,
                            seed=draw_seed,
                            cfg=cfg,
                        )
                        calibration_scores[target] = calibration
                        val_scores[target] = val_one
                        test_scores[target] = test_one
                    threshold_scores, threshold_source = threshold_fit_scores(cfg, calibration_scores)
                    append_method(
                        rows,
                        metadata={
                            **model_meta,
                            "method": "full_space_logistic",
                            "classifier": "logistic",
                            "source_group": group_name,
                            **source_group_metadata(group_name, languages, tiers, source_ns),
                            "rank": "",
                            "budget": budget,
                            "seed": seed,
                        },
                        target_tiers=tiers,
                        validation_scores=val_scores,
                        test_scores=test_scores,
                        threshold_scores=threshold_scores,
                        threshold_source=threshold_source,
                        include_oracle_threshold=bool(cfg.include_oracle_thresholds),
                        threshold_scope=str(cfg.threshold_selection),
                    )

                for rank in list(cfg.ranks):
                    rank = int(rank)
                    if bool(baseline_cfg.include_random_subspace_logistic):
                        val_scores = {}
                        test_scores = {}
                        calibration_scores = {}
                        source_ns = []
                        for target in target_languages:
                            src = source_languages(group_name, languages, tiers, target_language=target)
                            source_ns.append(len(src))
                            source_x, source_y = cached_source(src)
                            basis_seed = stable_seed(
                                int(cfg.seed_offset),
                                seed,
                                model_meta["model"],
                                group_name,
                                rank,
                                "random_subspace_basis",
                            )
                            basis = cached_random(source_x.shape[1], rank, basis_seed)
                            source_features = projected_features(source_x, basis)
                            draw_seed = stable_seed(
                                int(cfg.seed_offset),
                                seed,
                                model_meta["model"],
                                target,
                                budget,
                                group_name,
                                rank,
                                "random_subspace_logistic",
                            )
                            calibration, val_one, test_one = budgeted_logistic_scores(
                                cache=cache,
                                target=target,
                                source_y=source_y,
                                source_features=source_features,
                                transform=lambda x, basis=basis: projected_features(x, basis),
                                budget=budget,
                                seed=draw_seed,
                                cfg=cfg,
                            )
                            calibration_scores[target] = calibration
                            val_scores[target] = val_one
                            test_scores[target] = test_one
                        threshold_scores, threshold_source = threshold_fit_scores(cfg, calibration_scores)
                        append_method(
                            rows,
                            metadata={
                                **model_meta,
                                "method": "random_subspace_logistic",
                                "classifier": "logistic_random_subspace",
                                "source_group": group_name,
                                **source_group_metadata(group_name, languages, tiers, source_ns),
                                "rank": rank,
                                "budget": budget,
                                "seed": seed,
                            },
                            target_tiers=tiers,
                            validation_scores=val_scores,
                            test_scores=test_scores,
                            threshold_scores=threshold_scores,
                            threshold_source=threshold_source,
                            include_oracle_threshold=bool(cfg.include_oracle_thresholds),
                            threshold_scope=str(cfg.threshold_selection),
                        )
                    if bool(baseline_cfg.include_unsupervised_pca_logistic):
                        val_scores = {}
                        test_scores = {}
                        calibration_scores = {}
                        source_ns = []
                        for target in target_languages:
                            src = source_languages(group_name, languages, tiers, target_language=target)
                            source_ns.append(len(src))
                            source_x, source_y = cached_source(src)
                            if rank > min(source_x.shape):
                                continue
                            basis = cached_pca(src, rank)
                            source_features = projected_features(source_x, basis)
                            draw_seed = stable_seed(
                                int(cfg.seed_offset),
                                seed,
                                model_meta["model"],
                                target,
                                budget,
                                group_name,
                                rank,
                                "pca_subspace_logistic",
                            )
                            calibration, val_one, test_one = budgeted_logistic_scores(
                                cache=cache,
                                target=target,
                                source_y=source_y,
                                source_features=source_features,
                                transform=lambda x, basis=basis: projected_features(x, basis),
                                budget=budget,
                                seed=draw_seed,
                                cfg=cfg,
                            )
                            calibration_scores[target] = calibration
                            val_scores[target] = val_one
                            test_scores[target] = test_one
                        if set(val_scores) != set(target_languages):
                            continue
                        threshold_scores, threshold_source = threshold_fit_scores(cfg, calibration_scores)
                        append_method(
                            rows,
                            metadata={
                                **model_meta,
                                "method": "pca_subspace_logistic",
                                "classifier": "logistic_unsupervised_pca",
                                "source_group": group_name,
                                **source_group_metadata(group_name, languages, tiers, source_ns),
                                "rank": rank,
                                "budget": budget,
                                "seed": seed,
                            },
                            target_tiers=tiers,
                            validation_scores=val_scores,
                            test_scores=test_scores,
                            threshold_scores=threshold_scores,
                            threshold_source=threshold_source,
                            include_oracle_threshold=bool(cfg.include_oracle_thresholds),
                            threshold_scope=str(cfg.threshold_selection),
                        )


def append_data_matched_baselines(
    rows: list[dict[str, object]],
    *,
    cache: ActivationCache,
    model_meta: dict[str, object],
    languages: list[str],
    tiers: dict[str, str],
    target_languages: list[str],
    cfg: DictConfig,
) -> None:
    if not bool(cfg.data_matched_baselines.enabled):
        return
    append_budgeted_source_direction_baselines(
        rows,
        cache=cache,
        model_meta=model_meta,
        languages=languages,
        tiers=tiers,
        target_languages=target_languages,
        cfg=cfg,
    )
    append_data_matched_logistic_baselines(
        rows,
        cache=cache,
        model_meta=model_meta,
        languages=languages,
        tiers=tiers,
        target_languages=target_languages,
        cfg=cfg,
    )


def append_subspace_methods(
    rows: list[dict[str, object]],
    *,
    cache: ActivationCache,
    model_meta: dict[str, object],
    languages: list[str],
    tiers: dict[str, str],
    target_languages: list[str],
    cfg: DictConfig,
) -> None:
    basis_cache: dict[tuple[str, int, str], tuple[torch.Tensor, int] | None] = {}

    def cached_basis(group_name: str, rank: int, target: str) -> tuple[torch.Tensor, int] | None:
        key = (group_name, rank, target)
        if key not in basis_cache:
            src = source_languages(group_name, languages, tiers, target_language=target)
            if not src or rank > len(src):
                basis_cache[key] = None
            else:
                basis_cache[key] = (
                    subspace_basis(language_direction_matrix(cache, src, str(cfg.train_split)), rank),
                    len(src),
                )
        return basis_cache[key]

    for group in list(cfg.subspace_source_groups):
        group_name = str(group)
        for rank in list(cfg.ranks):
            rank = int(rank)
            for budget in list(cfg.budgets):
                budget = int(budget)
                if budget <= 0:
                    continue
                for seed in range(int(cfg.seeds)):
                    directions: dict[str, torch.Tensor] = {}
                    calibration_scores: dict[str, ScoreBundle] = {}
                    source_ns: list[int] = []
                    for target in target_languages:
                        cached = cached_basis(group_name, rank, target)
                        if cached is None:
                            continue
                        basis, source_n = cached
                        source_ns.append(source_n)
                        draw_seed = stable_seed(
                            int(cfg.seed_offset), seed, model_meta["model"], target, budget, group_name, rank, "subspace"
                        )
                        few_h, few_n = sampled_pair(cache, target, str(cfg.train_split), budget, draw_seed)
                        directions[target] = contrast_direction(few_h, few_n, basis=basis)
                        calibration_scores[target] = direction_scores(directions[target], few_h, few_n)
                    if set(directions) != set(target_languages):
                        continue
                    val_scores, test_scores = target_score_maps_from_directions(
                        cache,
                        target_languages,
                        directions,
                        val_split=str(cfg.val_split),
                        test_split=str(cfg.test_split),
                    )
                    threshold_scores, threshold_source = threshold_fit_scores(cfg, calibration_scores)
                    append_method(
                        rows,
                        metadata={
                            **model_meta,
                            "method": "subspace_constrained_direction",
                            "classifier": "projected_mean_difference",
                            "source_group": group_name,
                            "source_languages": "target_specific" if "loo" in group_name else ",".join(source_languages(group_name, languages, tiers)),
                            "source_n": min(source_ns) if min(source_ns) == max(source_ns) else "target_specific",
                            "rank": rank,
                            "budget": budget,
                            "seed": seed,
                        },
                        target_tiers=tiers,
                        validation_scores=val_scores,
                        test_scores=test_scores,
                        threshold_scores=threshold_scores,
                        threshold_source=threshold_source,
                        include_oracle_threshold=bool(cfg.include_oracle_thresholds),
                        threshold_scope=str(cfg.threshold_selection),
                    )


def append_subspace_logistic_methods(
    rows: list[dict[str, object]],
    *,
    cache: ActivationCache,
    model_meta: dict[str, object],
    languages: list[str],
    tiers: dict[str, str],
    target_languages: list[str],
    cfg: DictConfig,
) -> None:
    if not bool(cfg.subspace_logistic.enabled):
        return
    direction_cache: dict[tuple[str, str], tuple[torch.Tensor, int] | None] = {}
    basis_cache: dict[tuple[str, int, str], tuple[torch.Tensor, int] | None] = {}
    source_cache: dict[tuple[str, str], tuple[torch.Tensor, torch.Tensor] | None] = {}
    eval_cache: dict[str, tuple[torch.Tensor, dict[tuple[str, str], tuple[int, int]]]] = {}

    def cached_source_directions(group_name: str, target: str) -> tuple[torch.Tensor, int] | None:
        key = (group_name, target)
        if key not in direction_cache:
            src = source_languages(group_name, languages, tiers, target_language=target)
            if not src:
                direction_cache[key] = None
            else:
                direction_cache[key] = language_direction_matrix(cache, src, str(cfg.train_split)), len(src)
        return direction_cache[key]

    def cached_basis(group_name: str, rank: int, target: str) -> tuple[torch.Tensor, int] | None:
        key = (group_name, rank, target)
        if key not in basis_cache:
            source_directions = cached_source_directions(group_name, target)
            if source_directions is None:
                basis_cache[key] = None
            else:
                directions, source_n = source_directions
                basis_cache[key] = None if rank > directions.shape[0] else (subspace_basis(directions, rank), source_n)
        return basis_cache[key]

    def cached_source(group_name: str, target: str) -> tuple[torch.Tensor, torch.Tensor] | None:
        key = (group_name, target)
        if key not in source_cache:
            src = source_languages(group_name, languages, tiers, target_language=target)
            if not src:
                source_cache[key] = None
            else:
                source_cache[key] = stack_pairs(cache.pair(language, str(cfg.train_split)) for language in src)
        return source_cache[key]

    def cached_eval(target: str) -> tuple[torch.Tensor, dict[tuple[str, str], tuple[int, int]]]:
        if target not in eval_cache:
            eval_cache[target] = eval_x_for_targets(cache, [target], [str(cfg.val_split), str(cfg.test_split)])
        return eval_cache[target]

    for group in list(cfg.subspace_source_groups):
        group_name = str(group)
        for rank in list(cfg.ranks):
            rank = int(rank)
            if bool(cfg.subspace_logistic.get("include_zero_shot", False)):
                val_scores: dict[str, ScoreBundle] = {}
                test_scores: dict[str, ScoreBundle] = {}
                calibration_scores: dict[str, ScoreBundle] = {}
                source_ns: list[int] = []
                for target in target_languages:
                    cached = cached_basis(group_name, rank, target)
                    source_pair = cached_source(group_name, target)
                    if cached is None or source_pair is None:
                        continue
                    basis, source_n = cached
                    source_x, source_y = source_pair
                    source_ns.append(source_n)
                    source_features = projected_features(source_x, basis)
                    train_seed = stable_seed(
                        int(cfg.seed_offset),
                        model_meta["model"],
                        target,
                        group_name,
                        rank,
                        "subspace_logistic",
                        "zero_shot",
                    )
                    calibration, val_one, test_one = zero_shot_logistic_scores(
                        cache=cache,
                        target=target,
                        source_y=source_y,
                        source_features=source_features,
                        transform=lambda x, basis=basis: projected_features(x, basis),
                        seed=train_seed,
                        cfg=cfg,
                    )
                    calibration_scores[f"source_train::{target}"] = calibration
                    val_scores[target] = val_one
                    test_scores[target] = test_one
                if set(val_scores) == set(target_languages):
                    append_method(
                        rows,
                        metadata={
                            **model_meta,
                            "method": "subspace_logistic",
                            "classifier": "logistic",
                            "source_group": group_name,
                            "source_languages": "target_specific"
                            if "loo" in group_name
                            else ",".join(source_languages(group_name, languages, tiers)),
                            "source_n": min(source_ns) if min(source_ns) == max(source_ns) else "target_specific",
                            "rank": rank,
                            "budget": 0,
                            "seed": "",
                        },
                        target_tiers=tiers,
                        validation_scores=val_scores,
                        test_scores=test_scores,
                        threshold_scores=calibration_scores,
                        threshold_source="source_train",
                        include_oracle_threshold=False,
                        threshold_scope=str(cfg.threshold_selection),
                    )
            for budget in list(cfg.budgets):
                budget = int(budget)
                if budget <= 0:
                    continue
                for seed in range(int(cfg.seeds)):
                    val_scores: dict[str, ScoreBundle] = {}
                    test_scores: dict[str, ScoreBundle] = {}
                    calibration_scores: dict[str, ScoreBundle] = {}
                    source_ns: list[int] = []
                    for target in target_languages:
                        cached = cached_basis(group_name, rank, target)
                        source_pair = cached_source(group_name, target)
                        if cached is None or source_pair is None:
                            continue
                        basis, source_n = cached
                        source_x, source_y = source_pair
                        source_ns.append(source_n)
                        draw_seed = stable_seed(
                            int(cfg.seed_offset), seed, model_meta["model"], target, budget, group_name, rank, "subspace_logistic"
                        )
                        few_h, few_n = sampled_pair(cache, target, str(cfg.train_split), budget, draw_seed)
                        target_x, target_y = stack_pairs([(few_h, few_n)])
                        train_x = torch.cat([source_x, target_x], dim=0) @ basis
                        train_y = torch.cat([source_y, target_y], dim=0)
                        eval_x, spec = cached_eval(target)
                        calibration_eval_x = target_x @ basis
                        scored_x = torch.cat([calibration_eval_x, eval_x @ basis], dim=0)
                        logits = logistic_logits(
                            train_x,
                            train_y,
                            scored_x,
                            seed=draw_seed,
                            l2=float(cfg.subspace_logistic.l2),
                            lr=float(cfg.subspace_logistic.lr),
                            epochs=int(cfg.subspace_logistic.epochs),
                            standardize=bool(cfg.subspace_logistic.standardize),
                            balanced_loss=bool(cfg.subspace_logistic.balanced_loss),
                        )
                        calibration_logits = logits[: target_x.shape[0]]
                        eval_logits = logits[target_x.shape[0] :]
                        calibration_scores[target] = ScoreBundle(
                            calibration_logits[: few_h.shape[0]],
                            calibration_logits[few_h.shape[0] :],
                        )
                        val_one, test_one = split_logits(eval_logits, spec)
                        val_scores[target] = val_one[target]
                        test_scores[target] = test_one[target]
                    if set(val_scores) != set(target_languages):
                        continue
                    threshold_scores, threshold_source = threshold_fit_scores(cfg, calibration_scores)
                    append_method(
                        rows,
                        metadata={
                            **model_meta,
                            "method": "subspace_logistic",
                            "classifier": "logistic",
                            "source_group": group_name,
                            "source_languages": "target_specific" if "loo" in group_name else ",".join(source_languages(group_name, languages, tiers)),
                            "source_n": min(source_ns) if min(source_ns) == max(source_ns) else "target_specific",
                            "rank": rank,
                            "budget": budget,
                            "seed": seed,
                        },
                        target_tiers=tiers,
                        validation_scores=val_scores,
                        test_scores=test_scores,
                        threshold_scores=threshold_scores,
                        threshold_source=threshold_source,
                        include_oracle_threshold=bool(cfg.include_oracle_thresholds),
                        threshold_scope=str(cfg.threshold_selection),
                    )
                    if bool(cfg.subspace_logistic.get("include_refit_target_direction", False)):
                        val_scores = {}
                        test_scores = {}
                        calibration_scores = {}
                        source_ns = []
                        for target in target_languages:
                            source_directions = cached_source_directions(group_name, target)
                            source_pair = cached_source(group_name, target)
                            if source_directions is None or source_pair is None:
                                continue
                            directions, source_n = source_directions
                            if rank > min(directions.shape[0] + 1, directions.shape[1]):
                                continue
                            source_x, source_y = source_pair
                            source_ns.append(source_n)
                            draw_seed = stable_seed(
                                int(cfg.seed_offset),
                                seed,
                                model_meta["model"],
                                target,
                                budget,
                                group_name,
                                rank,
                                "subspace_logistic_refit_target_direction",
                            )
                            few_h, few_n = sampled_pair(cache, target, str(cfg.train_split), budget, draw_seed)
                            basis = target_refit_subspace_basis(directions, few_h, few_n, rank)
                            target_x, target_y = stack_pairs([(few_h, few_n)])
                            train_x = torch.cat([source_x, target_x], dim=0) @ basis
                            train_y = torch.cat([source_y, target_y], dim=0)
                            eval_x, spec = cached_eval(target)
                            calibration_eval_x = target_x @ basis
                            scored_x = torch.cat([calibration_eval_x, eval_x @ basis], dim=0)
                            logits = logistic_logits(
                                train_x,
                                train_y,
                                scored_x,
                                seed=draw_seed,
                                l2=float(cfg.subspace_logistic.l2),
                                lr=float(cfg.subspace_logistic.lr),
                                epochs=int(cfg.subspace_logistic.epochs),
                                standardize=bool(cfg.subspace_logistic.standardize),
                                balanced_loss=bool(cfg.subspace_logistic.balanced_loss),
                            )
                            calibration_logits = logits[: target_x.shape[0]]
                            eval_logits = logits[target_x.shape[0] :]
                            calibration_scores[target] = ScoreBundle(
                                calibration_logits[: few_h.shape[0]],
                                calibration_logits[few_h.shape[0] :],
                            )
                            val_one, test_one = split_logits(eval_logits, spec)
                            val_scores[target] = val_one[target]
                            test_scores[target] = test_one[target]
                        if set(val_scores) != set(target_languages):
                            continue
                        threshold_scores, threshold_source = threshold_fit_scores(cfg, calibration_scores)
                        append_method(
                            rows,
                            metadata={
                                **model_meta,
                                "method": "subspace_logistic_refit_target_direction",
                                "classifier": "logistic",
                                "source_group": group_name,
                                "source_languages": "target_specific"
                                if "loo" in group_name
                                else ",".join(source_languages(group_name, languages, tiers)),
                                "source_n": min(source_ns) if min(source_ns) == max(source_ns) else "target_specific",
                                "rank": rank,
                                "budget": budget,
                                "seed": seed,
                            },
                            target_tiers=tiers,
                            validation_scores=val_scores,
                            test_scores=test_scores,
                            threshold_scores=threshold_scores,
                            threshold_source=threshold_source,
                            include_oracle_threshold=bool(cfg.include_oracle_thresholds),
                            threshold_scope=str(cfg.threshold_selection),
                        )
                    if bool(cfg.subspace_logistic.get("include_residual_memory", False)) and "loo" not in group_name:
                        shared_target = target_languages[0]
                        cached = cached_basis(group_name, rank, shared_target)
                        source_pair = cached_source(group_name, shared_target)
                        if cached is None or source_pair is None:
                            continue
                        base_basis, source_n = cached
                        source_x, source_y = source_pair
                        memory_pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
                        memory_samples: list[tuple[str, torch.Tensor, torch.Tensor]] = []
                        memory_directions: list[torch.Tensor] = []
                        for target in target_languages:
                            draw_seed = stable_seed(
                                int(cfg.seed_offset),
                                seed,
                                model_meta["model"],
                                target,
                                budget,
                                group_name,
                                rank,
                                "subspace_logistic_residual_memory",
                            )
                            few_h, few_n = sampled_pair(cache, target, str(cfg.train_split), budget, draw_seed)
                            memory_pairs.append((few_h, few_n))
                            memory_samples.append((target, few_h, few_n))
                            memory_directions.append(contrast_direction(few_h, few_n))
                        target_directions = torch.stack(memory_directions, dim=0)
                        target_x, target_y = stack_pairs(memory_pairs)
                        eval_x, spec = eval_x_for_targets(cache, target_languages, [str(cfg.val_split), str(cfg.test_split)])
                        for memory_rank_raw in list(cfg.subspace_logistic.get("residual_memory_ranks", [1])):
                            memory_rank = int(memory_rank_raw)
                            basis = residual_memory_subspace_basis(base_basis, target_directions, memory_rank)
                            train_x = torch.cat([source_x, target_x], dim=0) @ basis
                            train_y = torch.cat([source_y, target_y], dim=0)
                            scored_x = torch.cat([target_x @ basis, eval_x @ basis], dim=0)
                            train_seed = stable_seed(
                                int(cfg.seed_offset),
                                seed,
                                model_meta["model"],
                                budget,
                                group_name,
                                rank,
                                memory_rank,
                                "subspace_logistic_residual_memory",
                            )
                            logits = logistic_logits(
                                train_x,
                                train_y,
                                scored_x,
                                seed=train_seed,
                                l2=float(cfg.subspace_logistic.l2),
                                lr=float(cfg.subspace_logistic.lr),
                                epochs=int(cfg.subspace_logistic.epochs),
                                standardize=bool(cfg.subspace_logistic.standardize),
                                balanced_loss=bool(cfg.subspace_logistic.balanced_loss),
                            )
                            calibration_logits = logits[: target_x.shape[0]]
                            eval_logits = logits[target_x.shape[0] :]
                            calibration_scores = {}
                            offset = 0
                            for target, few_h, few_n in memory_samples:
                                harmful_logits = calibration_logits[offset : offset + few_h.shape[0]]
                                offset += few_h.shape[0]
                                harmless_logits = calibration_logits[offset : offset + few_n.shape[0]]
                                offset += few_n.shape[0]
                                calibration_scores[target] = ScoreBundle(harmful_logits, harmless_logits)
                            val_scores, test_scores = split_logits(eval_logits, spec)
                            threshold_scores, threshold_source = threshold_fit_scores(cfg, calibration_scores)
                            append_method(
                                rows,
                                metadata={
                                    **model_meta,
                                    "method": "subspace_logistic_residual_memory",
                                    "classifier": "logistic",
                                    "source_group": group_name,
                                    "source_languages": ",".join(source_languages(group_name, languages, tiers)),
                                    "source_n": source_n,
                                    "rank": basis.shape[1],
                                    "base_rank": rank,
                                    "memory_rank": memory_rank,
                                    "memory_languages": ",".join(target_languages),
                                    "budget": budget,
                                    "seed": seed,
                                },
                                target_tiers=tiers,
                                validation_scores=val_scores,
                                test_scores=test_scores,
                                threshold_scores=threshold_scores,
                                threshold_source=threshold_source,
                                include_oracle_threshold=bool(cfg.include_oracle_thresholds),
                                threshold_scope=str(cfg.threshold_selection),
                            )


def append_omniguard_methods(
    rows: list[dict[str, object]],
    *,
    cache: OmniguardRepresentationCache,
    model_meta: dict[str, object],
    languages: list[str],
    tiers: dict[str, str],
    target_languages: list[str],
    cfg: DictConfig,
) -> None:
    if not bool(cfg.omniguard.enabled):
        return

    model_meta = {
        **model_meta,
        "layer": cache.layer,
        "token_position": "mean_prompt_tokens",
        "representation": "mean_pooled_prompt",
        "layer_selection": "uscore",
    }
    eval_x, spec = eval_x_for_targets(cache, target_languages, [str(cfg.val_split), str(cfg.test_split)])
    hidden_dims = [int(x) for x in list(cfg.omniguard.hidden_dims)]

    if bool(cfg.omniguard.get("include_full", True)):
        full_train_x, full_train_y = stack_pairs(cache.pair(language, str(cfg.train_split)) for language in languages)
        for seed in range(int(cfg.omniguard.seeds)):
            train_seed = stable_seed(int(cfg.seed_offset), seed, model_meta["model"], "omniguard_full")
            logits = mlp_logits(
                full_train_x,
                full_train_y,
                eval_x,
                seed=train_seed,
                hidden_dims=hidden_dims,
                dropout=float(cfg.omniguard.dropout),
                l2=float(cfg.omniguard.l2),
                lr=float(cfg.omniguard.lr),
                epochs=int(cfg.omniguard.epochs),
                batch_size=int(cfg.omniguard.batch_size),
                standardize=bool(cfg.omniguard.standardize),
                balanced_loss=bool(cfg.omniguard.balanced_loss),
            )
            val_scores, test_scores = split_logits(logits, spec)
            append_method(
                rows,
                metadata={
                    **model_meta,
                    "method": "omniguard_full_polyrefuse",
                    "classifier": "mlp",
                    "source_group": "all",
                    "source_languages": ",".join(languages),
                    "source_n": len(languages),
                    "rank": "",
                    "budget": "full",
                    "seed": seed,
                },
                target_tiers=tiers,
                validation_scores=val_scores,
                test_scores=test_scores,
                include_oracle_threshold=bool(cfg.include_oracle_thresholds),
                threshold_scope=str(cfg.threshold_selection),
            )

    source_langs = source_languages(str(cfg.omniguard.budgeted_source_group), languages, tiers)
    source_train_x, source_train_y = stack_pairs(cache.pair(language, str(cfg.train_split)) for language in source_langs)

    if bool(cfg.omniguard.get("include_budgeted_lrl", True)):
        for budget in list(cfg.budgets):
            budget = int(budget)
            for seed in range(int(cfg.omniguard.seeds)):
                chunks_x = [source_train_x]
                chunks_y = [source_train_y]
                calibration_chunks: list[torch.Tensor] = []
                calibration_spec: list[tuple[str, str, int, int]] = []
                for target in target_languages:
                    draw_seed = stable_seed(
                        int(cfg.seed_offset),
                        seed,
                        model_meta["model"],
                        target,
                        budget,
                        "omniguard_budgeted",
                    )
                    few_h, few_n = sampled_pair(cache, target, str(cfg.train_split), budget, draw_seed)
                    target_x, target_y = stack_pairs([(few_h, few_n)])
                    chunks_x.append(target_x)
                    chunks_y.append(target_y)
                    calibration_chunks.append(target_x)
                    calibration_spec.append(("calibration", target, few_h.shape[0], few_n.shape[0]))
                train_x = torch.cat(chunks_x, dim=0)
                train_y = torch.cat(chunks_y, dim=0)
                calibration_x = torch.cat(calibration_chunks, dim=0)
                scored_x = torch.cat([calibration_x, eval_x], dim=0)
                train_seed = stable_seed(int(cfg.seed_offset), seed, model_meta["model"], budget, "omniguard_budgeted")
                logits = mlp_logits(
                    train_x,
                    train_y,
                    scored_x,
                    seed=train_seed,
                    hidden_dims=hidden_dims,
                    dropout=float(cfg.omniguard.dropout),
                    l2=float(cfg.omniguard.l2),
                    lr=float(cfg.omniguard.lr),
                    epochs=int(cfg.omniguard.epochs),
                    batch_size=int(cfg.omniguard.batch_size),
                    standardize=bool(cfg.omniguard.standardize),
                    balanced_loss=bool(cfg.omniguard.balanced_loss),
                )
                calibration_logits = logits[: calibration_x.shape[0]]
                eval_logits = logits[calibration_x.shape[0] :]
                calibration_scores = split_logits_by_split(calibration_logits, calibration_spec)["calibration"]
                val_scores, test_scores = split_logits(eval_logits, spec)
                threshold_scores, threshold_source = threshold_fit_scores(cfg, calibration_scores)
                append_method(
                    rows,
                    metadata={
                        **model_meta,
                        "method": "omniguard_budgeted_lrl",
                        "classifier": "mlp",
                        "source_group": str(cfg.omniguard.budgeted_source_group),
                        "source_languages": ",".join(source_langs),
                        "source_n": len(source_langs),
                        "rank": "",
                        "budget": budget,
                        "seed": seed,
                    },
                    target_tiers=tiers,
                    validation_scores=val_scores,
                    test_scores=test_scores,
                    threshold_scores=threshold_scores,
                    threshold_source=threshold_source,
                    include_oracle_threshold=bool(cfg.include_oracle_thresholds),
                    threshold_scope=str(cfg.threshold_selection),
                )

    if bool(cfg.omniguard.get("include_finetuned_lrl", False)):
        finetune_lr = float(cfg.omniguard.get("finetune_lr", cfg.omniguard.lr))
        finetune_l2 = float(cfg.omniguard.get("finetune_l2", cfg.omniguard.l2))
        finetune_epochs = int(cfg.omniguard.get("finetune_epochs", cfg.omniguard.epochs))
        finetune_batch_size = int(cfg.omniguard.get("finetune_batch_size", cfg.omniguard.batch_size))
        for seed in range(int(cfg.omniguard.seeds)):
            base_seed = stable_seed(int(cfg.seed_offset), seed, model_meta["model"], "omniguard_finetuned_lrl_base")
            base_probe = fit_mlp_probe(
                source_train_x,
                source_train_y,
                seed=base_seed,
                hidden_dims=hidden_dims,
                dropout=float(cfg.omniguard.dropout),
                l2=float(cfg.omniguard.l2),
                lr=float(cfg.omniguard.lr),
                epochs=int(cfg.omniguard.epochs),
                batch_size=int(cfg.omniguard.batch_size),
                standardize=bool(cfg.omniguard.standardize),
                balanced_loss=bool(cfg.omniguard.balanced_loss),
            )
            with torch.no_grad():
                source_logits = base_probe.logits(source_train_x)
                eval_logits = base_probe.logits(eval_x)
            val_scores, test_scores = split_logits(eval_logits, spec)
            append_method(
                rows,
                metadata={
                    **model_meta,
                    "method": "omniguard_finetuned_lrl",
                    "classifier": "mlp_finetune",
                    "source_group": str(cfg.omniguard.budgeted_source_group),
                    "source_languages": ",".join(source_langs),
                    "source_n": len(source_langs),
                    "rank": "",
                    "budget": 0,
                    "seed": seed,
                },
                target_tiers=tiers,
                validation_scores=val_scores,
                test_scores=test_scores,
                threshold_scores={"source_train": source_train_logit_scores(source_logits, source_train_y)},
                threshold_source="source_train",
                include_oracle_threshold=False,
                threshold_scope=str(cfg.threshold_selection),
            )

            for budget in list(cfg.budgets):
                budget = int(budget)
                if budget <= 0:
                    continue
                val_scores: dict[str, ScoreBundle] = {}
                test_scores: dict[str, ScoreBundle] = {}
                calibration_scores: dict[str, ScoreBundle] = {}
                for target in target_languages:
                    draw_seed = stable_seed(
                        int(cfg.seed_offset),
                        seed,
                        model_meta["model"],
                        target,
                        budget,
                        "omniguard_budgeted",
                    )
                    few_h, few_n = sampled_pair(cache, target, str(cfg.train_split), budget, draw_seed)
                    target_x, target_y = stack_pairs([(few_h, few_n)])
                    target_eval_x, target_spec = eval_x_for_targets(
                        cache,
                        [target],
                        [str(cfg.val_split), str(cfg.test_split)],
                    )
                    scored_x = torch.cat([target_x, target_eval_x], dim=0)
                    finetune_seed = stable_seed(
                        int(cfg.seed_offset),
                        seed,
                        model_meta["model"],
                        target,
                        budget,
                        "omniguard_finetuned_lrl",
                    )
                    tuned_probe = fit_mlp_probe(
                        target_x,
                        target_y,
                        seed=finetune_seed,
                        hidden_dims=hidden_dims,
                        dropout=float(cfg.omniguard.dropout),
                        l2=finetune_l2,
                        lr=finetune_lr,
                        epochs=finetune_epochs,
                        batch_size=finetune_batch_size,
                        standardize=bool(cfg.omniguard.standardize),
                        balanced_loss=bool(cfg.omniguard.balanced_loss),
                        initial_probe=base_probe,
                    )
                    with torch.no_grad():
                        logits = tuned_probe.logits(scored_x)
                    calibration_logits = logits[: target_x.shape[0]]
                    eval_logits = logits[target_x.shape[0] :]
                    calibration_scores[target] = ScoreBundle(
                        calibration_logits[: few_h.shape[0]],
                        calibration_logits[few_h.shape[0] :],
                    )
                    val_one, test_one = split_logits(eval_logits, target_spec)
                    val_scores[target] = val_one[target]
                    test_scores[target] = test_one[target]
                if set(val_scores) != set(target_languages):
                    continue
                threshold_scores, threshold_source = threshold_fit_scores(cfg, calibration_scores)
                append_method(
                    rows,
                    metadata={
                        **model_meta,
                        "method": "omniguard_finetuned_lrl",
                        "classifier": "mlp_finetune",
                        "source_group": str(cfg.omniguard.budgeted_source_group),
                        "source_languages": ",".join(source_langs),
                        "source_n": len(source_langs),
                        "rank": "",
                        "budget": budget,
                        "seed": seed,
                    },
                    target_tiers=tiers,
                    validation_scores=val_scores,
                    test_scores=test_scores,
                    threshold_scores=threshold_scores,
                    threshold_source=threshold_source,
                    include_oracle_threshold=bool(cfg.include_oracle_thresholds),
                    threshold_scope=str(cfg.threshold_selection),
                )


def test_macro_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        row
        for row in rows
        if row.get("split") == "test" and str(row.get("target_language", "")).startswith("__macro")
    ]


def run_probe_method_comparison(cfg: DictConfig) -> None:
    languages = [str(language) for language in cfg.dataset.languages]
    tiers = {str(language): str(tier) for language, tier in cfg.dataset.resource_tier.items()}
    if cfg.languages is None:
        target_tiers = set(cfg.target_tiers)
        target_languages = [language for language in languages if tiers[language] in target_tiers]
    else:
        target_languages = list(cfg.languages)
    target_languages = [str(language) for language in target_languages]
    device_name = str(cfg.device)
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else device_name)

    all_rows: list[dict[str, object]] = []
    for model_cfg in cfg.models:
        model = str(model_cfg.name)
        layer = int(model_cfg.layer)
        representation_source = str(cfg.get("representation_source", "activations"))
        base_model_meta = {
            "model": model,
            "model_short": str(model_cfg.short),
        }
        if representation_source == "activations":
            model_meta = {
                **base_model_meta,
                "layer": layer,
                "token_position": str(cfg.token_position),
                "representation": "single_token_activation",
                "layer_selection": "fixed_analysis_layer",
            }
            acts_root = Path(str(cfg.activations_root_template).format(model=model))
            cache = ActivationCache(acts_root, layer, str(cfg.token_position), device)
        elif representation_source == "omniguard":
            omni_root_for_cache = Path(
                str(cfg.omniguard.representations_root_template).format(model=model)
            )
            uscore_root_for_cache = Path(
                str(cfg.omniguard.uscore_root_template).format(model=model)
            )
            selected_path_for_cache = selected_layer_file(uscore_root_for_cache)
            if cfg.omniguard.layer_override is not None:
                cache_layer = int(cfg.omniguard.layer_override)
            elif selected_path_for_cache.exists():
                cache_layer = read_selected_layer(selected_path_for_cache)
            elif bool(cfg.omniguard.require_uscore_layer):
                raise FileNotFoundError(
                    f"missing OMNIGuard U-Score selected layer: {selected_path_for_cache}. "
                    "Run scripts/recoverability/extract_omniguard_representations.py and "
                    "scripts/recoverability/compute_omniguard_uscore.py first."
                )
            else:
                cache_layer = layer
            model_meta = {
                **base_model_meta,
                "layer": cache_layer,
                "token_position": "mean_prompt_tokens",
                "representation": "mean_pooled_prompt",
                "layer_selection": "uscore",
            }
            cache = OmniguardRepresentationCache(omni_root_for_cache, cache_layer, device)
        else:
            raise ValueError(f"unknown representation_source {representation_source!r}")
        rows: list[dict[str, object]] = []

        if bool(cfg.include_direction_baselines):
            append_direction_baselines(
                rows,
                cache=cache,
                model_meta=model_meta,
                languages=languages,
                tiers=tiers,
                target_languages=target_languages,
                cfg=cfg,
            )
        if bool(cfg.include_target_only):
            append_target_direction_methods(
                rows,
                cache=cache,
                model_meta=model_meta,
                target_languages=target_languages,
                tiers=tiers,
                cfg=cfg,
            )
        append_data_matched_baselines(
            rows,
            cache=cache,
            model_meta=model_meta,
            languages=languages,
            tiers=tiers,
            target_languages=target_languages,
            cfg=cfg,
        )
        if bool(cfg.include_subspaces):
            append_subspace_methods(
                rows,
                cache=cache,
                model_meta=model_meta,
                languages=languages,
                tiers=tiers,
                target_languages=target_languages,
                cfg=cfg,
            )
            append_subspace_logistic_methods(
                rows,
                cache=cache,
                model_meta=model_meta,
                languages=languages,
                tiers=tiers,
                target_languages=target_languages,
                cfg=cfg,
            )
        if bool(cfg.omniguard.enabled):
            omni_root = Path(str(cfg.omniguard.representations_root_template).format(model=model))
            uscore_root = Path(str(cfg.omniguard.uscore_root_template).format(model=model))
            selected_path = selected_layer_file(uscore_root)
            if cfg.omniguard.layer_override is not None:
                omni_layer = int(cfg.omniguard.layer_override)
            elif selected_path.exists():
                omni_layer = read_selected_layer(selected_path)
            elif bool(cfg.omniguard.require_uscore_layer):
                raise FileNotFoundError(
                    f"missing OMNIGuard U-Score selected layer: {selected_path}. "
                    "Run scripts/recoverability/extract_omniguard_representations.py and "
                    "scripts/recoverability/compute_omniguard_uscore.py first."
                )
            else:
                omni_layer = layer
            missing = [
                representation_file(omni_root, language, str(cfg.val_split), "harmful", omni_layer)
                for language in target_languages
                if not representation_file(omni_root, language, str(cfg.val_split), "harmful", omni_layer).exists()
            ]
            if missing:
                raise FileNotFoundError(
                    f"missing OMNIGuard mean-pooled representations for selected layer {omni_layer}; "
                    f"first missing file: {missing[0]}"
                )
            append_omniguard_methods(
                rows,
                cache=OmniguardRepresentationCache(omni_root, omni_layer, device),
                model_meta=model_meta,
                languages=languages,
                tiers=tiers,
                target_languages=target_languages,
                cfg=cfg,
            )

        out_dir = Path(cfg.output_root) / model
        write_rows(out_dir / "method_rows.csv", rows, PREFERRED_FIELDS)
        write_rows(out_dir / "test_macro_summary.csv", test_macro_rows(rows), PREFERRED_FIELDS)
        all_rows.extend(rows)
        print(f"[done] {model}: wrote {len(rows)} rows to {out_dir}")

    combined_dir = Path(cfg.output_root) / "_combined"
    write_rows(combined_dir / "method_rows.csv", all_rows, PREFERRED_FIELDS)
    write_rows(combined_dir / "test_macro_summary.csv", test_macro_rows(all_rows), PREFERRED_FIELDS)
    print(f"[done] wrote {len(all_rows)} combined rows to {combined_dir}")


@hydra.main(version_base=None, config_path="../../configs", config_name="recoverability/compute_probe_method_comparison")
def main(cfg: DictConfig) -> None:
    run_probe_method_comparison(cfg)


if __name__ == "__main__":
    main()
