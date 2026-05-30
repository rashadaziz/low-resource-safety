"""Crosslingual zero-shot transfer for PolyRefuse harmfulness subspaces."""

from dataclasses import dataclass
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

from multilingual_latent_safety.activation_store import ActivationCache
from multilingual_latent_safety.csv_io import write_rows
from multilingual_latent_safety.probe_evaluation import (
    ScoreBundle,
    append_method,
    language_direction_matrix,
    source_languages,
    source_pair,
    source_train_logit_scores,
    split_logits_by_split,
    logistic_logits,
    subspace_basis,
)
from multilingual_latent_safety.runtime import stable_seed


@dataclass(frozen=True)
class SourcePlan:
    condition: str
    source_languages: tuple[str, ...]
    targets: tuple[str, ...]
    source_added: str


def languages_with_tier(languages: list[str], tiers: dict[str, str], tier: str) -> list[str]:
    return [language for language in languages if tiers[language] == tier]


def build_source_plans(
    languages: list[str],
    tiers: dict[str, str],
    *,
    conditions: list[str],
) -> list[SourcePlan]:
    high = languages_with_tier(languages, tiers, "high")
    mid = languages_with_tier(languages, tiers, "mid")
    low = languages_with_tier(languages, tiers, "low")
    plans: list[SourcePlan] = []
    if "hrl_to_target" in conditions:
        plans.append(
            SourcePlan(
                condition="hrl_to_target",
                source_languages=tuple(high),
                targets=tuple(mid + low),
                source_added="",
            )
        )
    if "hrl_plus_lrl_full" in conditions:
        for added in low:
            targets = [language for language in low if language != added]
            plans.append(
                SourcePlan(
                    condition="hrl_plus_lrl_full",
                    source_languages=tuple(high + [added]),
                    targets=tuple(targets),
                    source_added=added,
                )
            )
    if "hrl_plus_mrl_full" in conditions:
        for added in mid:
            plans.append(
                SourcePlan(
                    condition="hrl_plus_mrl_full",
                    source_languages=tuple(high + [added]),
                    targets=tuple(low),
                    source_added=added,
                )
            )
    return [plan for plan in plans if plan.targets]


def target_scores_from_logits(logits: torch.Tensor, n_harmful: int) -> ScoreBundle:
    return ScoreBundle(logits[:n_harmful], logits[n_harmful:])


def score_targets_once(
    *,
    cache: ActivationCache,
    targets: tuple[str, ...],
    splits: tuple[str, ...],
    basis: torch.Tensor,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    seed: int,
    cfg: DictConfig,
) -> tuple[ScoreBundle, dict[str, dict[str, ScoreBundle]]]:
    source_features = train_x @ basis
    chunks: list[torch.Tensor] = []
    spec: list[tuple[str, str, int, int]] = []
    for split in splits:
        for target in targets:
            harmful, harmless = cache.pair(target, split)
            chunks.extend([harmful, harmless])
            spec.append((split, target, harmful.shape[0], harmless.shape[0]))
    eval_x = torch.cat(chunks, dim=0) @ basis
    scored_x = torch.cat([source_features, eval_x], dim=0)
    logits = logistic_logits(
        source_features,
        train_y,
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
    return source_train_logit_scores(source_logits, train_y), split_logits_by_split(eval_logits, spec)


def run_crosslingual_subspace_transfer(cfg: DictConfig) -> None:
    languages = [str(language) for language in cfg.dataset.languages]
    tiers = {str(language): str(tier) for language, tier in cfg.dataset.resource_tier.items()}
    plans = build_source_plans(languages, tiers, conditions=[str(item) for item in cfg.conditions])
    device_name = str(cfg.device)
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else device_name)
    all_rows: list[dict[str, object]] = []

    for model_cfg in cfg.models:
        root = Path(str(cfg.activations_root_template).format(model=str(model_cfg.name)))
        cache = ActivationCache(root, int(model_cfg.layer), str(cfg.token_position), device)
        rows: list[dict[str, object]] = []
        for plan in plans:
            source = list(plan.source_languages)
            rank = int(cfg.rank)
            directions = language_direction_matrix(cache, source, str(cfg.train_split))
            if rank > directions.shape[0]:
                continue
            basis = subspace_basis(directions, rank)
            train_x, train_y = source_pair(cache, source, str(cfg.train_split))
            validation_scores: dict[str, ScoreBundle] = {}
            test_scores: dict[str, ScoreBundle] = {}
            threshold_scores: dict[str, ScoreBundle] = {}
            seed = stable_seed(int(cfg.seed_offset), str(model_cfg.name), plan.condition, plan.source_added)
            source_scores, split_scores = score_targets_once(
                cache=cache,
                targets=plan.targets,
                splits=(str(cfg.val_split), str(cfg.test_split)),
                basis=basis,
                train_x=train_x,
                train_y=train_y,
                seed=seed,
                cfg=cfg,
            )
            validation_scores.update(split_scores[str(cfg.val_split)])
            test_scores.update(split_scores[str(cfg.test_split)])
            threshold_scores.update({f"source_train::{target}": source_scores for target in plan.targets})
            append_method(
                rows,
                metadata={
                    "model": str(model_cfg.name),
                    "model_short": str(model_cfg.short),
                    "layer": int(model_cfg.layer),
                    "method": "crosslingual_subspace_logistic",
                    "classifier": "logistic_subspace",
                    "source_condition": plan.condition,
                    "source_added": plan.source_added,
                    "source_group": plan.condition,
                    "source_languages": ",".join(source),
                    "source_n": len(source),
                    "rank": rank,
                    "budget": 0,
                    "seed": "",
                },
                target_tiers=tiers,
                validation_scores=validation_scores,
                test_scores=test_scores,
                threshold_scores=threshold_scores,
                threshold_source="source_train",
                threshold_scope=str(cfg.threshold_selection),
            )
        out = Path(cfg.output_root) / str(model_cfg.name) / "crosslingual_subspace_transfer.csv"
        write_rows(out, rows)
        all_rows.extend(rows)
        print(f"[done] {model_cfg.short}: {len(rows)} rows")

    combined = Path(cfg.output_root) / "_combined" / "crosslingual_subspace_transfer.csv"
    write_rows(combined, all_rows)
    print(f"[done] combined: {len(all_rows)} rows")


@hydra.main(version_base=None, config_path="../../configs", config_name="ood_transfer/compute_crosslingual_subspace_transfer")
def main(cfg: DictConfig) -> None:
    run_crosslingual_subspace_transfer(cfg)


if __name__ == "__main__":
    main()
