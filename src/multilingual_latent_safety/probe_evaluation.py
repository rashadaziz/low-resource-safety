"""Reusable probe, threshold, and subspace evaluation helpers."""

import copy
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F

from multilingual_latent_safety.activation_store import ActivationCache
from multilingual_latent_safety.analysis import auc
from multilingual_latent_safety.probes import FeatureScaler
from multilingual_latent_safety.runtime import set_seed


EPS = 1e-8


@dataclass(frozen=True)
class ScoreBundle:
    harmful: torch.Tensor
    harmless: torch.Tensor


@dataclass(frozen=True)
class LogisticProbe:
    """Stored linear probe with the standardizer needed for inference."""

    weight: torch.Tensor
    bias: torch.Tensor
    scaler: FeatureScaler | None = None

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(device=self.weight.device, dtype=torch.float32)
        if self.scaler is not None:
            x = self.scaler.transform(x)
        return x @ self.weight + self.bias


@dataclass(frozen=True)
class MlpProbe:
    """Stored nonlinear probe with the standardizer needed for inference."""

    model: torch.nn.Module
    scaler: FeatureScaler | None = None

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        parameter = next(self.model.parameters())
        x = x.to(device=parameter.device, dtype=torch.float32)
        if self.scaler is not None:
            x = self.scaler.transform(x)
        self.model.eval()
        return self.model(x).squeeze(-1)


@dataclass(frozen=True)
class ThresholdChoice:
    threshold: float
    validation_macro_f1: float


def normalize_vector(vector: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    vector = vector.to(torch.float32)
    return vector / vector.norm().clamp(min=eps)


def normalize_rows(matrix: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    matrix = matrix.to(torch.float32)
    return matrix / matrix.norm(dim=1, keepdim=True).clamp(min=eps)


def subspace_basis(directions: torch.Tensor, rank: int) -> torch.Tensor:
    """Column-orthonormal basis from a row stack of source directions."""
    if directions.ndim != 2:
        raise ValueError(f"directions must be 2D, got shape {tuple(directions.shape)}")
    if rank < 1:
        raise ValueError(f"rank must be positive, got {rank}")
    max_rank = min(directions.shape)
    if rank > max_rank:
        raise ValueError(f"rank {rank} exceeds max rank {max_rank}")
    _, _, vh = torch.linalg.svd(directions.to(torch.float32), full_matrices=False)
    return vh[:rank].T.contiguous()


def pca_subspace_basis(activations: torch.Tensor, rank: int) -> torch.Tensor:
    """Column-orthonormal PCA basis from unlabeled activations."""
    if activations.ndim != 2:
        raise ValueError(f"activations must be 2D, got shape {tuple(activations.shape)}")
    if rank < 1:
        raise ValueError(f"rank must be positive, got {rank}")
    max_rank = min(activations.shape)
    if rank > max_rank:
        raise ValueError(f"rank {rank} exceeds max rank {max_rank}")
    centered = activations.to(torch.float32) - activations.to(torch.float32).mean(dim=0, keepdim=True)
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    return vh[:rank].T.contiguous()


def random_subspace_basis(dim: int, rank: int, *, seed: int, device: torch.device | None = None) -> torch.Tensor:
    """Deterministic random column-orthonormal basis."""
    if dim < 1:
        raise ValueError(f"dim must be positive, got {dim}")
    if rank < 1:
        raise ValueError(f"rank must be positive, got {rank}")
    if rank > dim:
        raise ValueError(f"rank {rank} exceeds dim {dim}")
    generator = torch.Generator().manual_seed(int(seed))
    random_matrix = torch.randn(dim, rank, generator=generator, dtype=torch.float32)
    q, _ = torch.linalg.qr(random_matrix, mode="reduced")
    return q if device is None else q.to(device)


def project_onto_basis(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    if x.ndim == 1:
        return basis @ (basis.T @ x.to(torch.float32))
    if x.ndim == 2:
        return (x.to(torch.float32) @ basis) @ basis.T
    raise ValueError(f"x must be 1D or 2D, got shape {tuple(x.shape)}")


def contrast_direction(
    harmful: torch.Tensor,
    harmless: torch.Tensor,
    basis: torch.Tensor | None = None,
) -> torch.Tensor:
    direction = harmful.to(torch.float32).mean(dim=0) - harmless.to(torch.float32).mean(dim=0)
    if basis is not None:
        direction = project_onto_basis(direction, basis)
    return normalize_vector(direction)


def target_refit_subspace_basis(
    source_directions: torch.Tensor,
    harmful: torch.Tensor,
    harmless: torch.Tensor,
    rank: int,
) -> torch.Tensor:
    """Refit a source subspace after appending one target train direction."""

    target_direction = contrast_direction(harmful, harmless).to(source_directions.device)
    directions = torch.cat([source_directions.to(torch.float32), target_direction[None, :]], dim=0)
    return subspace_basis(directions, rank)


def residual_memory_subspace_basis(
    base_basis: torch.Tensor,
    target_directions: torch.Tensor,
    memory_rank: int,
    eps: float = EPS,
) -> torch.Tensor:
    """Augment a fixed base basis with supervised target residual directions."""

    if base_basis.ndim != 2:
        raise ValueError(f"base_basis must be 2D, got shape {tuple(base_basis.shape)}")
    if target_directions.ndim != 2:
        raise ValueError(f"target_directions must be 2D, got shape {tuple(target_directions.shape)}")
    if target_directions.shape[1] != base_basis.shape[0]:
        raise ValueError(
            "target direction width must match basis dimension, got "
            f"{target_directions.shape[1]} vs {base_basis.shape[0]}"
        )
    if memory_rank < 0:
        raise ValueError(f"memory_rank must be non-negative, got {memory_rank}")
    if memory_rank == 0:
        return base_basis.to(torch.float32).contiguous()

    base_basis = base_basis.to(torch.float32)
    residuals = target_directions.to(torch.float32) - (target_directions.to(torch.float32) @ base_basis) @ base_basis.T
    residuals = residuals[residuals.norm(dim=1) > eps]
    if residuals.numel() == 0:
        return base_basis.contiguous()

    max_rank = min(memory_rank, residuals.shape[0], residuals.shape[1])
    _, _, vh = torch.linalg.svd(residuals, full_matrices=False)
    residual_basis = vh[:max_rank].T.contiguous()
    residual_basis = residual_basis - base_basis @ (base_basis.T @ residual_basis)
    residual_basis, _ = torch.linalg.qr(residual_basis, mode="reduced")
    residual_basis = residual_basis[:, :max_rank]
    return torch.cat([base_basis, residual_basis.to(base_basis.device)], dim=1).contiguous()


def direction_scores(direction: torch.Tensor, harmful: torch.Tensor, harmless: torch.Tensor) -> ScoreBundle:
    direction = normalize_vector(direction).to(harmful.device)
    return ScoreBundle(
        harmful=harmful.to(torch.float32) @ direction,
        harmless=harmless.to(torch.float32) @ direction,
    )


def average_precision(harmful_scores: torch.Tensor, harmless_scores: torch.Tensor) -> float:
    harmful_scores = harmful_scores.detach().to(torch.float64).flatten()
    harmless_scores = harmless_scores.detach().to(torch.float64).flatten()
    if harmful_scores.numel() == 0:
        return float("nan")
    scores = torch.cat([harmful_scores, harmless_scores])
    labels = torch.cat([torch.ones_like(harmful_scores), torch.zeros_like(harmless_scores)])
    order = torch.argsort(scores, descending=True)
    labels = labels[order]
    tp = torch.cumsum(labels, dim=0)
    ranks = torch.arange(1, labels.numel() + 1, dtype=torch.float64, device=labels.device)
    precision_at_k = tp / ranks
    return float((precision_at_k * labels).sum() / harmful_scores.numel())


def binary_metrics(scores: ScoreBundle, threshold: float) -> dict[str, float | int]:
    harmful = scores.harmful.to(torch.float32).flatten()
    harmless = scores.harmless.to(torch.float32).flatten()
    hp = harmful > float(threshold)
    np = harmless > float(threshold)
    tp = int(hp.sum().item())
    fn = int((~hp).sum().item())
    fp = int(np.sum().item())
    tn = int((~np).sum().item())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 0.0 if precision + recall <= 0 else 2.0 * precision * recall / (precision + recall)
    tnr = tn / max(tn + fp, 1)
    total = tp + fp + tn + fn
    return {
        "auc": auc(harmful, harmless),
        "average_precision": average_precision(harmful, harmless),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": (tp + tn) / max(total, 1),
        "balanced_accuracy": 0.5 * (recall + tnr),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "n_harmful": int(harmful.numel()),
        "n_harmless": int(harmless.numel()),
    }


def f1_at_threshold(scores: ScoreBundle, threshold: float) -> float:
    harmful = scores.harmful.to(torch.float32).flatten()
    harmless = scores.harmless.to(torch.float32).flatten()
    tp = int((harmful > float(threshold)).sum().item())
    fn = int((harmful <= float(threshold)).sum().item())
    fp = int((harmless > float(threshold)).sum().item())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return 0.0 if precision + recall <= 0 else 2.0 * precision * recall / (precision + recall)


def candidate_thresholds(score_bundles: Iterable[ScoreBundle]) -> list[float]:
    return [float(x.item()) for x in candidate_threshold_tensor(score_bundles)]


def candidate_threshold_tensor(score_bundles: Iterable[ScoreBundle]) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    for scores in score_bundles:
        chunks.extend([scores.harmful.detach().flatten().cpu(), scores.harmless.detach().flatten().cpu()])
    if not chunks:
        raise ValueError("cannot choose a threshold without validation scores")
    values = torch.unique(torch.cat(chunks).to(torch.float64)).sort().values
    if values.numel() == 1:
        return torch.stack([values[0] - 1e-6, values[0] + 1e-6])
    mids = (values[:-1] + values[1:]) / 2.0
    return torch.cat([values[:1] - 1e-6, mids, values[-1:] + 1e-6])


def f1_curve(scores: ScoreBundle, thresholds: torch.Tensor, *, class_balance: bool = False) -> torch.Tensor:
    thresholds = thresholds.to(torch.float64).cpu()
    harmful = scores.harmful.detach().flatten().to(torch.float64).cpu().sort().values
    harmless = scores.harmless.detach().flatten().to(torch.float64).cpu().sort().values
    tp = harmful.numel() - torch.searchsorted(harmful, thresholds, right=True).to(torch.float64)
    fp = harmless.numel() - torch.searchsorted(harmless, thresholds, right=True).to(torch.float64)
    fn = torch.full_like(tp, float(harmful.numel())) - tp
    if class_balance:
        harmful_weight = 0.5 / max(harmful.numel(), 1)
        harmless_weight = 0.5 / max(harmless.numel(), 1)
        tp = tp * harmful_weight
        fn = fn * harmful_weight
        fp = fp * harmless_weight
    precision_denom = tp + fp
    recall_denom = tp + fn
    precision = torch.where(precision_denom > 0, tp / precision_denom, torch.zeros_like(tp))
    recall = torch.where(recall_denom > 0, tp / recall_denom, torch.zeros_like(tp))
    denom = precision + recall
    return torch.where(denom > 0, 2.0 * precision * recall / denom, torch.zeros_like(denom))


def macro_average(metric_rows: list[dict[str, float | int]]) -> dict[str, float]:
    if not metric_rows:
        raise ValueError("cannot average empty metrics")
    keys = [
        "auc",
        "average_precision",
        "precision",
        "recall",
        "f1",
        "accuracy",
        "balanced_accuracy",
    ]
    return {key: float(torch.tensor([float(row[key]) for row in metric_rows]).mean().item()) for key in keys}


def select_global_threshold(validation_scores: dict[str, ScoreBundle]) -> ThresholdChoice:
    return select_global_threshold_for_objective(validation_scores, objective="macro_f1")


def select_global_threshold_for_objective(
    validation_scores: dict[str, ScoreBundle],
    *,
    objective: str,
) -> ThresholdChoice:
    thresholds = candidate_threshold_tensor(validation_scores.values())
    if thresholds.numel() == 0:
        raise ValueError("failed to select threshold")
    if objective == "macro_f1":
        class_balance = False
    elif objective == "balanced_macro_f1":
        class_balance = True
    else:
        raise ValueError(f"unknown threshold objective {objective!r}")
    macro_f1 = torch.stack(
        [f1_curve(scores, thresholds, class_balance=class_balance) for scores in validation_scores.values()]
    ).mean(dim=0)
    best_idx = int(torch.argmax(macro_f1).item())
    return ThresholdChoice(threshold=float(thresholds[best_idx].item()), validation_macro_f1=float(macro_f1[best_idx].item()))


def select_language_thresholds(validation_scores: dict[str, ScoreBundle]) -> dict[str, ThresholdChoice]:
    choices: dict[str, ThresholdChoice] = {}
    for language, scores in validation_scores.items():
        thresholds = candidate_threshold_tensor([scores])
        if thresholds.numel() == 0:
            raise ValueError(f"failed to select threshold for {language}")
        f1 = f1_curve(scores, thresholds)
        best_idx = int(torch.argmax(f1).item())
        choices[language] = ThresholdChoice(float(thresholds[best_idx].item()), float(f1[best_idx].item()))
    return choices


def language_threshold_fit_scores(
    threshold_fit_scores: dict[str, ScoreBundle],
    validation_scores: dict[str, ScoreBundle],
) -> dict[str, ScoreBundle]:
    if set(threshold_fit_scores) == set(validation_scores):
        return threshold_fit_scores

    aligned: dict[str, ScoreBundle] = {}
    for language in validation_scores:
        source_train_key = f"source_train::{language}"
        if language in threshold_fit_scores:
            aligned[language] = threshold_fit_scores[language]
        elif source_train_key in threshold_fit_scores:
            aligned[language] = threshold_fit_scores[source_train_key]
        elif "source_train" in threshold_fit_scores:
            aligned[language] = threshold_fit_scores["source_train"]
        else:
            raise ValueError("language-oracle threshold score languages must match validation score languages")
    return aligned


def append_evaluation_rows(
    rows: list[dict[str, object]],
    *,
    metadata: dict[str, object],
    target_tiers: dict[str, str],
    validation_scores: dict[str, ScoreBundle],
    test_scores: dict[str, ScoreBundle],
    threshold_scores: dict[str, ScoreBundle] | None = None,
    threshold_source: str = "validation",
    threshold_scope: str = "global",
) -> None:
    threshold_fit_scores = validation_scores if threshold_scores is None else threshold_scores
    if not threshold_fit_scores:
        raise ValueError("threshold score map must be non-empty")

    if threshold_scope == "global":
        threshold_choice = select_global_threshold_for_objective(threshold_fit_scores, objective="macro_f1")
        thresholds = {language: threshold_choice for language in validation_scores}
        threshold_value: float | str = threshold_choice.threshold
        selection_score = threshold_choice.validation_macro_f1
    elif threshold_scope == "global_balanced":
        threshold_choice = select_global_threshold_for_objective(
            threshold_fit_scores,
            objective="balanced_macro_f1",
        )
        thresholds = {language: threshold_choice for language in validation_scores}
        threshold_value = threshold_choice.threshold
        selection_score = threshold_choice.validation_macro_f1
    elif threshold_scope == "language_oracle":
        threshold_fit_scores = language_threshold_fit_scores(threshold_fit_scores, validation_scores)
        thresholds = select_language_thresholds(threshold_fit_scores)
        threshold_value = ""
        selection_score = float(
            torch.tensor(
                [thresholds[language].validation_macro_f1 for language in threshold_fit_scores],
                dtype=torch.float32,
            )
            .mean()
            .item()
        )
    else:
        raise ValueError(f"unknown threshold scope {threshold_scope!r}")

    for split, score_map in (("val", validation_scores), ("test", test_scores)):
        split_metric_rows: list[tuple[str, dict[str, float | int]]] = []
        for language, scores in score_map.items():
            threshold = thresholds[language].threshold
            metrics = binary_metrics(scores, threshold)
            split_metric_rows.append((language, metrics))
            rows.append(
                {
                    **metadata,
                    "split": split,
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
            split=split,
            target_language="__macro__",
            target_tier="all",
            threshold_scope=threshold_scope,
            threshold_source=threshold_source,
            threshold_value=threshold_value,
            selection_score=selection_score,
            metric_rows=[metrics for _, metrics in split_metric_rows],
        )
        for tier in ("high", "mid", "low"):
            tier_rows = [
                metrics for language, metrics in split_metric_rows if target_tiers[language] == tier
            ]
            if tier_rows:
                append_macro_row(
                    rows,
                    metadata=metadata,
                    split=split,
                    target_language=f"__macro_{tier}__",
                    target_tier=tier,
                    threshold_scope=threshold_scope,
                    threshold_source=threshold_source,
                    threshold_value=threshold_value,
                    selection_score=selection_score,
                    metric_rows=tier_rows,
                )


def append_macro_row(
    rows: list[dict[str, object]],
    *,
    metadata: dict[str, object],
    split: str,
    target_language: str,
    target_tier: str,
    threshold_scope: str,
    threshold_source: str,
    threshold_value: float | str,
    selection_score: float,
    metric_rows: list[dict[str, float | int]],
) -> None:
    macro = macro_average(metric_rows)
    rows.append(
        {
            **metadata,
            "split": split,
            "target_language": target_language,
            "target_tier": target_tier,
            "threshold_scope": threshold_scope,
            "threshold_source": threshold_source,
            "threshold": threshold_value,
            "threshold_selection_macro_f1_at_threshold": selection_score,
            "validation_macro_f1_at_threshold": selection_score,
            **macro,
            "tp": sum(int(row["tp"]) for row in metric_rows),
            "fp": sum(int(row["fp"]) for row in metric_rows),
            "tn": sum(int(row["tn"]) for row in metric_rows),
            "fn": sum(int(row["fn"]) for row in metric_rows),
            "n_harmful": sum(int(row["n_harmful"]) for row in metric_rows),
            "n_harmless": sum(int(row["n_harmless"]) for row in metric_rows),
        }
    )


def languages_by_tier(languages: Iterable[str], tiers: dict[str, str], wanted: Iterable[str]) -> list[str]:
    wanted_set = set(wanted)
    return [language for language in languages if tiers[language] in wanted_set]


def source_languages(
    group: str,
    languages: list[str],
    tiers: dict[str, str],
    target_language: str | None = None,
) -> list[str]:
    high = languages_by_tier(languages, tiers, ["high"])
    mid = languages_by_tier(languages, tiers, ["mid"])
    low = languages_by_tier(languages, tiers, ["low"])
    if group == "hrl":
        return high
    if group == "mrl":
        return mid
    if group == "hrl_mrl":
        return high + mid
    if group == "lrl_loo":
        if target_language is None:
            raise ValueError("lrl_loo requires target_language")
        return [language for language in low if language != target_language]
    if group == "all_loo":
        if target_language is None:
            raise ValueError("all_loo requires target_language")
        return [language for language in languages if language != target_language]
    if group == "all":
        return list(languages)
    raise ValueError(f"unknown source group {group!r}")


def stack_pairs(pairs: Iterable[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    xs: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []
    for harmful, harmless in pairs:
        harmful = harmful.to(torch.float32)
        harmless = harmless.to(torch.float32)
        xs.extend([harmful, harmless])
        ys.extend(
            [
                torch.ones(harmful.shape[0], dtype=torch.float32, device=harmful.device),
                torch.zeros(harmless.shape[0], dtype=torch.float32, device=harmless.device),
            ]
        )
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0)


def language_direction_matrix(
    cache: ActivationCache,
    languages: list[str],
    split: str,
) -> torch.Tensor:
    return torch.stack(
        [contrast_direction(*cache.pair(language, split)) for language in languages],
        dim=0,
    )


def pooled_contrast_direction(
    cache: ActivationCache,
    languages: Iterable[str],
    split: str,
) -> torch.Tensor:
    harmful_chunks: list[torch.Tensor] = []
    harmless_chunks: list[torch.Tensor] = []
    for language in languages:
        harmful, harmless = cache.pair(language, split)
        harmful_chunks.append(harmful)
        harmless_chunks.append(harmless)
    return contrast_direction(torch.cat(harmful_chunks, dim=0), torch.cat(harmless_chunks, dim=0))


def source_pair(
    cache: ActivationCache,
    languages: list[str],
    split: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    return stack_pairs(cache.pair(language, split) for language in languages)


def source_train_logit_scores(logits: torch.Tensor, labels: torch.Tensor) -> ScoreBundle:
    labels = labels.to(dtype=torch.float32, device=logits.device)
    return ScoreBundle(logits[labels > 0.5], logits[labels <= 0.5])


def source_train_direction_scores(
    cache: ActivationCache,
    src: list[str],
    split: str,
    direction: torch.Tensor,
) -> ScoreBundle:
    harmful_scores: list[torch.Tensor] = []
    harmless_scores: list[torch.Tensor] = []
    for language in src:
        scores = direction_scores(direction, *cache.pair(language, split))
        harmful_scores.append(scores.harmful)
        harmless_scores.append(scores.harmless)
    return ScoreBundle(torch.cat(harmful_scores, dim=0), torch.cat(harmless_scores, dim=0))


def split_logits_by_split(
    logits: torch.Tensor,
    spec: list[tuple[str, str, int, int]],
) -> dict[str, dict[str, ScoreBundle]]:
    out: dict[str, dict[str, ScoreBundle]] = {}
    offset = 0
    for split, language, n_harmful, n_harmless in spec:
        harmful = logits[offset : offset + n_harmful]
        offset += n_harmful
        harmless = logits[offset : offset + n_harmless]
        offset += n_harmless
        out.setdefault(split, {})[language] = ScoreBundle(harmful, harmless)
    return out


def append_method(
    rows: list[dict[str, object]],
    *,
    metadata: dict[str, object],
    target_tiers: dict[str, str],
    validation_scores: dict[str, ScoreBundle],
    test_scores: dict[str, ScoreBundle],
    threshold_scores: dict[str, ScoreBundle] | None = None,
    threshold_source: str = "validation",
    include_oracle_threshold: bool = False,
    threshold_scope: str = "global",
) -> None:
    append_evaluation_rows(
        rows,
        metadata=metadata,
        target_tiers=target_tiers,
        validation_scores=validation_scores,
        test_scores=test_scores,
        threshold_scores=threshold_scores,
        threshold_source=threshold_source,
        threshold_scope=threshold_scope,
    )
    if include_oracle_threshold:
        append_evaluation_rows(
            rows,
            metadata={**metadata, "method": f"{metadata['method']}_threshold_oracle"},
            target_tiers=target_tiers,
            validation_scores=validation_scores,
            test_scores=test_scores,
            threshold_scores=threshold_scores,
            threshold_source=threshold_source,
            threshold_scope="language_oracle",
        )


def build_mlp(dim: int, hidden_dims: list[int], dropout: float) -> torch.nn.Sequential:
    layers: list[torch.nn.Module] = []
    in_dim = int(dim)
    for hidden_dim in hidden_dims:
        layers.append(torch.nn.Linear(in_dim, int(hidden_dim)))
        layers.append(torch.nn.ReLU())
        if float(dropout) > 0:
            layers.append(torch.nn.Dropout(float(dropout)))
        in_dim = int(hidden_dim)
    layers.append(torch.nn.Linear(in_dim, 1))
    model = torch.nn.Sequential(*layers)
    for layer in model:
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(layer.weight)
            torch.nn.init.zeros_(layer.bias)
    return model


def train_mlp_model(
    model: torch.nn.Module,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    *,
    seed: int,
    l2: float,
    lr: float,
    epochs: int,
    batch_size: int,
    balanced_loss: bool,
) -> torch.nn.Module:
    train_y = train_y.to(device=train_x.device, dtype=torch.float32)
    model.train()

    pos_weight = None
    if balanced_loss:
        n_pos = train_y.sum().clamp(min=1.0)
        n_neg = (train_y.numel() - train_y.sum()).clamp(min=1.0)
        pos_weight = torch.tensor([float(n_neg / n_pos)], dtype=torch.float32, device=train_x.device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(l2))
    n = train_x.shape[0]
    effective_batch = n if int(batch_size) <= 0 else min(int(batch_size), n)
    gen = torch.Generator().manual_seed(int(seed))
    for epoch in range(int(epochs)):
        order = torch.randperm(n, generator=gen)
        if train_x.device.type != "cpu":
            order = order.to(train_x.device)
        for start in range(0, n, effective_batch):
            idx = order[start : start + effective_batch]
            opt.zero_grad(set_to_none=True)
            logits = model(train_x[idx]).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, train_y[idx], pos_weight=pos_weight)
            loss.backward()
            opt.step()
    model.eval()
    return model


def fit_mlp_probe(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    *,
    seed: int,
    hidden_dims: list[int],
    dropout: float,
    l2: float,
    lr: float,
    epochs: int,
    batch_size: int,
    standardize: bool,
    balanced_loss: bool,
    initial_probe: MlpProbe | None = None,
) -> MlpProbe:
    set_seed(seed)
    train_x = train_x.to(torch.float32)
    if initial_probe is None:
        scaler = FeatureScaler.fit(train_x) if standardize else None
        model = build_mlp(train_x.shape[1], hidden_dims, dropout).to(train_x.device)
    else:
        scaler = initial_probe.scaler
        model = copy.deepcopy(initial_probe.model).to(train_x.device)
    if scaler is not None:
        train_x = scaler.transform(train_x)

    model = train_mlp_model(
        model,
        train_x,
        train_y,
        seed=seed,
        l2=l2,
        lr=lr,
        epochs=epochs,
        batch_size=batch_size,
        balanced_loss=balanced_loss,
    )
    return MlpProbe(model=model, scaler=scaler)


def mlp_logits(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    eval_x: torch.Tensor,
    *,
    seed: int,
    hidden_dims: list[int],
    dropout: float,
    l2: float,
    lr: float,
    epochs: int,
    batch_size: int,
    standardize: bool,
    balanced_loss: bool,
) -> torch.Tensor:
    probe = fit_mlp_probe(
        train_x,
        train_y,
        seed=seed,
        hidden_dims=hidden_dims,
        dropout=dropout,
        l2=l2,
        lr=lr,
        epochs=epochs,
        batch_size=batch_size,
        standardize=standardize,
        balanced_loss=balanced_loss,
    )
    with torch.no_grad():
        return probe.logits(eval_x).to(torch.float32)


def fit_logistic_probe(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    *,
    seed: int,
    l2: float,
    lr: float,
    epochs: int,
    standardize: bool,
    balanced_loss: bool,
) -> LogisticProbe:
    set_seed(seed)
    train_x = train_x.to(torch.float32)
    train_y = train_y.to(device=train_x.device, dtype=torch.float32)
    scaler = FeatureScaler.fit(train_x) if standardize else None
    if scaler is not None:
        train_x = scaler.transform(train_x)

    model = torch.nn.Linear(train_x.shape[1], 1).to(train_x.device)
    torch.nn.init.xavier_uniform_(model.weight)
    torch.nn.init.zeros_(model.bias)

    pos_weight = None
    if balanced_loss:
        n_pos = train_y.sum().clamp(min=1.0)
        n_neg = (train_y.numel() - train_y.sum()).clamp(min=1.0)
        pos_weight = torch.tensor([float(n_neg / n_pos)], dtype=torch.float32, device=train_x.device)

    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(l2))
    for epoch in range(int(epochs)):
        opt.zero_grad(set_to_none=True)
        logits = model(train_x).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, train_y, pos_weight=pos_weight)
        loss.backward()
        opt.step()

    with torch.no_grad():
        return LogisticProbe(
            weight=model.weight.detach().flatten().to(torch.float32),
            bias=model.bias.detach().reshape(()).to(torch.float32),
            scaler=scaler,
        )


def logistic_logits(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    eval_x: torch.Tensor,
    *,
    seed: int,
    l2: float,
    lr: float,
    epochs: int,
    standardize: bool,
    balanced_loss: bool,
) -> torch.Tensor:
    probe = fit_logistic_probe(
        train_x,
        train_y,
        seed=seed,
        l2=l2,
        lr=lr,
        epochs=epochs,
        standardize=standardize,
        balanced_loss=balanced_loss,
    )
    with torch.no_grad():
        return probe.logits(eval_x).to(torch.float32)
