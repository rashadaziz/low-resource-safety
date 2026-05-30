"""Small probing utilities for hidden-state harmfulness readouts."""


from dataclasses import dataclass

import torch
import torch.nn.functional as F

from multilingual_latent_safety.analysis import auc, projected_scores
from multilingual_latent_safety.runtime import set_seed


@dataclass(frozen=True)
class FeatureScaler:
    """Per-feature standardizer fit on the probe train examples."""

    mean: torch.Tensor
    scale: torch.Tensor

    @classmethod
    def fit(cls, x: torch.Tensor, eps: float = 1e-6) -> "FeatureScaler":
        x = x.to(torch.float32)
        mean = x.mean(dim=0, keepdim=True)
        scale = x.std(dim=0, unbiased=False, keepdim=True).clamp(min=eps)
        return cls(mean=mean, scale=scale)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return (x.to(torch.float32) - self.mean) / self.scale


def binary_labels(n_positive: int, n_negative: int) -> torch.Tensor:
    return torch.cat(
        [
            torch.ones(n_positive, dtype=torch.float32),
            torch.zeros(n_negative, dtype=torch.float32),
        ]
    )


def sample_balanced_indices(
    n_harmful: int,
    n_harmless: int,
    n_per_class: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw ``n`` harmful and ``n`` harmless indices without replacement."""

    if n_per_class < 1:
        raise ValueError(f"n_per_class must be positive, got {n_per_class}")
    if n_per_class > n_harmful or n_per_class > n_harmless:
        raise ValueError(
            f"n_per_class={n_per_class} exceeds class sizes "
            f"harmful={n_harmful}, harmless={n_harmless}"
        )
    g = torch.Generator().manual_seed(int(seed))
    harmful_idx = torch.randperm(n_harmful, generator=g)[:n_per_class]
    harmless_idx = torch.randperm(n_harmless, generator=g)[:n_per_class]
    return harmful_idx, harmless_idx


def mean_difference_direction(
    harmful: torch.Tensor,
    harmless: torch.Tensor,
    normalize: bool = True,
) -> torch.Tensor:
    """Return ``mu_harmful - mu_harmless`` with the positive side meaning harmful."""

    direction = harmful.to(torch.float32).mean(dim=0) - harmless.to(torch.float32).mean(dim=0)
    if normalize:
        direction = direction / direction.norm().clamp(min=1e-8)
    return direction


def mean_difference_from_labels(
    x: torch.Tensor,
    y: torch.Tensor,
    normalize: bool = True,
) -> torch.Tensor:
    y_bool = y.to(device=x.device, dtype=torch.bool)
    if y_bool.sum() == 0 or (~y_bool).sum() == 0:
        raise ValueError("mean-difference probe needs both classes")
    return mean_difference_direction(x[y_bool], x[~y_bool], normalize=normalize)


def random_unit_vector(dim: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(int(seed))
    vector = torch.randn(dim, generator=g, dtype=torch.float32)
    return vector / vector.norm().clamp(min=1e-8)


def shuffled_labels(y: torch.Tensor, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(int(seed))
    return y[torch.randperm(y.numel(), generator=g)]


def direction_auc(direction: torch.Tensor, harmful: torch.Tensor, harmless: torch.Tensor) -> float:
    direction = direction.to(device=harmful.device, dtype=torch.float32)
    return auc(projected_scores(harmful, direction), projected_scores(harmless, direction))


def standardized_train_eval(
    train_x: torch.Tensor,
    eval_x: torch.Tensor,
    standardize: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    train_x = train_x.to(torch.float32)
    eval_x = eval_x.to(torch.float32)
    if not standardize:
        return train_x, eval_x
    scaler = FeatureScaler.fit(train_x)
    return scaler.transform(train_x), scaler.transform(eval_x)


def logistic_regression_scores(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    eval_x: torch.Tensor,
    seed: int,
    l2: float = 1e-2,
    lr: float = 5e-2,
    epochs: int = 250,
    standardize: bool = True,
) -> torch.Tensor:
    """Fit an L2-regularized logistic probe and return logits on ``eval_x``."""

    set_seed(seed)
    x_train, x_eval = standardized_train_eval(train_x, eval_x, standardize=standardize)
    y = train_y.to(torch.float32)
    y = y.to(x_train.device)
    linear = torch.nn.Linear(x_train.shape[1], 1).to(x_train.device)
    torch.nn.init.zeros_(linear.weight)
    torch.nn.init.zeros_(linear.bias)
    opt = torch.optim.AdamW(linear.parameters(), lr=lr, weight_decay=0.0)
    for epoch in range(int(epochs)):
        opt.zero_grad(set_to_none=True)
        logits = linear(x_train).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        loss = loss + float(l2) * linear.weight.square().sum()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return linear(x_eval).squeeze(-1).to(torch.float32)


class SmallMLP(torch.nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def mlp_scores(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    eval_x: torch.Tensor,
    seed: int,
    hidden_dim: int = 64,
    dropout: float = 0.1,
    l2: float = 1e-3,
    lr: float = 1e-3,
    epochs: int = 250,
    standardize: bool = True,
) -> torch.Tensor:
    """Fit a small nonlinear probe and return logits on ``eval_x``."""

    set_seed(seed)
    x_train, x_eval = standardized_train_eval(train_x, eval_x, standardize=standardize)
    y = train_y.to(torch.float32)
    y = y.to(x_train.device)
    model = SmallMLP(x_train.shape[1], int(hidden_dim), float(dropout)).to(x_train.device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=float(l2))
    for epoch in range(int(epochs)):
        opt.zero_grad(set_to_none=True)
        logits = model(x_train)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        return model(x_eval).to(torch.float32)


def score_auc_from_logits(
    harmful_scores: torch.Tensor,
    harmless_scores: torch.Tensor,
) -> float:
    return auc(harmful_scores.to(torch.float32), harmless_scores.to(torch.float32))


def seed_interval(values: list[float], alpha: float = 0.05) -> tuple[float, float, float]:
    """Return mean and percentile interval for seed-level scalar values."""

    if not values:
        raise ValueError("cannot summarize an empty value list")
    x = torch.tensor(values, dtype=torch.float64)
    lo = torch.quantile(x, alpha / 2).item()
    hi = torch.quantile(x, 1.0 - alpha / 2).item()
    return float(x.mean().item()), float(lo), float(hi)
