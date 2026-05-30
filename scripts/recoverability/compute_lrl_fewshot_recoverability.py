"""Few-shot stability test for LRL harmfulness readouts.

The experiment fixes the layer selected from HRL validation/coupling analyses,
draws n harmful and n harmless target-language train examples, fits simple
readouts, and evaluates held-out LRL AUC. The goal is stability evidence, not
model selection.
"""


from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from safetensors.torch import load_file

from multilingual_latent_safety.activation_store import activation_token_index
from multilingual_latent_safety.analysis import load_subset_activations
from multilingual_latent_safety.csv_io import write_rows
from multilingual_latent_safety.paths import pooled_direction_file
from multilingual_latent_safety.probes import (
    binary_labels,
    direction_auc,
    logistic_regression_scores,
    mean_difference_direction,
    mean_difference_from_labels,
    mlp_scores,
    random_unit_vector,
    sample_balanced_indices,
    score_auc_from_logits,
    shuffled_labels,
)
from multilingual_latent_safety.runtime import stable_seed


FIELDNAMES = [
    "model",
    "model_short",
    "layer",
    "token_position",
    "language",
    "n_per_class",
    "seed",
    "readout",
    "control",
    "auc",
]


def load_activation_pair(
    acts_root: Path,
    language: str,
    split: str,
    layer: int,
    token_position: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    tok = activation_token_index(acts_root, language, split, "harmful", layer, token_position)
    harmful = load_subset_activations(acts_root, language, split, "harmful", layer, tok)
    harmless = load_subset_activations(acts_root, language, split, "harmless", layer, tok)
    return harmful, harmless


def make_result_row(
    model: str,
    model_short: str,
    layer: int,
    token_position: str,
    language: str,
    n_per_class: int,
    seed: int,
    readout: str,
    control: str,
    auc_value: float,
) -> dict[str, object]:
    return {
        "model": model,
        "model_short": model_short,
        "layer": layer,
        "token_position": token_position,
        "language": language,
        "n_per_class": n_per_class,
        "seed": seed,
        "readout": readout,
        "control": control,
        "auc": f"{auc_value:.6f}",
    }


def evaluate_logistic_probe(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    harmful_test: torch.Tensor,
    harmless_test: torch.Tensor,
    seed: int,
    cfg: DictConfig,
) -> float:
    eval_x = torch.cat([harmful_test, harmless_test], dim=0)
    scores = logistic_regression_scores(
        train_x,
        train_y,
        eval_x,
        seed=seed,
        l2=float(cfg.l2),
        lr=float(cfg.lr),
        epochs=int(cfg.epochs),
        standardize=bool(cfg.standardize),
    )
    return score_auc_from_logits(scores[: harmful_test.shape[0]], scores[harmful_test.shape[0] :])


def evaluate_mlp_probe(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    harmful_test: torch.Tensor,
    harmless_test: torch.Tensor,
    seed: int,
    cfg: DictConfig,
) -> float:
    eval_x = torch.cat([harmful_test, harmless_test], dim=0)
    scores = mlp_scores(
        train_x,
        train_y,
        eval_x,
        seed=seed,
        hidden_dim=int(cfg.hidden_dim),
        dropout=float(cfg.dropout),
        l2=float(cfg.l2),
        lr=float(cfg.lr),
        epochs=int(cfg.epochs),
        standardize=bool(cfg.standardize),
    )
    return score_auc_from_logits(scores[: harmful_test.shape[0]], scores[harmful_test.shape[0] :])


def run_language(
    *,
    model: str,
    model_short: str,
    layer: int,
    language: str,
    cfg: DictConfig,
) -> list[dict[str, object]]:
    token_position = str(cfg.token_position)
    device_name = str(cfg.device)
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else device_name)
    acts_root = Path(str(cfg.activations_root_template).format(model=model))
    pooled_root = Path(str(cfg.pooled_direction_root_template).format(model=model))

    harmful_train, harmless_train = load_activation_pair(
        acts_root, language, str(cfg.train_split), layer, token_position
    )
    harmful_test, harmless_test = load_activation_pair(
        acts_root, language, str(cfg.test_split), layer, token_position
    )
    harmful_train = harmful_train.to(device)
    harmless_train = harmless_train.to(device)
    harmful_test = harmful_test.to(device)
    harmless_test = harmless_test.to(device)

    rows: list[dict[str, object]] = []
    seeds = [int(cfg.seed_offset) + i for i in range(int(cfg.seeds))]
    budgets = [int(n) for n in cfg.budgets]
    x_axis = [0, *budgets]

    if cfg.include_hrl_direction:
        hrl_direction = load_file(pooled_direction_file(pooled_root, "hrl", token_position, layer))[
            "direction"
        ].to(device=device, dtype=torch.float32)
        hrl_auc = direction_auc(hrl_direction, harmful_test, harmless_test)
        for seed in seeds:
            for n_per_class in x_axis:
                rows.append(
                    make_result_row(
                        model,
                        model_short,
                        layer,
                        token_position,
                        language,
                        n_per_class,
                        seed,
                        "hrl_direction",
                        "none",
                        hrl_auc,
                    )
                )

    if cfg.include_full_local:
        full_direction = mean_difference_direction(
            harmful_train,
            harmless_train,
            normalize=bool(cfg.probe.normalize_mean_direction),
        )
        full_auc = direction_auc(full_direction, harmful_test, harmless_test)
        for seed in seeds:
            for n_per_class in x_axis:
                rows.append(
                    make_result_row(
                        model,
                        model_short,
                        layer,
                        token_position,
                        language,
                        n_per_class,
                        seed,
                        "full_local_mean_difference",
                        "reference",
                        full_auc,
                    )
                )

    for n_per_class in budgets:
        for seed in seeds:
            draw_seed = stable_seed(seed, model, language, n_per_class)
            h_idx, l_idx = sample_balanced_indices(
                harmful_train.shape[0],
                harmless_train.shape[0],
                n_per_class,
                draw_seed,
            )
            h_idx = h_idx.to(device)
            l_idx = l_idx.to(device)
            h_train = harmful_train[h_idx]
            l_train = harmless_train[l_idx]
            train_x = torch.cat([h_train, l_train], dim=0)
            train_y = binary_labels(h_train.shape[0], l_train.shape[0])

            local_direction = mean_difference_direction(
                h_train,
                l_train,
                normalize=bool(cfg.probe.normalize_mean_direction),
            )
            rows.append(
                make_result_row(
                    model,
                    model_short,
                    layer,
                    token_position,
                    language,
                    n_per_class,
                    seed,
                    "local_mean_difference",
                    "none",
                    direction_auc(local_direction, harmful_test, harmless_test),
                )
            )

            rows.append(
                make_result_row(
                    model,
                    model_short,
                    layer,
                    token_position,
                    language,
                    n_per_class,
                    seed,
                    "logistic_regression",
                    "none",
                    evaluate_logistic_probe(
                        train_x,
                        train_y,
                        harmful_test,
                        harmless_test,
                        draw_seed,
                        cfg.probe.logistic,
                    ),
                )
            )

            rows.append(
                make_result_row(
                    model,
                    model_short,
                    layer,
                    token_position,
                    language,
                    n_per_class,
                    seed,
                    "mlp",
                    "none",
                    evaluate_mlp_probe(
                        train_x,
                        train_y,
                        harmful_test,
                        harmless_test,
                        draw_seed,
                        cfg.probe.mlp,
                    ),
                )
            )

            if cfg.include_permutation_control:
                perm_y = shuffled_labels(
                    train_y,
                    stable_seed(seed, model, language, n_per_class, "perm"),
                )
                perm_direction = mean_difference_from_labels(
                    train_x,
                    perm_y,
                    normalize=bool(cfg.probe.normalize_mean_direction),
                )
                rows.append(
                    make_result_row(
                        model,
                        model_short,
                        layer,
                        token_position,
                        language,
                        n_per_class,
                        seed,
                        "permutation_mean_difference",
                        "permutation_label",
                        direction_auc(perm_direction, harmful_test, harmless_test),
                    )
                )

            if cfg.include_random_direction_control:
                direction = random_unit_vector(
                    harmful_test.shape[1],
                    stable_seed(seed, model, language, n_per_class, "random_direction"),
                ).to(device)
                rows.append(
                    make_result_row(
                        model,
                        model_short,
                        layer,
                        token_position,
                        language,
                        n_per_class,
                        seed,
                        "random_direction",
                        "random_direction",
                        direction_auc(direction, harmful_test, harmless_test),
                    )
                )

    return rows


def run_lrl_fewshot_recoverability(cfg: DictConfig) -> None:
    for model_cfg in cfg.models:
        model = str(model_cfg.name)
        model_short = str(model_cfg.short)
        layer = int(model_cfg.layer)
        model_rows: list[dict[str, object]] = []
        for language in cfg.languages:
            rows = run_language(
                model=model,
                model_short=model_short,
                layer=layer,
                language=str(language),
                cfg=cfg,
            )
            model_rows.extend(rows)
            print(f"[done] {model_short}/{language}: {len(rows)} rows")
        out = Path(cfg.output_root) / model / "fewshot_curves.csv"
        write_rows(out, model_rows, FIELDNAMES)
        print(f"[write] {out} ({len(model_rows)} rows)")


@hydra.main(version_base=None, config_path="../../configs", config_name="recoverability/compute_lrl_fewshot_recoverability")
def main(cfg: DictConfig) -> None:
    run_lrl_fewshot_recoverability(cfg)


if __name__ == "__main__":
    main()
