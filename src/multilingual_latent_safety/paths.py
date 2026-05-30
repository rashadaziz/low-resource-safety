"""Path builders for experiment artifacts."""

from pathlib import Path


def activations_dir(root: str | Path, language: str, split: str, subset: str) -> Path:
    return Path(root) / language / split / subset


def activations_file(root: str | Path, language: str, split: str, subset: str, layer: int) -> Path:
    return activations_dir(root, language, split, subset) / f"layer_{layer:03d}.safetensors"


def direction_dir(root: str | Path, language: str, split: str, token_position: str | int) -> Path:
    return Path(root) / language / split / f"tok={token_position}"


def direction_file(root: str | Path, language: str, split: str, token_position: str | int, layer: int) -> Path:
    return direction_dir(root, language, split, token_position) / f"layer_{layer:03d}.safetensors"


def cosine_dir(root: str | Path, split: str, token_position: str | int) -> Path:
    return Path(root) / f"split={split}" / f"tok={token_position}"


def cosine_file(root: str | Path, split: str, token_position: str | int, layer: int) -> Path:
    return cosine_dir(root, split, token_position) / f"layer_{layer:03d}.safetensors"


def projection_dir(
    root: str | Path, split: str, token_position: str | int, dir_language: str, test_language: str
) -> Path:
    return (
        Path(root)
        / f"split={split}"
        / f"tok={token_position}"
        / f"dir={dir_language}"
        / f"test={test_language}"
    )


def projection_file(
    root: str | Path,
    split: str,
    token_position: str | int,
    dir_language: str,
    test_language: str,
    layer: int,
) -> Path:
    return projection_dir(root, split, token_position, dir_language, test_language) / f"layer_{layer:03d}.safetensors"


def auc_dir(root: str | Path, split: str, token_position: str | int) -> Path:
    return Path(root) / f"split={split}" / f"tok={token_position}"


def auc_file(root: str | Path, split: str, token_position: str | int, layer: int) -> Path:
    return auc_dir(root, split, token_position) / f"layer_{layer:03d}.safetensors"


def auc_gap_dir(root: str | Path, split: str, token_position: str | int) -> Path:
    return Path(root) / f"split={split}" / f"tok={token_position}"


def auc_gap_file(root: str | Path, split: str, token_position: str | int, layer: int) -> Path:
    return auc_gap_dir(root, split, token_position) / f"layer_{layer:03d}.safetensors"


def orthogonal_mass_dir(root: str | Path, split: str, token_position: str | int) -> Path:
    return Path(root) / f"split={split}" / f"tok={token_position}"


def orthogonal_mass_file(root: str | Path, split: str, token_position: str | int, layer: int) -> Path:
    return orthogonal_mass_dir(root, split, token_position) / f"layer_{layer:03d}.safetensors"


def completions_dir(root: str | Path, language: str, split: str, subset: str) -> Path:
    return Path(root) / language / split


def completions_file(root: str | Path, language: str, split: str, subset: str) -> Path:
    return completions_dir(root, language, split, subset) / f"{subset}.jsonl"


def refusal_score_dir(root: str | Path, language: str, split: str, subset: str) -> Path:
    return Path(root) / language / split


def refusal_score_file(root: str | Path, language: str, split: str, subset: str) -> Path:
    return refusal_score_dir(root, language, split, subset) / f"{subset}.jsonl"


def intervention_completions_dir(
    root: str | Path,
    intervention_name: str,
    direction_source_lang: str,
    language: str,
    split: str,
    direction_layer: int | None = None,
) -> Path:
    base = Path(root) / f"iv={intervention_name}" / f"src={direction_source_lang}"
    if direction_layer is not None:
        base = base / f"L={direction_layer:03d}"
    return base / language / split


def intervention_completions_file(
    root: str | Path,
    intervention_name: str,
    direction_source_lang: str,
    language: str,
    split: str,
    subset: str,
    direction_layer: int | None = None,
) -> Path:
    return (
        intervention_completions_dir(
            root, intervention_name, direction_source_lang, language, split, direction_layer
        )
        / f"{subset}.jsonl"
    )


def pooled_direction_dir(root: str | Path, pool: str, token_position: str | int) -> Path:
    return Path(root) / f"pool={pool}" / f"tok={token_position}"


def pooled_direction_file(
    root: str | Path, pool: str, token_position: str | int, layer: int
) -> Path:
    return pooled_direction_dir(root, pool, token_position) / f"layer_{layer:03d}.safetensors"
