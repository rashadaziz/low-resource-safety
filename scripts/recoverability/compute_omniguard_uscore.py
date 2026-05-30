"""Compute OMNIGuard U-Score layer selection from mean-pooled representations."""


from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

from multilingual_latent_safety.omniguard import (
    load_representations,
    selected_layer_file,
    uscore_pair,
    write_csv,
    write_selected_layer,
    representation_dir,
)


@hydra.main(version_base=None, config_path="../../configs", config_name="recoverability/compute_omniguard_uscore")
def main(cfg: DictConfig) -> None:
    all_languages = [str(language) for language in cfg.dataset.languages if str(language) != str(cfg.anchor_language)]
    tiers = {str(language): str(tier) for language, tier in cfg.dataset.resource_tier.items()}
    selection_tiers = set(cfg.selection_tiers)
    languages = [language for language in all_languages if tiers[language] in selection_tiers]
    if not languages:
        raise ValueError(f"selection_tiers={sorted(selection_tiers)} matches no non-anchor languages")
    root = Path(cfg.representations_root)
    layers = resolve_available_layers(
        root,
        str(cfg.anchor_language),
        str(cfg.split),
        str(list(cfg.subsets)[0]),
        cfg.layers,
    )
    out_root = Path(cfg.output_root)
    rows: list[dict[str, object]] = []

    for layer in layers:
        layer_scores: list[float] = []
        selection_scores: list[float] = []
        tier_scores: dict[str, list[float]] = {"high": [], "mid": [], "low": []}
        for language in languages:
            subset_scores: list[float] = []
            subset_matched: list[float] = []
            subset_random: list[float] = []
            subset_n = 0
            for subset in cfg.subsets:
                anchor = load_representations(
                    root, str(cfg.anchor_language), str(cfg.split), str(subset), layer
                )
                target = load_representations(root, language, str(cfg.split), str(subset), layer)
                score, matched, random_baseline, n = uscore_pair(anchor, target)
                subset_scores.append(score)
                subset_matched.append(matched)
                subset_random.append(random_baseline)
                subset_n += n
                rows.append(
                    {
                        "model": cfg.model.name,
                        "layer": layer,
                        "language": language,
                        "tier": tiers[language],
                        "subset": subset,
                        "uscore": score,
                        "matched_cosine": matched,
                        "random_cosine": random_baseline,
                        "n_pairs": n,
                    }
                )
            language_score = float(torch.tensor(subset_scores, dtype=torch.float32).mean().item())
            layer_scores.append(language_score)
            if tiers[language] in selection_tiers:
                selection_scores.append(language_score)
            tier_scores[tiers[language]].append(language_score)
            rows.append(
                {
                    "model": cfg.model.name,
                    "layer": layer,
                    "language": language,
                    "tier": tiers[language],
                    "subset": "__mean__",
                    "uscore": language_score,
                    "matched_cosine": float(torch.tensor(subset_matched).mean().item()),
                    "random_cosine": float(torch.tensor(subset_random).mean().item()),
                    "n_pairs": subset_n,
                }
            )
        macro = float(torch.tensor(layer_scores, dtype=torch.float32).mean().item())
        selection_macro = float(torch.tensor(selection_scores, dtype=torch.float32).mean().item())
        rows.append(
            {
                "model": cfg.model.name,
                "layer": layer,
                "language": "__macro__",
                "tier": "all",
                "subset": "__mean__",
                "uscore": macro,
                "selection_uscore": selection_macro,
                "selection_tiers": ",".join(sorted(selection_tiers)),
                "matched_cosine": "",
                "random_cosine": "",
                "n_pairs": "",
                "uscore_high": mean_or_empty(tier_scores["high"]),
                "uscore_mid": mean_or_empty(tier_scores["mid"]),
                "uscore_low": mean_or_empty(tier_scores["low"]),
            }
        )

    macro_rows = [row for row in rows if row["language"] == "__macro__"]
    selected_layer = int(max(macro_rows, key=lambda row: float(row["selection_uscore"]))["layer"])
    write_csv(out_root / "uscore_by_layer.csv", rows)
    write_selected_layer(
        selected_layer_file(out_root),
        model=str(cfg.model.name),
        selected_layer=selected_layer,
        rows=rows,
    )
    print(f"[done] selected layer {selected_layer}; wrote {out_root}")


def mean_or_empty(values: list[float]) -> float | str:
    if not values:
        return ""
    return float(torch.tensor(values, dtype=torch.float32).mean().item())


def resolve_available_layers(root: Path, language: str, split: str, subset: str, layers_cfg) -> list[int]:
    available = sorted(
        int(path.stem.split("_")[-1])
        for path in representation_dir(root, language, split, subset).glob("layer_*.safetensors")
    )
    if not available:
        raise FileNotFoundError(
            f"no OMNIGuard representations found under {representation_dir(root, language, split, subset)}"
        )
    if layers_cfg == "all":
        return available
    requested = [int(layer) for layer in layers_cfg]
    missing = sorted(set(requested) - set(available))
    if missing:
        raise FileNotFoundError(f"requested layers missing from representations: {missing}")
    return requested


if __name__ == "__main__":
    main()
