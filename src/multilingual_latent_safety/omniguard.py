
"""OMNIGuard representation loading and U-Score helpers."""

import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from multilingual_latent_safety.csv_io import write_rows


def representation_dir(root: str | Path, language: str, split: str, subset: str) -> Path:
    return Path(root) / language / split / subset


def representation_file(root: str | Path, language: str, split: str, subset: str, layer: int) -> Path:
    return representation_dir(root, language, split, subset) / f"layer_{layer:03d}.safetensors"


def selected_layer_file(root: str | Path) -> Path:
    return Path(root) / "selected_layer.json"


def load_representations(
    root: str | Path,
    language: str,
    split: str,
    subset: str,
    layer: int,
) -> torch.Tensor:
    return load_file(representation_file(root, language, split, subset, layer))["representations"].to(torch.float32)


def cosine_rows(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = torch.nn.functional.normalize(a.to(torch.float32), dim=1)
    b = torch.nn.functional.normalize(b.to(torch.float32), dim=1)
    return (a * b).sum(dim=1)


def uscore_pair(source: torch.Tensor, target: torch.Tensor) -> tuple[float, float, float, int]:
    n = min(source.shape[0], target.shape[0])
    if n < 2:
        raise ValueError("U-Score needs at least two aligned examples")
    source = source[:n].to(torch.float32)
    target = target[:n].to(torch.float32)
    source = torch.nn.functional.normalize(source, dim=1)
    target = torch.nn.functional.normalize(target, dim=1)
    sim = source @ target.T
    matched = float(sim.diag().mean().item())
    random_baseline = float((sim.sum() - sim.diag().sum()).item() / (n * (n - 1)))
    return matched - random_baseline, matched, random_baseline, n


def write_csv(path: str | Path, rows: list[dict[str, object]]) -> None:
    write_rows(path, rows)


def read_selected_layer(path: str | Path) -> int:
    with Path(path).open() as f:
        data = json.load(f)
    return int(data["selected_layer"])


def write_selected_layer(path: str | Path, *, model: str, selected_layer: int, rows: list[dict[str, object]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    selected = [row for row in rows if int(row["layer"]) == int(selected_layer) and row["language"] == "__macro__"]
    payload = {
        "model": model,
        "selected_layer": int(selected_layer),
        "selection_metric": "selection_uscore",
    }
    if selected:
        payload["macro_uscore_all_tiers"] = float(selected[0]["uscore"])
        payload["selection_uscore"] = float(selected[0].get("selection_uscore", selected[0]["uscore"]))
        payload["selection_tiers"] = selected[0].get("selection_tiers", "")
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
