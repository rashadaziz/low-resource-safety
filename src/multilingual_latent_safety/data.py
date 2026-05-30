"""Dataset loading helpers for PolyRefuse-style JSONL files."""

import json
import random
from pathlib import Path

POLYREFUSE_LANGS = ["ar", "de", "en", "es", "fr", "it", "ja", "ko", "nl", "pl", "ru", "th", "yo", "zh"]
POLYREFUSE_SPLITS = ["train", "val", "test"]
POLYREFUSE_SUBSETS = ["harmful", "harmless"]


def polyrefuse_path(root: str | Path, subset: str, split: str, language: str) -> Path:
    return Path(root) / f"{subset}_{split}_translated_{language}.json"


def load_polyrefuse(
    root: str | Path,
    subset: str,
    split: str,
    language: str,
    max_samples: int | None = None,
    shuffle: bool = False,
    seed: int = 0,
) -> list[str]:
    """Return PolyRefuse instructions for one (subset, split, language) cell, optionally shuffled and truncated."""
    path = polyrefuse_path(root, subset, split, language)
    with open(path) as f:
        data = json.load(f)
    instructions = [d["instruction"] for d in data]
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(instructions)
    if max_samples is not None:
        instructions = instructions[:max_samples]
    return instructions
