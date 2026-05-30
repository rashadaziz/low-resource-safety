"""Small runtime helpers shared by experiment scripts."""

import hashlib
import random
from collections.abc import Iterator, Sequence
from typing import TypeVar

import numpy as np
import torch

T = TypeVar("T")


def stable_seed(*parts: object) -> int:
    text = "|".join(str(part) for part in parts)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def batched(seq: Sequence[T], n: int) -> Iterator[Sequence[T]]:
    """Yield fixed-size slices from a sequence."""
    for i in range(0, len(seq), n):
        yield seq[i : i + n]
