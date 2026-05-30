"""Utilities for scored refusal-completion artifacts."""

import json
from pathlib import Path


def refusal_values(path: str | Path) -> list[int]:
    """Read binary refusal labels from a scored JSONL file."""
    path = Path(path)
    values: list[int] = []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if "__meta__" in row:
                continue
            refusal = row.get("refusal")
            if refusal not in (0, 1):
                raise ValueError(f"expected refusal 0/1 in {path}, got {refusal!r}")
            values.append(int(refusal))
    return values


def refusal_rate(path: str | Path) -> tuple[float, int]:
    values = refusal_values(path)
    total = len(values)
    if total == 0:
        raise ValueError(f"no scored rows in {path}")
    return sum(values) / total, total


def refusal_rate_percent(path: str | Path) -> tuple[float, int]:
    """Refusal rate as a percentage, plus row count."""
    rate, total = refusal_rate(path)
    return 100.0 * rate, total
