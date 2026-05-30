"""JSONL artifact helpers."""

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any


def read_json(path: str | Path) -> Any:
    """Read a JSON file."""
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, payload: Any, *, indent: int = 2) -> None:
    """Write a JSON file, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=indent, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read non-empty JSONL rows as dictionaries."""
    rows: list[dict[str, Any]] = []
    with Path(path).open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
    return rows


def write_jsonl(
    path: str | Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    ensure_ascii: bool = False,
) -> None:
    """Write dictionaries to JSONL, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(dict(row), ensure_ascii=ensure_ascii) + "\n")


def read_jsonl_with_meta(path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Read a JSONL artifact whose first row may be metadata."""
    rows = read_jsonl(path)
    if not rows:
        return {}, []
    first = rows[0]
    if "__meta__" in first:
        return dict(first["__meta__"]), rows[1:]
    if "completion" not in first:
        return dict(first), rows[1:]
    return {}, rows


def write_jsonl_with_meta(
    path: str | Path,
    meta: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    ensure_ascii: bool = False,
) -> None:
    """Write a metadata row followed by data rows."""
    write_jsonl(path, [{"__meta__": dict(meta)}, *rows], ensure_ascii=ensure_ascii)
