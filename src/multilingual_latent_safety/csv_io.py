"""CSV writing helpers for experiment result rows."""

import csv
from pathlib import Path


def read_rows(path: str | Path) -> list[dict[str, str]]:
    """Read CSV rows as dictionaries."""
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def csv_value(value: object) -> object:
    if isinstance(value, float):
        return f"{value:.8g}"
    return value


def write_rows(
    path: str | Path,
    rows: list[dict[str, object]],
    preferred_fields: list[str] | tuple[str, ...] = (),
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = set().union(*(row.keys() for row in rows)) if rows else set(preferred_fields)
    fieldnames = [field for field in preferred_fields if field in keys]
    fieldnames.extend(sorted(keys - set(fieldnames)))
    if not fieldnames:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key, "")) for key in fieldnames})
