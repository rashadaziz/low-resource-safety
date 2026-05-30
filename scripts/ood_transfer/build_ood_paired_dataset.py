"""Build PolyRefuse-compatible paired OOD safety datasets."""

import csv
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import hydra
import openpyxl
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf

from multilingual_latent_safety.csv_io import write_rows
from multilingual_latent_safety.json_io import read_jsonl

INSTRUCTION_COLUMNS = (
    "instruction",
    "prompt",
    "question",
    "query",
    "input",
    "text",
    "unsafe_prompt",
    "harmful_prompt",
)

INDOSAFETY_CATEGORY_COLUMNS = ("risk_area", "types_of_harm", "specific_harms")

OOD_ROW_FIELDS = [
    "dataset",
    "language",
    "pseudo_language",
    "split",
    "label",
    "instruction",
    "source_id",
    "paired_harmless_language",
    "harmful_source_split",
    "harmless_target_language",
    "harmless_translation_language",
    "harmless_source_language",
    "category",
    *INDOSAFETY_CATEGORY_COLUMNS,
]


@dataclass(frozen=True)
class OodInstruction:
    dataset: str
    language: str
    instruction: str
    source_id: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class LanguageSpec:
    code: str
    aliases: tuple[str, ...]
    harmless_language: str | None = None


def normalize_language_specs(raw_specs: Iterable[object]) -> list[LanguageSpec]:
    specs: list[LanguageSpec] = []
    for raw in raw_specs:
        if isinstance(raw, str):
            specs.append(LanguageSpec(code=raw, aliases=(raw,)))
            continue
        code = str(raw["code"])
        aliases = tuple(str(alias).lower() for alias in list(raw.get("aliases") or [code]))
        harmless_language = raw.get("harmless_language")
        specs.append(
            LanguageSpec(
                code=code,
                aliases=aliases,
                harmless_language=None if harmless_language is None else str(harmless_language),
            )
        )
    return specs


def harmless_language_for(
    language: str,
    available_languages: Iterable[str],
    fallback_language: str,
    override: str | None = None,
) -> str:
    available = set(str(item) for item in available_languages)
    if override is not None:
        return override
    if language in available:
        return language
    return fallback_language


def harmless_roots(polyrefuse_root: Path, translation_roots: Iterable[Path]) -> list[Path]:
    roots: list[Path] = []
    for root in translation_roots:
        if root not in roots:
            roots.append(root)
    if polyrefuse_root not in roots:
        roots.append(polyrefuse_root)
    return roots


def discover_harmless_languages(roots: Iterable[Path]) -> list[str]:
    languages: set[str] = set()
    prefix = "harmless_"
    marker = "_translated_"
    suffix = ".json"
    for root in roots:
        if not root.exists():
            continue
        for path in root.glob("harmless_*_translated_*.json"):
            name = path.name
            if not name.startswith(prefix) or marker not in name or not name.endswith(suffix):
                continue
            languages.add(name.rsplit(marker, 1)[1][: -len(suffix)])
    return sorted(languages)


def _read_csv_records(path: Path) -> list[dict[str, object]]:
    with path.open(newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _read_json_records(path: Path) -> list[dict[str, object]]:
    with path.open() as f:
        data = json.load(f)
    if isinstance(data, list):
        return [record for record in data if isinstance(record, dict)]
    if isinstance(data, dict):
        for key in ("data", "examples", "rows", "dataset"):
            value = data.get(key)
            if isinstance(value, list):
                return [record for record in value if isinstance(record, dict)]
    return []


def _read_xlsx_records(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    for sheet in workbook.worksheets:
        rows = sheet.iter_rows(values_only=True)
        header = next(rows, None)
        if header is None:
            continue
        columns = [str(value).strip() if value is not None else "" for value in header]
        for values in rows:
            record = {
                column: value
                for column, value in zip(columns, values)
                if column and value is not None
            }
            if record:
                records.append(record)
    return records


def _read_records(path: Path) -> list[dict[str, object]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _read_csv_records(path)
    if suffix == ".json":
        return _read_json_records(path)
    if suffix == ".jsonl":
        return read_jsonl(path)
    if suffix == ".xlsx":
        return _read_xlsx_records(path)
    return []


def _instruction_from_record(record: dict[str, object], columns: Iterable[str]) -> str | None:
    for column in columns:
        value = record.get(column)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _metadata_from_record(
    record: dict[str, object],
    columns: Iterable[str],
) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for column in columns:
        value = record.get(str(column))
        if value is None:
            continue
        text = str(value).strip()
        if text:
            metadata[str(column)] = text
    return metadata


def _matches_language(record: dict[str, object], path: Path, spec: LanguageSpec) -> bool:
    haystack = " ".join(
        [
            path.stem.lower(),
            str(record.get("language", "")).lower(),
            str(record.get("lang", "")).lower(),
            str(record.get("locale", "")).lower(),
            str(record.get("dialect", "")).lower(),
        ]
    )
    return any(alias and alias in haystack for alias in spec.aliases)


def _normalize_name(value: object) -> str:
    return str(value).strip().lower().replace("_", "-")


def _indosafety_instruction_for(record: dict[str, object], path: Path, spec: LanguageSpec) -> str | None:
    aliases = {_normalize_name(alias) for alias in spec.aliases}
    aliases.add(_normalize_name(spec.code))
    for column, value in record.items():
        column_name = _normalize_name(column)
        if column_name in aliases and isinstance(value, str) and value.strip():
            return value.strip()
    if spec.code == "id":
        value = record.get("prompt")
        if isinstance(value, str) and value.strip():
            return value.strip()
    if _matches_language(record, path, spec):
        return _instruction_from_record(record, INSTRUCTION_COLUMNS)
    return None


def _indosafety_file_split(path: Path) -> str:
    return "train" if "train" in path.stem.lower() else "test"


def load_indosafety_harmful(
    source_root: Path,
    language_specs: list[LanguageSpec],
    *,
    instruction_columns: Iterable[str] = INSTRUCTION_COLUMNS,
    category_columns: Iterable[str] = INDOSAFETY_CATEGORY_COLUMNS,
) -> dict[str, list[OodInstruction]]:
    split_examples = load_indosafety_harmful_splits(
        source_root,
        language_specs,
        instruction_columns=instruction_columns,
        category_columns=category_columns,
    )
    return {
        code: splits["train"] + splits["test"]
        for code, splits in split_examples.items()
    }


def load_indosafety_harmful_splits(
    source_root: Path,
    language_specs: list[LanguageSpec],
    *,
    instruction_columns: Iterable[str] = INSTRUCTION_COLUMNS,
    category_columns: Iterable[str] = INDOSAFETY_CATEGORY_COLUMNS,
) -> dict[str, dict[str, list[OodInstruction]]]:
    examples = {spec.code: {"train": [], "test": []} for spec in language_specs}
    if not source_root.exists():
        return examples

    files = [
        path
        for path in source_root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".csv", ".json", ".jsonl", ".xlsx"}
    ]
    for path in sorted(files):
        split = _indosafety_file_split(path)
        for idx, record in enumerate(_read_records(path)):
            for spec in language_specs:
                if split == "train" and spec.code != "id":
                    continue
                instruction = _indosafety_instruction_for(record, path, spec)
                if instruction is None:
                    instruction = _instruction_from_record(record, instruction_columns)
                    if instruction is not None and not _matches_language(record, path, spec):
                        instruction = None
                if instruction is not None:
                    source_id = f"{path.relative_to(source_root)}:{split}:{idx}:{spec.code}"
                    metadata = _metadata_from_record(record, category_columns)
                    metadata["harmful_source_split"] = split
                    examples[spec.code][split].append(
                        OodInstruction("indosafety", spec.code, instruction, source_id, metadata)
                    )
    return examples


def load_multijail_harmful(language_specs: list[LanguageSpec]) -> dict[str, list[OodInstruction]]:
    dataset = load_dataset("DAMO-NLP-SG/MultiJail", split="train")
    examples = {spec.code: [] for spec in language_specs}
    for row_idx, row in enumerate(dataset):
        for spec in language_specs:
            value = row.get(spec.code)
            if isinstance(value, str) and value.strip():
                source_id = str(row.get("id", row_idx))
                examples[spec.code].append(
                    OodInstruction(
                        dataset="multijail",
                        language=spec.code,
                        instruction=value.strip(),
                        source_id=source_id,
                    )
                )
    return examples


def harmless_json_path(root: Path, split: str, language: str) -> Path:
    return root / f"harmless_{split}_translated_{language}.json"


def load_harmless_records(roots: list[Path], language: str, split: str) -> list[dict[str, object]]:
    for root in roots:
        path = harmless_json_path(root, split, language)
        if not path.exists():
            continue
        with path.open() as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"expected list in harmless translation file: {path}")
        return [
            record
            for record in data
            if isinstance(record, dict)
            and isinstance(record.get("instruction"), str)
            and str(record["instruction"]).strip()
        ]
    searched = ", ".join(str(harmless_json_path(root, split, language)) for root in roots)
    raise FileNotFoundError(f"missing harmless translation file for {language}/{split}: {searched}")


def harmless_metadata(record: dict[str, object], requested_language: str) -> dict[str, str]:
    target_language = str(record.get("target_language", requested_language))
    translation_language = str(record.get("translation_language", target_language))
    source_language = str(record.get("source_language", ""))
    metadata = {
        "harmless_target_language": target_language,
        "harmless_translation_language": translation_language,
    }
    if source_language:
        metadata["harmless_source_language"] = source_language
    if "category" in record and record["category"] is not None:
        metadata["category"] = str(record["category"])
    return metadata


def build_harmless_examples(
    *,
    roots: list[Path],
    language: str,
    fallback_language: str,
    available_languages: Iterable[str],
    split: str,
    count: int | None = None,
    seed: int,
    harmless_language_override: str | None = None,
) -> list[OodInstruction]:
    harmless_language = harmless_language_for(
        language,
        available_languages,
        fallback_language,
        override=harmless_language_override,
    )
    records = load_harmless_records(roots, harmless_language, split)
    if not records:
        raise ValueError(f"no harmless PolyRefuse examples for {harmless_language}")

    rng = random.Random(seed)
    rng.shuffle(records)
    total = len(records) if count is None else count
    return [
        OodInstruction(
            dataset="polyrefuse_harmless",
            language=str(records[idx % len(records)].get("target_language", harmless_language)),
            instruction=str(records[idx % len(records)]["instruction"]),
            source_id=str(
                records[idx % len(records)].get(
                    "source_id",
                    f"{harmless_language}:{split}:{idx % len(records)}",
                )
            ),
            metadata=harmless_metadata(records[idx % len(records)], harmless_language),
        )
        for idx in range(total)
    ]


def write_polyrefuse_json(path: Path, examples: list[OodInstruction], *, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(
            [
                {
                    "instruction": example.instruction,
                    "label": label,
                    "source_dataset": example.dataset,
                    "source_language": example.language,
                    "source_id": example.source_id,
                    **example.metadata,
                }
                for example in examples
            ],
            f,
            ensure_ascii=False,
            indent=2,
        )


def polyrefuse_compatible_path(root: Path, subset: str, split: str, language: str) -> Path:
    return root / f"{subset}_{split}_translated_{language}.json"


def write_language_cell(
    *,
    output_root: Path,
    dataset_name: str,
    language: str,
    pseudo_language: str,
    train_harmful: list[OodInstruction],
    test_harmful: list[OodInstruction],
    train_harmless: list[OodInstruction],
    test_harmless: list[OodInstruction],
) -> list[dict[str, object]]:
    write_polyrefuse_json(
        polyrefuse_compatible_path(output_root, "harmful", "train", pseudo_language),
        train_harmful,
        label="harmful",
    )
    write_polyrefuse_json(
        polyrefuse_compatible_path(output_root, "harmful", "test", pseudo_language),
        test_harmful,
        label="harmful",
    )
    write_polyrefuse_json(
        polyrefuse_compatible_path(output_root, "harmless", "train", pseudo_language),
        train_harmless,
        label="harmless",
    )
    write_polyrefuse_json(
        polyrefuse_compatible_path(output_root, "harmless", "test", pseudo_language),
        test_harmless,
        label="harmless",
    )

    rows: list[dict[str, object]] = []
    for split, label, examples in (
        ("train", "harmful", train_harmful),
        ("test", "harmful", test_harmful),
        ("train", "harmless", train_harmless),
        ("test", "harmless", test_harmless),
    ):
        for example in examples:
            rows.append(
                {
                    "dataset": dataset_name,
                    "language": language,
                    "pseudo_language": pseudo_language,
                    "split": split,
                    "label": label,
                    "instruction": example.instruction,
                    "source_id": example.source_id,
                    "paired_harmless_language": example.language if label == "harmless" else "",
                    **example.metadata,
                }
            )
    return rows


def run_build_ood_paired_dataset(cfg: DictConfig) -> None:
    output_root = Path(cfg.output_root)
    rows: list[dict[str, object]] = []
    pseudo_languages: list[str] = []
    fallback_language = str(cfg.fallback_harmless_language)
    polyrefuse_root = Path(cfg.dataset.root)
    translation_root_cfg = cfg.get(
        "harmless_translation_roots",
        cfg.get("harmless_translation_root", []),
    )
    if translation_root_cfg is None:
        translation_roots = []
    elif isinstance(translation_root_cfg, str):
        translation_roots = [Path(translation_root_cfg)]
    else:
        translation_roots = [Path(root) for root in translation_root_cfg]
    roots = harmless_roots(polyrefuse_root, translation_roots)
    available_harmless_languages = discover_harmless_languages(roots)
    harmless_train_split = str(cfg.get("harmless_train_split", cfg.get("harmless_split", "train")))
    harmless_test_split = str(cfg.get("harmless_test_split", cfg.get("harmless_split", "test")))
    max_harmful = None if cfg.max_harmful_per_language is None else int(cfg.max_harmful_per_language)
    language_counts: dict[str, dict[str, int]] = {}

    if bool(cfg.multijail.enabled):
        specs = normalize_language_specs(cfg.multijail.languages)
        harmful_by_language = load_multijail_harmful(specs)
        for spec in specs:
            harmful = harmful_by_language[spec.code]
            if max_harmful is not None:
                harmful = harmful[:max_harmful]
            if not harmful:
                continue
            pseudo = f"multijail_{spec.code}"
            train_harmful: list[OodInstruction] = []
            test_harmful = harmful
            train_harmless = build_harmless_examples(
                roots=roots,
                language=spec.code,
                fallback_language=fallback_language,
                available_languages=available_harmless_languages,
                split=harmless_train_split,
                seed=int(cfg.seed),
                harmless_language_override=spec.harmless_language,
            )
            test_harmless = build_harmless_examples(
                roots=roots,
                language=spec.code,
                fallback_language=fallback_language,
                available_languages=available_harmless_languages,
                split=harmless_test_split,
                seed=int(cfg.seed) + 1,
                harmless_language_override=spec.harmless_language,
            )
            rows.extend(
                write_language_cell(
                    output_root=output_root,
                    dataset_name="multijail",
                    language=spec.code,
                    pseudo_language=pseudo,
                    train_harmful=train_harmful,
                    test_harmful=test_harmful,
                    train_harmless=train_harmless,
                    test_harmless=test_harmless,
                )
            )
            pseudo_languages.append(pseudo)
            language_counts[pseudo] = {
                "train_harmful": len(train_harmful),
                "test_harmful": len(test_harmful),
                "train_harmless": len(train_harmless),
                "test_harmless": len(test_harmless),
            }

    if bool(cfg.indosafety.enabled):
        specs = normalize_language_specs(cfg.indosafety.languages)
        source_root = Path(cfg.indosafety.source_root)
        harmful_by_language = load_indosafety_harmful_splits(
            source_root,
            specs,
            instruction_columns=list(cfg.indosafety.instruction_columns or INSTRUCTION_COLUMNS),
            category_columns=list(
                cfg.indosafety.get("category_columns") or INDOSAFETY_CATEGORY_COLUMNS
            ),
        )
        if not source_root.exists() and not bool(cfg.indosafety.optional):
            raise FileNotFoundError(f"IndoSafety source_root does not exist: {source_root}")
        for spec in specs:
            train_harmful = harmful_by_language[spec.code]["train"]
            test_harmful = harmful_by_language[spec.code]["test"]
            if max_harmful is not None:
                train_harmful = train_harmful[:max_harmful]
                test_harmful = test_harmful[:max_harmful]
            if not train_harmful and not test_harmful:
                if bool(cfg.indosafety.optional):
                    continue
                raise ValueError(f"no IndoSafety examples found for {spec.code}")
            pseudo = f"indosafety_{spec.code}"
            train_harmless = build_harmless_examples(
                roots=roots,
                language=spec.code,
                fallback_language=fallback_language,
                available_languages=available_harmless_languages,
                split=harmless_train_split,
                seed=int(cfg.seed),
                harmless_language_override=spec.harmless_language,
            )
            test_harmless = build_harmless_examples(
                roots=roots,
                language=spec.code,
                fallback_language=fallback_language,
                available_languages=available_harmless_languages,
                split=harmless_test_split,
                seed=int(cfg.seed) + 1,
                harmless_language_override=spec.harmless_language,
            )
            rows.extend(
                write_language_cell(
                    output_root=output_root,
                    dataset_name="indosafety",
                    language=spec.code,
                    pseudo_language=pseudo,
                    train_harmful=train_harmful,
                    test_harmful=test_harmful,
                    train_harmless=train_harmless,
                    test_harmless=test_harmless,
                )
            )
            pseudo_languages.append(pseudo)
            language_counts[pseudo] = {
                "train_harmful": len(train_harmful),
                "test_harmful": len(test_harmful),
                "train_harmless": len(train_harmless),
                "test_harmless": len(test_harmless),
            }

    write_rows(Path(cfg.rows_path), rows, OOD_ROW_FIELDS)
    metadata = {
        "languages": pseudo_languages,
        "splits": ["train", "test"],
        "subsets": ["harmful", "harmless"],
        "rows_path": str(cfg.rows_path),
        "harmless_roots": [str(root) for root in roots],
        "language_counts": language_counts,
    }
    metadata_path = output_root / "metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w") as f:
        json.dump(metadata, f, indent=2)
    config_path = output_root / "dataset_config.yaml"
    with config_path.open("w") as f:
        f.write(
            OmegaConf.to_yaml(
                {
                    "name": "ood_transfer",
                    "root": str(output_root),
                    "languages": pseudo_languages,
                    "splits": ["train", "test"],
                    "subsets": ["harmful", "harmless"],
                    "max_samples": None,
                    "shuffle": False,
                    "seed": int(cfg.seed),
                }
            )
        )
    print(f"[done] wrote {len(rows)} rows for {len(pseudo_languages)} OOD pseudo-languages")


@hydra.main(version_base=None, config_path="../../configs", config_name="ood_transfer/build_ood_paired_dataset")
def main(cfg: DictConfig) -> None:
    run_build_ood_paired_dataset(cfg)


if __name__ == "__main__":
    main()
