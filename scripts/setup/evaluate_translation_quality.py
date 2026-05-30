"""Wang-style translation-quality evaluation via back-translation.

This follows Wang et al. (Refusal Direction is Universal Across Safety-Aligned
Languages): back-translate translated prompts into English, then compare the
back-translations with the original English prompts using BLEU and SBERT cosine
similarity.
"""

import argparse
import asyncio
import csv
import json
from pathlib import Path

import sacrebleu
import torch
from transformers import AutoModel, AutoTokenizer

from translation_common import (
    load_instruction_items,
    translate_instruction_items,
    write_translated_items,
)


def discover_languages(
    root: Path,
    *,
    subset: str,
    split: str,
    source_language: str,
) -> list[str]:
    prefix = f"{subset}_{split}_translated_"
    languages: list[str] = []
    for path in sorted(root.glob(f"{prefix}*.json")):
        language = path.name.removeprefix(prefix).removesuffix(".json")
        if language != source_language:
            languages.append(language)
    return languages


def backtranslation_row_builder(
    *,
    language: str,
    source_language: str,
):
    def build_row(idx: int, item: dict, translated_instruction: str, dest_lang: str) -> dict:
        row = {
            "instruction": translated_instruction,
            "translated_instruction": item["instruction"],
            "source_language": language,
            "target_language": source_language,
            "translation_language": dest_lang,
            "source_id": item.get("source_id", str(idx)),
        }
        for key in (
            "category",
            "label",
            "source_instruction",
            "source_dataset",
            "risk_area",
            "types_of_harm",
            "specific_harms",
        ):
            if key in item:
                row[key] = item[key]
        return row

    return build_row


def load_or_create_backtranslations(
    *,
    translated_path: Path,
    backtranslation_path: Path,
    language: str,
    source_language: str,
    backtranslation_source_language: str,
    concurrency: int,
    overwrite: bool,
) -> list[dict]:
    if backtranslation_path.exists() and not overwrite:
        return load_instruction_items(backtranslation_path)

    items = load_instruction_items(translated_path)
    rows = translate_instruction_items(
        items,
        src_lang=backtranslation_source_language,
        target_language=source_language,
        dest_candidates=[source_language],
        concurrency=concurrency,
        desc=f"{language}->{source_language}",
        build_row=backtranslation_row_builder(
            language=language,
            source_language=source_language,
        ),
        show_progress=False,
    )
    translated = asyncio.run(rows)
    write_translated_items(backtranslation_path, translated)
    return translated


def corpus_bleu(candidates: list[str], references: list[str]) -> float:
    return float(sacrebleu.corpus_bleu(candidates, [references]).score)


def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    return (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


def load_sbert(model_name: str, device: str | None):
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(resolved_device)
    model.eval()
    return tokenizer, model, resolved_device


def encode_sentences(
    sentences: list[str],
    *,
    tokenizer,
    model,
    device: str,
    batch_size: int,
    max_length: int | None,
):
    embeddings = []
    with torch.no_grad():
        for start in range(0, len(sentences), batch_size):
            batch = sentences[start : start + batch_size]
            tokenizer_kwargs = {
                "padding": True,
                "truncation": True,
                "return_tensors": "pt",
            }
            if max_length is not None:
                tokenizer_kwargs["max_length"] = max_length
            encoded = tokenizer(
                batch,
                **tokenizer_kwargs,
            ).to(device)
            output = model(**encoded)
            pooled = mean_pool(output.last_hidden_state, encoded["attention_mask"])
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            embeddings.append(pooled.cpu())
    return torch.cat(embeddings, dim=0)


def sbert_similarity(candidates, references) -> float:
    return float((candidates * references).sum(dim=1).mean().item() * 100.0)


def source_ids(items: list[dict]) -> list[str]:
    return [str(item.get("source_id", str(idx))) for idx, item in enumerate(items)]


def validate_source_ids(
    *,
    language: str,
    expected: list[str],
    observed_items: list[dict],
) -> None:
    observed = source_ids(observed_items)
    if observed == expected:
        return
    for idx, (expected_id, observed_id) in enumerate(zip(expected, observed)):
        if expected_id != observed_id:
            raise ValueError(
                f"{language}: source_id mismatch at row {idx}: "
                f"expected {expected_id!r}, found {observed_id!r}"
            )
    raise ValueError(f"{language}: source_id mismatch")


def compute_sbert_candidate_embeddings(
    candidates: list[str],
    *,
    tokenizer,
    model,
    device: str,
    batch_size: int,
    max_length: int | None,
):
    candidate_embeddings = encode_sentences(
        candidates,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
    )
    return candidate_embeddings


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_subset_split(
    *,
    root: Path,
    subset: str,
    split: str,
    source_language: str,
    target_languages: list[str] | None,
    backtranslation_source_language: str,
    output_root: Path,
    concurrency: int,
    overwrite_backtranslations: bool,
    skip_sbert: bool,
    sbert_model_name: str,
    sbert_tokenizer,
    sbert_model,
    sbert_device: str | None,
    sbert_batch_size: int,
    sbert_max_length: int | None,
) -> list[dict]:
    reference_path = root / f"{subset}_{split}_translated_{source_language}.json"
    if not reference_path.exists():
        raise FileNotFoundError(reference_path)
    reference_items = load_instruction_items(reference_path)
    references = [str(item["instruction"]) for item in reference_items]
    reference_source_ids = source_ids(reference_items)

    languages = target_languages or discover_languages(
        root,
        subset=subset,
        split=split,
        source_language=source_language,
    )
    if not languages:
        raise ValueError("no target languages found")

    reference_embeddings = None
    if not skip_sbert:
        reference_embeddings = encode_sentences(
            references,
            tokenizer=sbert_tokenizer,
            model=sbert_model,
            device=sbert_device,
            batch_size=sbert_batch_size,
            max_length=sbert_max_length,
        )

    rows: list[dict] = []
    backtranslation_root = output_root / "backtranslations"
    for language in languages:
        translated_path = root / f"{subset}_{split}_translated_{language}.json"
        if not translated_path.exists():
            print(f"[miss] {translated_path}")
            continue
        backtranslation_path = (
            backtranslation_root / f"{subset}_{split}_{language}_to_{source_language}.json"
        )
        backtranslated_items = load_or_create_backtranslations(
            translated_path=translated_path,
            backtranslation_path=backtranslation_path,
            language=language,
            source_language=source_language,
            backtranslation_source_language=backtranslation_source_language,
            concurrency=concurrency,
            overwrite=overwrite_backtranslations,
        )
        if len(backtranslated_items) != len(references):
            raise ValueError(
                f"{language}: {len(backtranslated_items)} back-translations for "
                f"{len(references)} references"
            )
        validate_source_ids(
            language=language,
            expected=reference_source_ids,
            observed_items=backtranslated_items,
        )

        candidates = [str(item["instruction"]) for item in backtranslated_items]
        record = {
            "language": language,
            "subset": subset,
            "split": split,
            "n": len(candidates),
            "bleu": round(corpus_bleu(candidates, references), 4),
            "sbert": None,
            "reference_path": str(reference_path),
            "translated_path": str(translated_path),
            "backtranslation_path": str(backtranslation_path),
            "sbert_model": None if skip_sbert else sbert_model_name,
        }
        if not skip_sbert:
            candidate_embeddings = compute_sbert_candidate_embeddings(
                candidates,
                tokenizer=sbert_tokenizer,
                model=sbert_model,
                device=sbert_device,
                batch_size=sbert_batch_size,
                max_length=sbert_max_length,
            )
            record["sbert"] = round(
                sbert_similarity(candidate_embeddings, reference_embeddings),
                4,
            )
        rows.append(record)
        sbert_text = "NA" if record["sbert"] is None else f"{record['sbert']:.2f}"
        print(
            f"{subset}/{split}/{language}\t"
            f"BLEU={record['bleu']:.2f}\tSBERT={sbert_text}\tn={record['n']}"
        )

    if not rows:
        raise ValueError(f"no metrics computed for {subset}/{split}")

    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / f"{subset}_{split}_wang_metrics.json"
    csv_path = output_root / f"{subset}_{split}_wang_metrics.csv"
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
    write_csv(csv_path, rows)
    print(f"[write] {json_path}")
    print(f"[write] {csv_path}")
    return rows


def evaluate_aggregate_splits(
    *,
    root: Path,
    subset: str,
    splits: list[str],
    source_language: str,
    target_languages: list[str] | None,
    output_root: Path,
    skip_sbert: bool,
    sbert_model_name: str,
    sbert_tokenizer,
    sbert_model,
    sbert_device: str | None,
    sbert_batch_size: int,
    sbert_max_length: int | None,
) -> list[dict]:
    languages = target_languages
    if languages is None:
        discovered: set[str] = set()
        for split in splits:
            discovered.update(
                discover_languages(
                    root,
                    subset=subset,
                    split=split,
                    source_language=source_language,
                )
            )
        languages = sorted(discovered)

    split_references: dict[str, list[dict]] = {}
    references: list[str] = []
    for split in splits:
        reference_path = root / f"{subset}_{split}_translated_{source_language}.json"
        reference_items = load_instruction_items(reference_path)
        split_references[split] = reference_items
        references.extend(str(item["instruction"]) for item in reference_items)

    reference_embeddings = None
    if not skip_sbert:
        reference_embeddings = encode_sentences(
            references,
            tokenizer=sbert_tokenizer,
            model=sbert_model,
            device=sbert_device,
            batch_size=sbert_batch_size,
            max_length=sbert_max_length,
        )

    rows: list[dict] = []
    for language in languages:
        candidates: list[str] = []
        backtranslation_paths: list[str] = []
        for split in splits:
            reference_items = split_references[split]
            reference_source_ids = source_ids(reference_items)
            backtranslation_path = (
                output_root
                / "backtranslations"
                / f"{subset}_{split}_{language}_to_{source_language}.json"
            )
            backtranslated_items = load_instruction_items(backtranslation_path)
            if len(backtranslated_items) != len(reference_items):
                raise ValueError(
                    f"{language}: {subset}/{split} has "
                    f"{len(backtranslated_items)} back-translations for "
                    f"{len(reference_items)} references"
                )
            validate_source_ids(
                language=language,
                expected=reference_source_ids,
                observed_items=backtranslated_items,
            )
            candidates.extend(str(item["instruction"]) for item in backtranslated_items)
            backtranslation_paths.append(str(backtranslation_path))

        record = {
            "language": language,
            "subset": subset,
            "split": "all",
            "splits": " ".join(splits),
            "n": len(candidates),
            "bleu": round(corpus_bleu(candidates, references), 4),
            "sbert": None,
            "reference_path": " ".join(
                str(root / f"{subset}_{split}_translated_{source_language}.json")
                for split in splits
            ),
            "translated_path": " ".join(
                str(root / f"{subset}_{split}_translated_{language}.json")
                for split in splits
            ),
            "backtranslation_path": " ".join(backtranslation_paths),
            "sbert_model": None if skip_sbert else sbert_model_name,
        }
        if not skip_sbert:
            candidate_embeddings = compute_sbert_candidate_embeddings(
                candidates,
                tokenizer=sbert_tokenizer,
                model=sbert_model,
                device=sbert_device,
                batch_size=sbert_batch_size,
                max_length=sbert_max_length,
            )
            record["sbert"] = round(
                sbert_similarity(candidate_embeddings, reference_embeddings),
                4,
            )
        rows.append(record)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate translated prompt quality with Wang-style BLEU/SBERT back-translation."
    )
    parser.add_argument("--root", type=Path, default=Path("data/polyrefuse"))
    parser.add_argument("--subset", default="harmful")
    parser.add_argument(
        "--subsets",
        nargs="+",
        default=None,
        help="Optional plural override. Defaults to --subset.",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help="Optional plural override. Defaults to --split.",
    )
    parser.add_argument("--source-language", default="en")
    parser.add_argument("--target-languages", nargs="+", default=None)
    parser.add_argument(
        "--backtranslation-source-language",
        default="auto",
        help="googletrans source language for back-translation; use auto unless forced.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/translation_quality/polyrefuse"),
    )
    parser.add_argument(
        "--sbert-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument("--sbert-batch-size", type=int, default=64)
    parser.add_argument(
        "--sbert-max-length",
        type=int,
        default=256,
        help="Maximum wordpiece length before truncation; 256 matches all-MiniLM-L6-v2 SentenceTransformers config.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--overwrite-backtranslations", action="store_true")
    parser.add_argument("--skip-sbert", action="store_true")
    parser.add_argument(
        "--write-aggregate",
        action="store_true",
        help="Also write train/val/test-concatenated metrics with split='all'.",
    )
    args = parser.parse_args()

    sbert_tokenizer = None
    sbert_model = None
    sbert_device = None
    if not args.skip_sbert:
        sbert_tokenizer, sbert_model, sbert_device = load_sbert(args.sbert_model, args.device)

    all_rows: list[dict] = []
    for subset in args.subsets or [args.subset]:
        for split in args.splits or [args.split]:
            all_rows.extend(
                evaluate_subset_split(
                    root=args.root,
                    subset=subset,
                    split=split,
                    source_language=args.source_language,
                    target_languages=args.target_languages,
                    backtranslation_source_language=args.backtranslation_source_language,
                    output_root=args.output_root,
                    concurrency=args.concurrency,
                    overwrite_backtranslations=args.overwrite_backtranslations,
                    skip_sbert=args.skip_sbert,
                    sbert_model_name=args.sbert_model,
                    sbert_tokenizer=sbert_tokenizer,
                    sbert_model=sbert_model,
                    sbert_device=sbert_device,
                    sbert_batch_size=args.sbert_batch_size,
                    sbert_max_length=args.sbert_max_length,
                )
            )

    if len({(row["subset"], row["split"]) for row in all_rows}) > 1:
        combined_json_path = args.output_root / "all_wang_metrics.json"
        combined_csv_path = args.output_root / "all_wang_metrics.csv"
        combined_json_path.write_text(
            json.dumps(all_rows, ensure_ascii=False, indent=2) + "\n"
        )
        write_csv(combined_csv_path, all_rows)
        print(f"[write] {combined_json_path}")
        print(f"[write] {combined_csv_path}")

    aggregate_splits = args.splits or [args.split]
    if args.write_aggregate and len(aggregate_splits) > 1:
        aggregate_rows: list[dict] = []
        for subset in args.subsets or [args.subset]:
            aggregate_rows.extend(
                evaluate_aggregate_splits(
                    root=args.root,
                    subset=subset,
                    splits=aggregate_splits,
                    source_language=args.source_language,
                    target_languages=args.target_languages,
                    output_root=args.output_root,
                    skip_sbert=args.skip_sbert,
                    sbert_model_name=args.sbert_model,
                    sbert_tokenizer=sbert_tokenizer,
                    sbert_model=sbert_model,
                    sbert_device=sbert_device,
                    sbert_batch_size=args.sbert_batch_size,
                    sbert_max_length=args.sbert_max_length,
                )
            )
        aggregate_json_path = args.output_root / "aggregate_wang_metrics.json"
        aggregate_csv_path = args.output_root / "aggregate_wang_metrics.csv"
        aggregate_json_path.write_text(
            json.dumps(aggregate_rows, ensure_ascii=False, indent=2) + "\n"
        )
        write_csv(aggregate_csv_path, aggregate_rows)
        print(f"[write] {aggregate_json_path}")
        print(f"[write] {aggregate_csv_path}")


if __name__ == "__main__":
    main()
