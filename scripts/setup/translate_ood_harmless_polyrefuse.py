import argparse
from pathlib import Path

from translation_common import translate_file


DEFAULT_LANGUAGE_MAP = {
    "id": ["id"],
    "id_colloquial": ["id"],
    "jv": ["jw", "jv"],
    "su": ["su"],
    "min": ["min"],
    "bn": ["bn"],
    "vi": ["vi"],
}


def parse_language_map(values: list[str] | None) -> dict[str, list[str]]:
    mapping = {key: list(value) for key, value in DEFAULT_LANGUAGE_MAP.items()}
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"language map entries must be target=dest[,dest2]: {value}")
        target, dests = value.split("=", 1)
        mapping[target] = [dest.strip() for dest in dests.split(",") if dest.strip()]
    return mapping


def ood_row_builder(
    *,
    src_lang: str,
    target_language: str,
    split: str,
):
    def build_row(idx: int, item: dict, translated_instruction: str, dest_lang: str) -> dict:
        return {
            "instruction": translated_instruction,
            "category": item.get("category"),
            "source_instruction": item["instruction"],
            "source_language": src_lang,
            "target_language": target_language,
            "translation_language": dest_lang,
            "source_split": split,
            "source_id": f"{target_language}:{split}:{idx}",
        }

    return build_row


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Translate PolyRefuse harmless splits into an OOD-only root."
    )
    parser.add_argument("--source-root", type=Path, default=Path("data/polyrefuse"))
    parser.add_argument("--output-root", type=Path, default=Path("data/ood_harmless_translations"))
    parser.add_argument("--source-language", default="en")
    parser.add_argument(
        "--target-languages",
        nargs="+",
        default=list(DEFAULT_LANGUAGE_MAP),
    )
    parser.add_argument("--language-map", nargs="*", default=None)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    language_map = parse_language_map(args.language_map)
    for target_language in args.target_languages:
        if target_language not in language_map:
            raise ValueError(f"missing googletrans destination mapping for {target_language}")
        for split in args.splits:
            source_path = (
                args.source_root
                / f"harmless_{split}_translated_{args.source_language}.json"
            )
            dest_path = args.output_root / f"harmless_{split}_translated_{target_language}.json"
            if not source_path.exists():
                print(f"[miss] {source_path}")
                continue
            if dest_path.exists() and not args.overwrite:
                print(f"[skip] {dest_path}")
                continue
            print(f"[run ] {source_path} -> {dest_path}")
            translate_file(
                source_path=source_path,
                dest_path=dest_path,
                src_lang=args.source_language,
                target_language=target_language,
                dest_candidates=language_map[target_language],
                concurrency=args.concurrency,
                build_row=ood_row_builder(
                    src_lang=args.source_language,
                    target_language=target_language,
                    split=split,
                ),
                desc=f"{args.source_language}->{target_language}",
            )


if __name__ == "__main__":
    main()
