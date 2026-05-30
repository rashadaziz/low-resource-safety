import argparse
from pathlib import Path

from translation_common import translate_file


def polyrefuse_row(idx: int, item: dict, translated_instruction: str, dest_lang: str) -> dict:
    return {
        "instruction": translated_instruction,
        "category": item.get("category"),
        "source_id": item.get("source_id", str(idx)),
        "source_instruction": item.get("source_instruction", item.get("instruction")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Translate PolyRefuse files to a new language via googletrans.")
    parser.add_argument("--root", type=Path, default=Path("data/polyrefuse"))
    parser.add_argument("--source-language", default="en")
    parser.add_argument("--target-language", required=True)
    parser.add_argument("--subsets", nargs="+", default=["harmful", "harmless"])
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    for subset in args.subsets:
        for split in args.splits:
            src = args.root / f"{subset}_{split}_translated_{args.source_language}.json"
            dst = args.root / f"{subset}_{split}_translated_{args.target_language}.json"
            if not src.exists():
                print(f"[miss] source {src.name} not found — skip")
                continue
            if dst.exists() and not args.overwrite:
                print(f"[skip] {dst.name} exists (pass --overwrite to replace)")
                continue
            print(f"[run ] {src.name} -> {dst.name}")
            translate_file(
                source_path=src,
                dest_path=dst,
                src_lang=args.source_language,
                target_language=args.target_language,
                dest_candidates=[args.target_language],
                concurrency=args.concurrency,
                build_row=polyrefuse_row,
                desc=f"{args.source_language}->{args.target_language}",
                show_progress=not args.no_progress,
            )


if __name__ == "__main__":
    main()
