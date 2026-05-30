import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

POLYREFUSE_REPO = "https://github.com/mainlp/Multilingual-Refusal.git"
DEFAULT_DEST = Path("data/polyrefuse")


def normalize_schema(dest: Path) -> None:
    """Collapse ``instruction_translated`` (target language) into ``instruction`` where both exist.

    Upstream PolyRefuse multilingual files ship with ``instruction`` holding the English source and
    ``instruction_translated`` holding the translation; this rewrites each record to carry a single
    ``instruction`` field in the file's own language, so downstream loaders can treat every file uniformly.
    Safe to run repeatedly.
    """
    for path in sorted(dest.glob("*_translated_*.json")):
        with open(path) as f:
            items = json.load(f)
        changed = False
        for idx, item in enumerate(items):
            if "source_id" not in item:
                item["source_id"] = str(idx)
                changed = True
            if "instruction_translated" in item:
                if "source_instruction" not in item:
                    item["source_instruction"] = item.get("instruction")
                item["instruction"] = item.pop("instruction_translated")
                changed = True
        if changed:
            with open(path, "w") as f:
                json.dump(items, f, ensure_ascii=False, indent=2)
            print(f"[norm ] {path.name}")


def vendor(dest: Path, force: bool = False) -> None:
    """Clone PolyRefuse into ``dest`` with unified ``{subset}_{split}_translated_{lang}.json`` naming.

    Copies the upstream multilingual JSONs as-is and renames the English originals from
    ``dataset/splits/`` to match the ``_translated_en`` convention so loaders can treat
    every language uniformly.
    """
    if dest.exists() and any(dest.iterdir()) and not force:
        print(f"[skip] {dest} already populated — rerun with --force to rebuild.")
        return
    if force and dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        print(f"[clone] {POLYREFUSE_REPO}")
        subprocess.run(
            ["git", "clone", "--depth", "1", POLYREFUSE_REPO, tmp],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        multi = Path(tmp) / "PolyRefuse"
        en_originals = Path(tmp) / "dataset" / "splits"

        for p in multi.iterdir():
            if p.is_file() and p.suffix == ".json":
                shutil.copy2(p, dest / p.name)
            elif p.is_dir():
                shutil.copytree(p, dest / p.name, dirs_exist_ok=True)

        en_source_files = {
            ("harmful", "train"): "harmful_train.json",
            ("harmful", "val"): "harmful_val.json",
            ("harmless", "train"): "harmless_train_200_sampled.json",
            ("harmless", "val"): "harmless_val_200_sampled.json",
        }
        for (subset, split), src_name in en_source_files.items():
            shutil.copy2(en_originals / src_name, dest / f"{subset}_{split}_translated_en.json")

    normalize_schema(dest)

    files = sorted(p.name for p in dest.iterdir() if p.is_file())
    print(f"[done] {len(files)} files in {dest}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Vendor PolyRefuse dataset with unified _translated_{lang} naming.")
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--force", action="store_true", help="Wipe and re-download.")
    args = parser.parse_args()
    vendor(args.dest, force=args.force)


if __name__ == "__main__":
    main()
