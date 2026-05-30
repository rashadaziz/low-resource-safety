"""Download released AdaSteer vectors used by the PolyRefuse LRL baseline."""


import argparse
import shutil
import urllib.request
from pathlib import Path

from multilingual_latent_safety.adasteer import ADASTEER_SPECS


BASE_URL = "https://raw.githubusercontent.com/MuyuenLP/AdaSteer/master/vectors"
VECTOR_FILES = (
    "HD/class_a.pkl",
    "HD/class_b.pkl",
    "HD/proj.pkl",
    "RD/class_a.pkl",
    "RD/class_b.pkl",
    "RD/mean_diff.pkl",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="artifacts/external/adasteer")
    parser.add_argument(
        "--model-key",
        action="append",
        choices=sorted({spec.key for spec in ADASTEER_SPECS.values()}),
        help="Download one vector set. Repeat for multiple. Defaults to all supported sets.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def download_file(url: str, path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        print(f"[skip] {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    print(f"[get ] {url} -> {path}")
    with urllib.request.urlopen(url) as response, tmp.open("wb") as out:
        shutil.copyfileobj(response, out)
    tmp.replace(path)


def main() -> None:
    args = parse_args()
    keys = args.model_key or sorted({spec.key for spec in ADASTEER_SPECS.values()})
    root = Path(args.root)
    for model_key in keys:
        for rel in VECTOR_FILES:
            url = f"{BASE_URL}/{model_key}/{rel}"
            download_file(url, root / model_key / rel, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
