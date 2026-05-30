"""Score refusal activation sweep completions with the refusal judge."""


import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from results import (
    DEFAULT_LANGUAGES,
    DEFAULT_LAMBDAS,
    method_slug,
    selected_model_specs,
)


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key, value.strip().strip('"').strip("'"))


def quoted_override(key: str, value: str | Path) -> str:
    return f"{key}='{value}'"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--judge-root",
        default="artifacts/refusal_activation_sweep/refusal_gpt4omini",
    )
    parser.add_argument(
        "--completions-root",
        default="artifacts/completions_refusal_activation_sweep",
    )
    parser.add_argument("--generation", default="greedy")
    parser.add_argument("--split", default="val")
    parser.add_argument("--subset", default="harmful")
    parser.add_argument("--languages", default=",".join(DEFAULT_LANGUAGES))
    parser.add_argument("--models", default="")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    load_env(Path(".env"))
    load_env(Path(".env"))
    languages = [language.strip() for language in args.languages.split(",") if language.strip()]
    hydra_languages = "[" + ",".join(languages) + "]"

    for spec in selected_model_specs(args.models):
        model = str(spec["hf_model"])
        layer = int(spec["layer"])
        method = method_slug(layer)
        for lambda_value in DEFAULT_LAMBDAS:
            lambda_dir = f"lambda={float(lambda_value):g}"
            completions_root = (
                Path(args.completions_root)
                / model
                / f"gen={args.generation}"
                / method
                / lambda_dir
            )
            output_root = (
                Path(args.judge_root)
                / model
                / f"gen={args.generation}"
                / method
                / lambda_dir
            )
            cmd = [
                sys.executable,
                "scripts/refusal_gap/score_refusal.py",
                quoted_override("completions_root", completions_root),
                quoted_override("output_root", output_root),
                f"model_name={spec['model_name']}",
                f"splits=[{args.split}]",
                f"subsets=[{args.subset}]",
                f"dataset.languages={hydra_languages}",
                f"overwrite={str(bool(args.overwrite)).lower()}",
            ]
            print(f"[score] {spec['model_key']} {lambda_dir}")
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
