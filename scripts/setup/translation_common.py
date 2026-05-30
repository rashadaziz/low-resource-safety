"""Shared googletrans helpers for setup-time data translation scripts."""

import asyncio
from collections.abc import Callable, Sequence
import json
from pathlib import Path

from googletrans import Translator
from tqdm.asyncio import tqdm_asyncio

TranslatedRowBuilder = Callable[[int, dict, str, str], dict]


def load_instruction_items(path: Path) -> list[dict]:
    with path.open() as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"expected list in {path}")
    return [item for item in data if isinstance(item, dict) and item.get("instruction")]


def write_translated_items(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(list(rows), ensure_ascii=False, indent=2) + "\n")


async def translate_instruction_items(
    items: Sequence[dict],
    *,
    src_lang: str,
    target_language: str,
    dest_candidates: Sequence[str],
    concurrency: int,
    desc: str,
    build_row: TranslatedRowBuilder,
    show_progress: bool = True,
) -> list[dict]:
    translator = Translator()
    sem = asyncio.Semaphore(concurrency)
    candidates = list(dest_candidates)

    async def one(idx: int, item: dict) -> dict:
        last_exc: Exception | None = None
        async with sem:
            for dest_lang in candidates:
                try:
                    result = await translator.translate(
                        str(item["instruction"]),
                        src=src_lang,
                        dest=dest_lang,
                    )
                    return build_row(idx, item, result.text, dest_lang)
                except Exception as exc:
                    last_exc = exc
            raise RuntimeError(
                f"failed to translate {target_language} item {idx} with candidates {candidates}"
            ) from last_exc

    return await tqdm_asyncio.gather(
        *[one(idx, item) for idx, item in enumerate(items)],
        desc=desc,
        disable=not show_progress,
    )


def translate_file(
    *,
    source_path: Path,
    dest_path: Path,
    src_lang: str,
    target_language: str,
    dest_candidates: Sequence[str],
    concurrency: int,
    build_row: TranslatedRowBuilder,
    desc: str,
    show_progress: bool = True,
) -> None:
    items = load_instruction_items(source_path)
    translated = asyncio.run(
        translate_instruction_items(
            items,
            src_lang=src_lang,
            target_language=target_language,
            dest_candidates=dest_candidates,
            concurrency=concurrency,
            desc=desc,
            build_row=build_row,
            show_progress=show_progress,
        )
    )
    write_translated_items(dest_path, translated)
