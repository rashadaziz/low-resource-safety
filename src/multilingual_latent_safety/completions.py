"""Helpers for completion JSONL artifacts."""

from collections.abc import Sequence
from typing import Any


def completion_instruction(item: dict[str, Any], fallback_instructions: Sequence[str]) -> str:
    """Resolve the instruction paired with a completion row."""
    for key in ("instruction", "user_prompt", "prompt"):
        value = item.get(key)
        if isinstance(value, str) and value:
            return value
    prompt_id = item.get("prompt_id")
    if prompt_id is not None:
        return fallback_instructions[int(prompt_id)]
    raise KeyError("completion row has no instruction/prompt field or prompt_id fallback")


def completion_rows(
    instructions: Sequence[str],
    completions: Sequence[str],
) -> list[dict[str, int | str]]:
    """Build standard ``prompt_id``/``instruction``/``completion`` rows."""
    return [
        {"prompt_id": idx, "instruction": instruction, "completion": completion}
        for idx, (instruction, completion) in enumerate(zip(instructions, completions, strict=True))
    ]
