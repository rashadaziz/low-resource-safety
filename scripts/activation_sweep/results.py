"""Activation-sweep constants and result summaries for local scripts."""

from collections import defaultdict
from pathlib import Path
from typing import Iterable

from multilingual_latent_safety.analysis import load_resource_tiers
from multilingual_latent_safety.paths import completions_file
from multilingual_latent_safety.refusal_scoring import refusal_rate

DEFAULT_LANGUAGES: list[str] = [
    "en",
    "de",
    "fr",
    "es",
    "it",
    "nl",
    "pl",
    "ru",
    "zh",
    "ja",
    "ar",
    "ko",
    "th",
    "el",
    "he",
    "hi",
    "fa",
    "sw",
    "am",
    "my",
    "km",
    "si",
    "yo",
]

DEFAULT_HRL_LANGUAGES: list[str] = ["en", "de", "fr", "es", "it", "nl", "pl", "ru", "zh", "ja"]

DEFAULT_LAMBDAS: list[float] = [i / 4 for i in range(13)]

TIER_ORDER: list[str] = ["high", "mid", "low"]

TIER_LABELS: dict[str, str] = {
    "high": "HRL",
    "mid": "MRL",
    "low": "LRL",
}

DEFAULT_MODEL_SPECS: list[dict[str, str | int]] = [
    {
        "model_key": "qwen2.5-7b-instruct",
        "model_name": "qwen2.5-7b-instruct",
        "hf_model": "Qwen/Qwen2.5-7B-Instruct",
        "label": "Qwen2.5-7B-Instruct",
        "figure": "refusal_activation_sweep_qwen25_7b",
        "layer": 15,
    },
    {
        "model_key": "gemma-2-9b-it",
        "model_name": "gemma-2-9b-it",
        "hf_model": "google/gemma-2-9b-it",
        "label": "Gemma-2-9B-it",
        "figure": "refusal_activation_sweep_gemma2_9b",
        "layer": 20,
    },
    {
        "model_key": "llama-3.1-8b-instruct",
        "model_name": "llama-3.1-8b-instruct",
        "hf_model": "meta-llama/Llama-3.1-8B-Instruct",
        "label": "Llama-3.1-8B-Instruct",
        "figure": "refusal_activation_sweep_llama31_8b",
        "layer": 10,
    },
]

GPT4O_MINI_INPUT_PRICE_PER_MILLION = 0.15
GPT4O_MINI_OUTPUT_PRICE_PER_MILLION = 0.60


def method_slug(layer: int) -> str:
    return f"layer={int(layer)}"


def default_lambdas() -> list[float]:
    return list(DEFAULT_LAMBDAS)


def model_spec_by_hf_model(model: str) -> dict[str, str | int]:
    for spec in DEFAULT_MODEL_SPECS:
        if spec["hf_model"] == model:
            return dict(spec)
    raise KeyError(f"unknown model: {model}")


def model_specs_for(models: Iterable[str] | None = None) -> list[dict[str, str | int]]:
    if models is None:
        return [dict(spec) for spec in DEFAULT_MODEL_SPECS]
    requested = {model.strip() for model in models if model.strip()}
    specs = [
        dict(spec)
        for spec in DEFAULT_MODEL_SPECS
        if str(spec["model_key"]) in requested or str(spec["hf_model"]) in requested
    ]
    missing = sorted(
        requested
        - {
            value
            for spec in specs
            for value in (str(spec["model_key"]), str(spec["hf_model"]))
        }
    )
    if missing:
        raise KeyError(f"unknown model(s): {missing}")
    return specs


def selected_model_specs(models: str) -> list[dict[str, str | int]]:
    """Select configured sweep models by comma-separated model key or HF model ID."""
    if not models:
        return model_specs_for()
    try:
        return model_specs_for(models.split(","))
    except KeyError as exc:
        raise SystemExit(str(exc)) from exc


def language_rate_rows(
    score_root: str | Path,
    model: str,
    generation_name: str,
    layer: int,
    lambdas: Iterable[float],
    languages: Iterable[str],
    split: str = "val",
    subset: str = "harmful",
    dataset_config: str | Path = "configs/dataset/polyrefuse.yaml",
) -> list[dict[str, str | int | float]]:
    resource_tiers = load_resource_tiers(dataset_config)
    rows: list[dict[str, str | int | float]] = []
    for lambda_value in lambdas:
        lambda_root = (
            Path(score_root)
            / model
            / f"gen={generation_name}"
            / method_slug(layer)
            / f"lambda={float(lambda_value):g}"
        )
        for language in languages:
            path = completions_file(lambda_root, language, split, subset)
            if not path.exists():
                continue
            rate, n = refusal_rate(path)
            rows.append(
                {
                    "model": model,
                    "layer": int(layer),
                    "lambda": float(lambda_value),
                    "language": language,
                    "tier": resource_tiers[language],
                    "tier_label": TIER_LABELS[resource_tiers[language]],
                    "split": split,
                    "subset": subset,
                    "n": int(n),
                    "refusal_rate": float(rate),
                }
            )
    return rows


def add_language_peaks(
    rows: list[dict[str, str | int | float]]
) -> list[dict[str, str | int | float]]:
    grouped: dict[tuple[str, str], list[dict[str, str | int | float]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["model"]), str(row["language"]))].append(row)

    peak_by_key: dict[tuple[str, str], tuple[float, float]] = {}
    for key, key_rows in grouped.items():
        best = sorted(
            key_rows,
            key=lambda row: (-float(row["refusal_rate"]), float(row["lambda"])),
        )[0]
        peak_by_key[key] = (float(best["lambda"]), float(best["refusal_rate"]))

    enriched: list[dict[str, str | int | float]] = []
    for row in rows:
        peak_lambda, peak_rate = peak_by_key[(str(row["model"]), str(row["language"]))]
        new_row = dict(row)
        new_row["language_peak_lambda"] = peak_lambda
        new_row["language_peak_refusal_rate"] = peak_rate
        enriched.append(new_row)
    return enriched


def tier_summary_rows(
    rows: list[dict[str, str | int | float]]
) -> list[dict[str, str | int | float]]:
    grouped: dict[tuple[str, int, float, str, str], list[dict[str, str | int | float]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row["model"]),
            int(row["layer"]),
            float(row["lambda"]),
            str(row["tier"]),
            str(row["tier_label"]),
        )
        grouped[key].append(row)

    summary: list[dict[str, str | int | float]] = []
    for (model, layer, lambda_value, tier, tier_label), key_rows in sorted(
        grouped.items(), key=lambda item: (item[0][0], item[0][2], TIER_ORDER.index(item[0][3]))
    ):
        rates = [float(row["refusal_rate"]) for row in key_rows]
        n_prompts = sum(int(row["n"]) for row in key_rows)
        summary.append(
            {
                "model": model,
                "layer": layer,
                "lambda": lambda_value,
                "tier": tier,
                "tier_label": tier_label,
                "n_languages": len(key_rows),
                "n_prompts": n_prompts,
                "mean_refusal_rate": sum(rates) / len(rates),
            }
        )

    grouped_summary: dict[tuple[str, str], list[dict[str, str | int | float]]] = defaultdict(list)
    for row in summary:
        grouped_summary[(str(row["model"]), str(row["tier"]))].append(row)
    peak_by_key: dict[tuple[str, str], tuple[float, float]] = {}
    for key, key_rows in grouped_summary.items():
        best = sorted(
            key_rows,
            key=lambda row: (-float(row["mean_refusal_rate"]), float(row["lambda"])),
        )[0]
        peak_by_key[key] = (float(best["lambda"]), float(best["mean_refusal_rate"]))

    enriched: list[dict[str, str | int | float]] = []
    for row in summary:
        peak_lambda, peak_rate = peak_by_key[(str(row["model"]), str(row["tier"]))]
        new_row = dict(row)
        new_row["tier_peak_lambda"] = peak_lambda
        new_row["tier_peak_refusal_rate"] = peak_rate
        enriched.append(new_row)
    return enriched


def estimate_refusal_judge_cost(
    num_lambdas: int,
    num_prompts: int,
    num_languages: int,
    num_models: int,
    input_tokens_per_call: int = 518,
    output_tokens_per_call: int = 32,
) -> dict[str, float | int]:
    calls = int(num_lambdas) * int(num_prompts) * int(num_languages) * int(num_models)
    input_tokens = calls * int(input_tokens_per_call)
    output_tokens = calls * int(output_tokens_per_call)
    cost = (
        input_tokens / 1_000_000 * GPT4O_MINI_INPUT_PRICE_PER_MILLION
        + output_tokens / 1_000_000 * GPT4O_MINI_OUTPUT_PRICE_PER_MILLION
    )
    return {
        "calls": calls,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "estimated_cost_usd": cost,
    }
