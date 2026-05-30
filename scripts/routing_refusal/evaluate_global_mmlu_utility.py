"""Logit-scored Global-MMLU utility evaluation before and after steering."""

import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import hydra
import torch
from datasets import Dataset, get_dataset_config_names, load_dataset
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import load_file
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multilingual_latent_safety.conditional_vhrl import (
    annotate_gate_outputs,
    capture_final_prompt_activations,
    conditional_alpha_calibration,
    train_gates,
)
from multilingual_latent_safety.generation import batched, load_generation_model
from multilingual_latent_safety.interventions import (
    get_decoder_blocks,
    install_hooks,
    make_conditional_refusal_hook,
)
from multilingual_latent_safety.csv_io import write_rows as write_csv_rows
from multilingual_latent_safety.json_io import read_jsonl, write_jsonl
from multilingual_latent_safety.model import format_prompt, last_nonpad_positions
from multilingual_latent_safety.paths import pooled_direction_file
from multilingual_latent_safety.runtime import set_seed

CHOICE_LABELS = ("A", "B", "C", "D")
GLOBAL_MMLU_LANGUAGES = (
    "am",
    "ar",
    "bn",
    "cs",
    "de",
    "el",
    "en",
    "es",
    "fa",
    "fil",
    "fr",
    "ha",
    "he",
    "hi",
    "id",
    "ig",
    "it",
    "ja",
    "ko",
    "ky",
    "lt",
    "mg",
    "ms",
    "ne",
    "nl",
    "ny",
    "pl",
    "pt",
    "ro",
    "ru",
    "si",
    "sn",
    "so",
    "sr",
    "sv",
    "sw",
    "te",
    "tr",
    "uk",
    "vi",
    "yo",
    "zh",
)


@dataclass
class GlobalMMLUItem:
    language: str
    split: str
    sample_id: str
    subject: str
    subject_category: str
    question: str
    option_a: str
    option_b: str
    option_c: str
    option_d: str
    answer: str
    prompt: str = ""
    gate_logit: float = math.nan
    gate_pred_harmful: bool = False


def resolve_global_mmlu_languages(
    requested_languages: Sequence[str],
    available_languages: Iterable[str] = GLOBAL_MMLU_LANGUAGES,
) -> tuple[list[str], list[str]]:
    available = set(available_languages)
    present = [str(language) for language in requested_languages if str(language) in available]
    missing = [str(language) for language in requested_languages if str(language) not in available]
    return present, missing


def global_mmlu_instruction(item: GlobalMMLUItem) -> str:
    return (
        f"{item.question}\n\n"
        f"A. {item.option_a}\n"
        f"B. {item.option_b}\n"
        f"C. {item.option_c}\n"
        f"D. {item.option_d}\n\n"
        "Answer:"
    )


def normalize_answer(answer: Any) -> str:
    label = str(answer).strip().upper()
    if label not in CHOICE_LABELS:
        raise ValueError(f"Global-MMLU answer must be one of {CHOICE_LABELS}, got {answer!r}")
    return label


def item_from_row(language: str, split: str, row: Mapping[str, Any], fallback_id: int) -> GlobalMMLUItem:
    return GlobalMMLUItem(
        language=str(language),
        split=str(split),
        sample_id=str(row.get("sample_id") or f"{language}/{split}/{fallback_id}"),
        subject=str(row.get("subject") or ""),
        subject_category=str(row.get("subject_category") or ""),
        question=str(row.get("question") or ""),
        option_a=str(row.get("option_a") or ""),
        option_b=str(row.get("option_b") or ""),
        option_c=str(row.get("option_c") or ""),
        option_d=str(row.get("option_d") or ""),
        answer=normalize_answer(row.get("answer")),
    )


def select_language_dataset(ds: Dataset, max_samples: int | None, seed: int) -> Dataset:
    if max_samples is None:
        return ds
    if max_samples < 0:
        raise ValueError(f"max_samples_per_language must be non-negative or null, got {max_samples}")
    if max_samples >= len(ds):
        return ds
    return ds.shuffle(seed=int(seed)).select(range(int(max_samples)))


def load_language_items(
    dataset_name: str,
    language: str,
    split: str,
    tokenizer,
    chat_cfg: DictConfig,
    max_samples: int | None,
    seed: int,
) -> list[GlobalMMLUItem]:
    ds = load_dataset(dataset_name, language, split=split)
    ds = select_language_dataset(ds, max_samples=max_samples, seed=seed)
    items = [item_from_row(language, split, row, idx) for idx, row in enumerate(ds)]
    for item in items:
        item.prompt = format_prompt(tokenizer, global_mmlu_instruction(item), chat_cfg)
    return items


def answer_token_ids(tokenizer, variants: Sequence[str]) -> dict[str, tuple[int, ...]]:
    token_ids: dict[str, tuple[int, ...]] = {}
    for label in CHOICE_LABELS:
        ids: list[int] = []
        for variant in variants:
            rendered = str(variant).format(label=label)
            encoded = tokenizer.encode(rendered, add_special_tokens=False)
            if len(encoded) == 1 and int(encoded[0]) not in ids:
                ids.append(int(encoded[0]))
        if not ids:
            fallback = tokenizer.encode(label, add_special_tokens=False)
            if not fallback:
                raise ValueError(f"Could not tokenize answer label {label!r}")
            ids.append(int(fallback[-1]))
        token_ids[label] = tuple(ids)
    return token_ids


def choice_scores_from_logits(
    logits: torch.Tensor,
    label_token_ids: Mapping[str, Sequence[int]],
    variant_reduce: str,
) -> torch.Tensor:
    scores = []
    for label in CHOICE_LABELS:
        ids = torch.tensor(label_token_ids[label], device=logits.device, dtype=torch.long)
        label_logits = logits.index_select(dim=-1, index=ids)
        if variant_reduce == "max":
            scores.append(label_logits.max(dim=-1).values)
        elif variant_reduce == "logsumexp":
            scores.append(torch.logsumexp(label_logits, dim=-1))
        else:
            raise ValueError(f"Unknown variant_reduce={variant_reduce!r}")
    return torch.stack(scores, dim=-1)


def predict_labels(choice_scores: torch.Tensor) -> list[str]:
    indices = torch.argmax(choice_scores, dim=-1).tolist()
    return [CHOICE_LABELS[int(index)] for index in indices]


def score_margin(scores: Mapping[str, float], answer: str) -> float:
    correct = float(scores[answer])
    wrong = max(float(value) for label, value in scores.items() if label != answer)
    return correct - wrong


def score_entropy(scores: Mapping[str, float]) -> float:
    values = torch.tensor([float(scores[label]) for label in CHOICE_LABELS])
    probs = torch.softmax(values, dim=0)
    return float(-(probs * torch.log(probs.clamp_min(1e-12))).sum().item())


def row_from_scores(
    item: GlobalMMLUItem,
    model_name: str,
    mode: str,
    lambda_value: float | None,
    alpha: float | None,
    prediction: str,
    scores: Mapping[str, float],
) -> dict[str, Any]:
    return {
        "model": model_name,
        "language": item.language,
        "split": item.split,
        "mode": mode,
        "lambda": lambda_value,
        "alpha": alpha,
        "sample_id": item.sample_id,
        "subject": item.subject,
        "subject_category": item.subject_category,
        "answer": item.answer,
        "prediction": prediction,
        "correct": prediction == item.answer,
        "margin": score_margin(scores, item.answer),
        "entropy": score_entropy(scores),
        "gate_logit": item.gate_logit,
        "gate_pred_harmful": item.gate_pred_harmful,
        "score_A": scores["A"],
        "score_B": scores["B"],
        "score_C": scores["C"],
        "score_D": scores["D"],
    }


@torch.inference_mode()
def score_items_with_logits(
    model: torch.nn.Module,
    tokenizer,
    items: Sequence[GlobalMMLUItem],
    model_name: str,
    mode: str,
    batch_size: int,
    label_token_ids: Mapping[str, Sequence[int]],
    variant_reduce: str,
    lambda_value: float | None = None,
    alpha: float | None = None,
    direction: torch.Tensor | None = None,
    layer: int | None = None,
    gate_result: Any | None = None,
    gate_capture_layer: int | None = None,
) -> list[dict[str, Any]]:
    device = getattr(model, "device", next(model.parameters()).device)
    rows: list[dict[str, Any]] = []
    blocks = get_decoder_blocks(model) if gate_result is not None else None
    for batch_items in tqdm(list(batched(list(items), int(batch_size))), desc=f"score:{mode}"):
        prompts = [item.prompt for item in batch_items]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        positions = last_nonpad_positions(inputs["attention_mask"]).to(device)
        captured_activation: list[torch.Tensor] = []

        def capture_hook(module, hook_inputs, output):
            del module, hook_inputs
            residual = output[0] if isinstance(output, tuple) else output
            batch_index = torch.arange(residual.shape[0], device=residual.device)
            captured_activation.append(
                residual[batch_index, positions.to(residual.device), :].detach().float().cpu()
            )
            return output

        capture_handle = None
        if gate_result is not None:
            if blocks is None or gate_capture_layer is None:
                raise ValueError("Gate capture requires transformer blocks and gate_capture_layer")
            capture_handle = blocks[int(gate_capture_layer)].register_forward_hook(capture_hook)

        try:
            if direction is None:
                outputs = model(**inputs, use_cache=False)
            else:
                if layer is None or alpha is None:
                    raise ValueError("Conditional scoring requires layer and alpha")
                harmful_mask = torch.tensor(
                    [item.gate_pred_harmful for item in batch_items],
                    dtype=torch.bool,
                )
                hook_factory = lambda hook_device, hook_dtype: make_conditional_refusal_hook(
                    direction,
                    alpha,
                    harmful_mask,
                    hook_device,
                    hook_dtype,
                )
                with install_hooks(model, [int(layer)], hook_factory):
                    outputs = model(**inputs, use_cache=False)
        finally:
            if capture_handle is not None:
                capture_handle.remove()

        if gate_result is not None:
            if len(captured_activation) != 1:
                raise RuntimeError(
                    f"Expected one captured activation batch, got {len(captured_activation)}"
                )
            annotate_gate_outputs(
                batch_items,
                captured_activation[0],
                gate_result.gates,
                gate_result.thresholds,
            )

        positions = positions.to(logits_device(outputs))
        batch_index = torch.arange(len(batch_items), device=logits_device(outputs))
        logits = outputs.logits[batch_index, positions]
        choice_scores = choice_scores_from_logits(logits, label_token_ids, variant_reduce)
        predictions = predict_labels(choice_scores)
        scores_cpu = choice_scores.detach().float().cpu()
        for item, prediction, score_tensor in zip(batch_items, predictions, scores_cpu, strict=True):
            scores = {label: float(score_tensor[idx].item()) for idx, label in enumerate(CHOICE_LABELS)}
            rows.append(
                row_from_scores(
                    item=item,
                    model_name=model_name,
                    mode=mode,
                    lambda_value=lambda_value,
                    alpha=alpha,
                    prediction=prediction,
                    scores=scores,
                )
            )
        del outputs, inputs, logits, choice_scores
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return rows


def logits_device(outputs: Any) -> torch.device:
    return outputs.logits.device


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0, "accuracy": math.nan, "mean_margin": math.nan, "mean_entropy": math.nan}
    correct = sum(1 for row in rows if bool(row["correct"]))
    return {
        "n": len(rows),
        "accuracy": correct / len(rows),
        "mean_margin": sum(float(row["margin"]) for row in rows) / len(rows),
        "mean_entropy": sum(float(row["entropy"]) for row in rows) / len(rows),
        "gate_fire_rate": sum(1 for row in rows if bool(row["gate_pred_harmful"])) / len(rows),
    }


def subject_category_summaries(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("subject_category") or "")].append(row)
    summaries = []
    for category, category_rows in sorted(grouped.items()):
        summary = summarize_rows(category_rows)
        summary["subject_category"] = category
        summaries.append(summary)
    return summaries


def write_summary_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "model",
        "language",
        "split",
        "mode",
        "lambda",
        "alpha",
        "subject_category",
        "n",
        "accuracy",
        "mean_margin",
        "mean_entropy",
        "gate_fire_rate",
        "status",
    ]
    write_csv_rows(path, [{field: row.get(field, "") for field in fields} for row in rows], fields)


def output_file(output_root: Path, mode: str, language: str, split: str, lambda_value: float | None) -> Path:
    mode_dir = f"mode={mode}"
    if lambda_value is not None:
        mode_dir = f"{mode_dir}/lambda={float(lambda_value):g}"
    return output_root / mode_dir / language / f"{split}.jsonl"


def summarize_output_rows(
    rows: Sequence[Mapping[str, Any]],
    model_name: str,
    language: str,
    split: str,
    mode: str,
    lambda_value: float | None,
    alpha: float | None,
) -> list[dict[str, Any]]:
    summaries = []
    all_summary = summarize_rows(rows)
    all_summary.update(
        {
            "model": model_name,
            "language": language,
            "split": split,
            "mode": mode,
            "lambda": lambda_value,
            "alpha": alpha,
            "subject_category": "ALL",
            "status": "ok",
        }
    )
    summaries.append(all_summary)
    for category_summary in subject_category_summaries(rows):
        category_summary.update(
            {
                "model": model_name,
                "language": language,
                "split": split,
                "mode": mode,
                "lambda": lambda_value,
                "alpha": alpha,
                "status": "ok",
            }
        )
        summaries.append(category_summary)
    return summaries


def make_gate_config(cfg: DictConfig, eval_languages: Sequence[str]) -> DictConfig:
    gate_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    gate_cfg.target_languages = list(eval_languages)
    return gate_cfg


def maybe_hf_dataset_configs(dataset_name: str, use_remote_configs: bool) -> tuple[str, ...]:
    if not use_remote_configs:
        return GLOBAL_MMLU_LANGUAGES
    return tuple(str(name) for name in get_dataset_config_names(dataset_name))


def missing_language_summary_rows(
    model_name: str,
    split: str,
    missing_languages: Sequence[str],
) -> list[dict[str, Any]]:
    rows = []
    for language in missing_languages:
        rows.append(
            {
                "model": model_name,
                "language": language,
                "split": split,
                "mode": "",
                "lambda": "",
                "alpha": "",
                "subject_category": "ALL",
                "n": 0,
                "accuracy": "",
                "mean_margin": "",
                "mean_entropy": "",
                "gate_fire_rate": "",
                "status": "missing_from_global_mmlu",
            }
        )
    return rows


def run_global_mmlu_utility(cfg: DictConfig) -> None:
    set_seed(int(cfg.seed))
    dataset_name = str(cfg.dataset_name)
    split = str(cfg.split)
    output_root = Path(cfg.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    available_languages = maybe_hf_dataset_configs(dataset_name, bool(cfg.get("refresh_dataset_configs", False)))
    eval_languages, missing_languages = resolve_global_mmlu_languages(cfg.languages, available_languages)
    if not eval_languages:
        raise ValueError(f"No requested languages are available in {dataset_name}: {list(cfg.languages)}")

    model, tokenizer = load_generation_model(cfg.model)
    label_token_ids = answer_token_ids(tokenizer, list(cfg.answer_label_variants))
    model_name = str(cfg.model.name)
    modes = {str(mode) for mode in cfg.modes}
    summary_rows = missing_language_summary_rows(model_name, split, missing_languages)

    gate_result = None
    direction = None
    alpha_calibration = None
    if "conditional_vhrl" in modes:
        gate_cfg = make_gate_config(cfg, eval_languages)
        gate_result = train_gates(gate_cfg)
        direction_path = pooled_direction_file(
            Path(gate_cfg.pooled_root),
            "hrl",
            gate_cfg.token_position,
            int(gate_cfg.layer),
        )
        direction = load_file(direction_path)["direction"]
        alpha_calibration = conditional_alpha_calibration(
            gate_cfg,
            gate_result.source_languages,
            direction,
        )

    metadata = {
        "model": model_name,
        "dataset": dataset_name,
        "split": split,
        "requested_languages": list(cfg.languages),
        "evaluated_languages": eval_languages,
        "missing_languages": missing_languages,
        "answer_label_token_ids": {key: list(value) for key, value in label_token_ids.items()},
        "scoring": "final_prompt_position_choice_label_logits",
        "modes": sorted(modes),
    }
    if alpha_calibration is not None:
        metadata.update(alpha_calibration.metadata)
    (output_root / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))

    for language in eval_languages:
        print(f"[load] {dataset_name}/{language}/{split}")
        items = load_language_items(
            dataset_name=dataset_name,
            language=language,
            split=split,
            tokenizer=tokenizer,
            chat_cfg=cfg.model.chat,
            max_samples=cfg.max_samples_per_language,
            seed=int(cfg.seed),
        )
        gate_outputs_available = False
        if "base" in modes:
            out_path = output_file(output_root, "base", language, split, None)
            if out_path.exists() and not bool(cfg.overwrite):
                print(f"[skip] base {language} -> {out_path}")
                rows = read_jsonl(out_path)
            else:
                print(f"[run ] base {language}: {len(items)} items")
                rows = score_items_with_logits(
                    model=model,
                    tokenizer=tokenizer,
                    items=items,
                    model_name=model_name,
                    mode="base",
                    batch_size=int(cfg.evaluation.batch_size),
                    label_token_ids=label_token_ids,
                    variant_reduce=str(cfg.evaluation.variant_reduce),
                    gate_result=gate_result if "conditional_vhrl" in modes else None,
                    gate_capture_layer=int(cfg.layer) if "conditional_vhrl" in modes else None,
                )
                write_jsonl(out_path, rows)
                gate_outputs_available = "conditional_vhrl" in modes
            summary_rows.extend(
                summarize_output_rows(rows, model_name, language, split, "base", None, None)
            )

        if "conditional_vhrl" in modes:
            assert gate_result is not None
            assert direction is not None
            assert alpha_calibration is not None
            lambda_values = [float(value) for value in cfg.lambdas]
            pending_paths = [
                output_file(output_root, "conditional_vhrl", language, split, lambda_value)
                for lambda_value in lambda_values
            ]
            needs_gate_outputs = (
                not gate_outputs_available
                and (bool(cfg.overwrite) or any(not path.exists() for path in pending_paths))
            )
            if needs_gate_outputs:
                prompts = [item.prompt for item in items]
                activations = capture_final_prompt_activations(
                    model=model,
                    tokenizer=tokenizer,
                    prompts=prompts,
                    layer=int(cfg.layer),
                    batch_size=int(cfg.capture_batch_size),
                )
                annotate_gate_outputs(items, activations, gate_result.gates, gate_result.thresholds)
                del activations
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            for lambda_value in lambda_values:
                alpha = float(lambda_value) * float(alpha_calibration.base)
                out_path = output_file(output_root, "conditional_vhrl", language, split, lambda_value)
                if out_path.exists() and not bool(cfg.overwrite):
                    print(f"[skip] conditional_vhrl {language} lambda={lambda_value:g} -> {out_path}")
                    rows = read_jsonl(out_path)
                else:
                    print(
                        f"[run ] conditional_vhrl {language}: {len(items)} items "
                        f"lambda={lambda_value:g} alpha={alpha:.4f}"
                    )
                    rows = score_items_with_logits(
                        model=model,
                        tokenizer=tokenizer,
                        items=items,
                        model_name=model_name,
                        mode="conditional_vhrl",
                        batch_size=int(cfg.evaluation.batch_size),
                        label_token_ids=label_token_ids,
                        variant_reduce=str(cfg.evaluation.variant_reduce),
                        lambda_value=lambda_value,
                        alpha=alpha,
                        direction=direction,
                        layer=int(cfg.layer),
                    )
                    write_jsonl(out_path, rows)
                summary_rows.extend(
                    summarize_output_rows(
                        rows,
                        model_name,
                        language,
                        split,
                        "conditional_vhrl",
                        lambda_value,
                        alpha,
                    )
                )

    write_summary_csv(output_root / "summary.csv", summary_rows)
    print(f"[done] wrote {output_root / 'summary.csv'}")


@hydra.main(version_base=None, config_path="../../configs", config_name="routing_refusal/evaluate_global_mmlu_utility")
def main(cfg: DictConfig) -> None:
    run_global_mmlu_utility(cfg)


if __name__ == "__main__":
    main()
