#!/usr/bin/env python3
"""Convert Dolly, MBPP, and GSM8K source files into canonical JSONL traces."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path
from typing import Any, Iterable

from transformers import AutoTokenizer

from src.metrics import percentile
from src.workload import TraceRequest, write_trace


def _read_json(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, list):
            raise ValueError(f"{path}: expected a JSON array")
        return value
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _prompt(category: str, row: dict[str, Any]) -> tuple[str, str]:
    if category == "general":
        instruction = str(row["instruction"]).strip()
        context = str(row.get("context") or "").strip()
        return (instruction if not context else f"{instruction}\n\nContext:\n{context}",
                str(row.get("id") or ""))
    if category == "code":
        text = str(row["text"]).strip()
        tests = row.get("test_list") or []
        test_text = "\n".join(str(test) for test in tests)
        prompt = f"Solve this Python programming task. Return the implementation.\n\n{text}"
        if test_text:
            prompt += f"\n\nThe implementation must satisfy these examples:\n{test_text}"
        return prompt, str(row.get("task_id") or "")
    question = str(row["question"]).strip()
    return f"Solve the following math problem and show your reasoning.\n\n{question}", str(row.get("id") or "")


def _token_count(tokenizer: Any, prompt: str) -> int:
    messages = [{"role": "user", "content": prompt}]
    tokens = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
    )
    return len(tokens)


def build(
    *, category: str, source: Path, output: Path, tokenizer_name: str,
    minimum_tokens: int, maximum_tokens: int, count: int,
    request_rate: float, max_output_tokens: int, seed: int,
) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=True)
    candidates = []
    for index, row in enumerate(_read_json(source)):
        prompt, source_id = _prompt(category, row)
        tokens = _token_count(tokenizer, prompt)
        if minimum_tokens <= tokens <= maximum_tokens:
            candidates.append((index, source_id, prompt, tokens))
    if len(candidates) < count:
        raise ValueError(
            f"{category}: only {len(candidates)} samples in {minimum_tokens}-{maximum_tokens} tokens"
        )
    random.Random(seed).shuffle(candidates)
    selected = candidates[:count]
    requests = [TraceRequest(
        request_id=f"{category}-{position:04d}", category=category,
        prompt=prompt, prompt_tokens=tokens,
        max_output_tokens=max_output_tokens,
        arrival_offset_s=position / request_rate,
        dataset_source=source.name, original_sample_id=str(source_id or index), seed=seed,
        metadata={"source_row": index, "token_count_includes_chat_template": True},
    ) for position, (index, source_id, prompt, tokens) in enumerate(selected)]
    write_trace(output, requests)
    values = [row.prompt_tokens for row in requests]
    summary = {
        "category": category,
        "source": str(source.resolve()),
        "trace": str(output.resolve()),
        "seed": seed,
        "request_rate": request_rate,
        "max_output_tokens": max_output_tokens,
        "available_in_bucket": len(candidates),
        "number_of_prompts": len(values),
        "input_token_min": min(values),
        "input_token_max": max(values),
        "input_token_mean": statistics.mean(values),
        "input_token_p10": percentile(values, 10),
        "input_token_p50": percentile(values, 50),
        "input_token_p90": percentile(values, 90),
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", choices=("general", "code", "math"), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-30B-A3B")
    parser.add_argument("--minimum-tokens", type=int, default=64)
    parser.add_argument("--maximum-tokens", type=int, default=192)
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--request-rate", type=float, required=True)
    parser.add_argument("--max-output-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(build(
        category=args.category, source=args.source, output=args.output,
        tokenizer_name=args.tokenizer, minimum_tokens=args.minimum_tokens,
        maximum_tokens=args.maximum_tokens, count=args.count,
        request_rate=args.request_rate, max_output_tokens=args.max_output_tokens,
        seed=args.seed,
    ), indent=2))


if __name__ == "__main__":
    main()
