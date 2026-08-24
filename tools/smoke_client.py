#!/usr/bin/env python3
"""Send a deterministic, content-based smoke batch to an OpenAI endpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.request import Request, urlopen


def post_json(url: str, payload: dict[str, object]) -> dict[str, object]:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=300) as response:
        return json.load(response)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    prompts = [json.loads(line) for line in args.prompts.read_text().splitlines() if line.strip()]
    with args.output.open("x", encoding="utf-8") as stream:
        for item in prompts:
            response = post_json(
                f"{args.url.rstrip('/')}/v1/chat/completions",
                {
                    "model": args.model,
                    "messages": [{"role": "user", "content": item["prompt"]}],
                    "temperature": 0.0,
                    "seed": 0,
                    "max_tokens": 32,
                },
            )
            stream.write(json.dumps({"request": item, "response": response}) + "\n")
            stream.flush()


if __name__ == "__main__":
    main()
