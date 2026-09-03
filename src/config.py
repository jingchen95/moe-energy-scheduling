"""Load the model configuration and build the vLLM server command."""

from __future__ import annotations

import argparse
import json
import os
import shlex
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected a YAML mapping in {path}")
    return value


def build_vllm_command(config: dict[str, Any], extra_args: list[str]) -> list[str]:
    """Return the command for the one supported vLLM deployment."""
    model = config["model"]
    parallel = config["parallel"]
    server = config["server"]

    command = [
        os.environ.get("PYTHON", "python"),
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(model["id"]),
        "--served-model-name",
        str(model["served_model_name"]),
        "--host",
        str(server["host"]),
        "--port",
        str(server["port"]),
        "--dtype",
        str(model["dtype"]),
        "--max-model-len",
        str(model["max_model_len"]),
        "--gpu-memory-utilization",
        str(model["gpu_memory_utilization"]),
        "--seed",
        str(model["seed"]),
        "--tensor-parallel-size",
        str(parallel["tensor_parallel_size"]),
        "--data-parallel-size",
        str(parallel["data_parallel_size"]),
        "--data-parallel-backend",
        str(parallel["data_parallel_backend"]),
        "--all2all-backend",
        str(parallel["all2all_backend"]),
    ]
    if model.get("revision"):
        command.extend(["--revision", str(model["revision"])])
    if model.get("trust_remote_code"):
        command.append("--trust-remote-code")
    if parallel.get("enable_expert_parallel"):
        command.append("--enable-expert-parallel")
    if parallel.get("expert_placement_strategy"):
        command.extend(
            ["--expert-placement-strategy", str(parallel["expert_placement_strategy"])]
        )
    if server.get("disable_log_requests"):
        command.append("--disable-log-requests")
    if server.get("enable_prefix_caching") is False:
        command.append("--no-enable-prefix-caching")
    command.extend(extra_args)
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--print-json", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("config", type=Path)
    parser.add_argument("extra", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    extra = args.extra[1:] if args.extra[:1] == ["--"] else args.extra
    command = build_vllm_command(load_yaml(args.config), extra)
    print(json.dumps(command) if args.print_json else shlex.join(command), flush=True)
    if args.execute:
        os.execvp(command[0], command)


if __name__ == "__main__":
    main()
