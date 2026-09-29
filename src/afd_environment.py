"""Environment validation helpers for the isolated Milestone M AFD runtime."""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


_INSPECT_CODE = r"""
import importlib.metadata
import json
import pathlib
import sys
import sysconfig
import torch
import vllm

try:
    plugin_version = importlib.metadata.version("vllm-afd-plugin")
except importlib.metadata.PackageNotFoundError:
    plugin_version = None

print(json.dumps({
    "python_version": sys.version.replace("\\n", " "),
    "python_executable": str(pathlib.Path(sys.executable).absolute()),
    "python_base_prefix": str(pathlib.Path(sys.base_prefix).absolute()),
    "python_include": str(pathlib.Path(sysconfig.get_path("include")).absolute()),
    "python_h_available": pathlib.Path(sysconfig.get_path("include"), "Python.h").is_file(),
    "vllm_version": vllm.__version__,
    "vllm_root": str(pathlib.Path(vllm.__file__).resolve()),
    "torch_version": torch.__version__,
    "torch_root": str(pathlib.Path(torch.__file__).resolve()),
    "cuda_userspace_version": torch.version.cuda,
    "afd_plugin_version": plugin_version,
}))
"""


@dataclass(frozen=True)
class RuntimeMetadata:
    python_version: str
    python_executable: str
    python_base_prefix: str
    python_include: str
    python_h_available: bool
    vllm_version: str
    vllm_root: str
    torch_version: str
    torch_root: str
    cuda_userspace_version: str | None
    afd_plugin_version: str | None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RuntimeMetadata":
        return cls(**{field: value.get(field) for field in cls.__annotations__})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def reject_cuda_env_interpreter(interpreter: str | Path) -> Path:
    """Fail before an AFD command can accidentally use the protected env."""
    selected = Path(interpreter).expanduser().absolute()
    if "cuda-env" in selected.parts:
        raise ValueError(
            f"AFD interpreter must not come from protected cuda-env: {selected}"
        )
    return selected


def inspect_runtime(interpreter: str | Path) -> RuntimeMetadata:
    selected = Path(interpreter).expanduser().absolute()
    completed = subprocess.run(
        [str(selected), "-c", _INSPECT_CODE], capture_output=True, text=True
    )
    if completed.returncode:
        raise RuntimeError(
            f"runtime inspection failed for {selected}: {completed.stderr.strip()}"
        )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"runtime inspection returned no output: {selected}")
    return RuntimeMetadata.from_dict(json.loads(lines[-1]))


def validate_runtime_pair(
    old: RuntimeMetadata,
    afd: RuntimeMetadata,
    *,
    expected_plugin_commit: str,
    plugin_commit: str,
) -> None:
    if old.vllm_version != "0.15.1":
        raise ValueError(f"protected environment has vLLM {old.vllm_version}, expected 0.15.1")
    if afd.vllm_version != "0.26.0":
        raise ValueError(f"AFD environment has vLLM {afd.vllm_version}, expected 0.26.0")
    reject_cuda_env_interpreter(afd.python_executable)
    if Path(old.python_executable).absolute() == Path(afd.python_executable).absolute():
        raise ValueError("protected and AFD environments resolve to the same interpreter")
    if Path(old.vllm_root).resolve() == Path(afd.vllm_root).resolve():
        raise ValueError("protected and AFD environments resolve to the same vLLM package")
    if not afd.afd_plugin_version:
        raise ValueError("AFD plugin distribution is not installed in the AFD environment")
    if not afd.python_h_available:
        raise ValueError(
            f"AFD Python development header is unavailable under {afd.python_include}"
        )
    if plugin_commit != expected_plugin_commit:
        raise ValueError(
            f"AFD plugin commit {plugin_commit} does not match pin {expected_plugin_commit}"
        )


def environment_record(
    old: RuntimeMetadata,
    afd: RuntimeMetadata,
    *,
    plugin_repository: str,
    plugin_commit: str,
    plugin_clean: bool,
    uv_version: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "protected_cuda_env": old.to_dict(),
        "isolated_afd_env": afd.to_dict(),
        "afd_plugin": {
            "repository": plugin_repository,
            "commit": plugin_commit,
            "clean": plugin_clean,
        },
        "uv_version": uv_version,
    }
